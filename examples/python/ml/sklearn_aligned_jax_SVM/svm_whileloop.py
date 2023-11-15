import jax
import jaxlib
from jax import lax
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random, device_put
from sklearn.datasets import make_classification


#定义RBF核函数
@jit
def rbf_kernel(X1, X2, gamma = 1.0):
    return np.exp(-gamma * np.sum((X1 - X2) ** 2))

@jit
def initialize_params(X, Y, C=1.0, tol=1e-3, max_passes=5):
    m, n = X.shape
    alphas = np.zeros(m)
    b = 0.0
    E = np.zeros(m)
    passes = 0
    eta = 0
    L = 0
    H = 0
    kernel = np.zeros((m, m))
    #计算所有训练样本的核函数
    for i in range(m):
        for j in range(m):
            # kernel[i, j] = rbf_kernel(X[i], X[j])
            kernel = kernel.at[i, j].set(rbf_kernel(X[i], X[j]))
    return alphas, b, E, passes, eta, L, H, kernel


#定义优化的目标函数
def objective(alphas, Y, kernel):
    return np.sum(alphas) - 0.5 * np.sum(Y * Y * kernel * alphas * alphas)

@jit
#计算误差
def compute_error(i, alphas, b, Y, kernel):
    return np.dot((alphas * Y), kernel[:, i]) + b - Y[i]

@jit
#使用SMO进行优化
def smo_optimize(X, Y, alphas, b, E, passes, eta, L, H, kernel, C=1.0, tol=1e-3, max_passes=5):
    m, n = X.shape

    def condition(state):
        alphas, b, E, passes, eta, L, H, kernel = state
        return passes < max_passes

    def body(state):
        alphas, b, E, passes, eta, L, H, kernel = state
        num_changed_alphas = 0
        for i in range(m):
            E = E.at[i].set(compute_error(i, alphas, b, Y, kernel))
            # E[i] = compute_error(i, alphas, b, Y, kernel)
            if ((Y[i] * E[i] < -tol and alphas[i] < C) or (Y[i] * E[i] > tol and alphas[i] > 0)):
                j = random.randint(random.PRNGKey(i), (), 0, m - 1)
                while j == i:
                    j = random.randint(random.PRNGKey(i), (), 0, m - 1)
                E = E.at[j].set(compute_error(j, alphas, b, Y, kernel))
                # E[j] = compute_error(j, alphas, b, Y, kernel)
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]
                if Y[i] == Y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                if L == H:
                    continue
                eta = 2 * kernel[i, j] - kernel[i, i] - kernel[j, j]
                if eta >= 0:
                    continue
                alphas = alphas.at[j].set(alphas[j] - Y[j] * (E[i] - E[j]) / eta)
                alphas = alphas.at[j].set(min(H, alphas[j]))
                alphas = alphas.at[j].set(max(L, alphas[j]))
                # alphas[j] -= Y[j] * (E[i] - E[j]) / eta
                # alphas[j] = min(H, alphas[j])
                # alphas[j] = max(L, alphas[j])
                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas = alphas.at[j].set(alpha_j_old)
                    # alphas[j] = alpha_j_old
                    continue
                alphas = alphas.at[i].set(alphas[i] + Y[i] * Y[j] * (alpha_j_old - alphas[j]))
                # alphas[i] += Y[i]*Y[j]*(alpha_j_old - alphas[j])
                b1 = b - E[i] - Y[i] * (alphas[i] - alpha_i_old) * kernel[i, i] - Y[j] * (alphas[j] - alpha_j_old) * kernel[
                    i, j]
                b2 = b - E[j] - Y[i] * (alphas[i] - alpha_i_old) * kernel[i, j] - Y[j] * (alphas[j] - alpha_j_old) * kernel[
                    j, j]
                if 0 < alphas[i] and alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
        return alphas, b, E, passes, eta, L, H, kernel

    initial_state = alphas, b, E, passes, eta, L, H, kernel
    alphas, b, E, passes, eta, L, H, kernel = lax.while_loop(condition, body, initial_state)

    return alphas, b

@jit
#根据参数，构建最终的决策函数
def decision_function(alphas, b, X, Y, X_sample):
    result = b
    for i in range(len(X)):
        result += alphas[i] * Y[i] * rbf_kernel(X[i], X_sample)
    return result

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind


print(f"jax version: {jax.__version__}")
print(f"jaxlib version: {jaxlib.__version__}")
print(f"Found {num_devices} JAX devices of type {device_type}.")

device = jax.devices('gpu')[0]

X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=42)
#将标签转化为1和-1
Y = np.where(y == 0, -1, y)
X = device_put(X)
Y = device_put(Y)

# alphas, b, E, passes, eta, L, H, kernel = initialize_params(X, Y)
alphas, b, E, passes, eta, L, H, kernel = device_put(initialize_params(X, Y))

alphas, b = smo_optimize(X, Y, alphas, b, E, passes, eta, L, H, kernel)
predictions = np.sign([decision_function(alphas, b, X, Y, X[i]) for i in range(len(X))])

accuracy = np.mean(predictions == Y)
print(f'Accuracy: {accuracy * 100:.2f}%')
