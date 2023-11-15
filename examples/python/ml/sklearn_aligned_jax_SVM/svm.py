# bazel run -c opt //examples/python/utils:nodectl -- up
# bazel run -c opt //examples/python/ml/jax_svm:jax_svm_sklearn

import argparse
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import jax.lax as lax

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import examples.python.utils.dataset_utils as dsutil
import spu.utils.distributed as ppd


class SVM:
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
        n_epochs=1000,
        step_size=0.001,
        lambda_param=0.01,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.alpha_ = None
        self.bias_ = None

        self.n_epochs = n_epochs
        self.step_size = step_size
        self.lambda_param = lambda_param

    def decision_function(self, X):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = jnp.zeros(n_features)
        b = 0
        xs = jnp.array(X)
        ys = jnp.array(y)

        def epoch_loop(_, args):
            def sample_loop(i, args):
                wj, bj = args
                x = xs[i]
                y = ys[i]
                condition = y * (jnp.dot(x, wj) - bj) >= 1
                delta_w = jnp.where(
                    condition,
                    self.step_size * (2 * self.lambda_param * wj),
                    self.step_size * (2 * self.lambda_param * wj - jnp.dot(x, y)),
                )
                delta_b = jnp.where(condition, 0, self.step_size * y)
                wj = -delta_w
                bj = -delta_b
                return wj, bj

            wi, bi = args
            return jax.lax.fori_loop(0, n_samples, sample_loop, (wi, bi))

        return jax.lax.fori_loop(0, self.n_epochs, epoch_loop, (w, b))

    def get_metadata_routing(self):
        pass

    def get_params(self, deep=True):
        pass

    def predict(self, X):
        approx = jnp.dot(X, self.alpha_) - self.bias_
        return jnp.sign(approx)

    def predict_log_proba(self, X):
        pass

    def predict_proba(self, X):
        pass

    def score(self, X, y, sample_weight=None):
        pass

    def set_fit_request(self, *, sample_weight=None):
        pass

    def set_params(self, **params):
        pass

    def set_score_request(self, *, sample_weight=None):
        pass


parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
parser.add_argument("--n_epochs", default=1000, type=int)
parser.add_argument("--step_size", default=0.001, type=float)
parser.add_argument("--lambda_param", default=0.01, type=float)
args = parser.parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def predict(x, w, b):
    approx = jnp.dot(x, w) - b
    return jnp.sign(approx)


# 生成数据
point_num = 100
train_point = round(point_num * 0.8)
test_point = point_num - train_point
n_features = 5
# 两个簇
centers = [[2, 2], [8, 2]]
X_data, y_data = datasets.make_blobs(
    n_samples=point_num, n_features=n_features, centers=centers, cluster_std=1
)

# 四个簇
# centers = [[2, 2], [8, 2], [2, 8], [8, 8]]
# X_data, y_data = datasets.make_blobs(
#     n_samples=point_num, n_features=n_features, centers=centers, cluster_std=1
# )

# 月牙，二维
# X_data, y_data = datasets.make_moons(n_samples=point_num, noise=0.1, random_state=0)

# 同心圆，二维
# X_data, y_data = datasets.make_circles(n_samples=point_num, noise=0.04, factor=0.7)

# 四维同心圆
# 生成二维的线性不可分数据
# X_data, y_data = datasets.make_circles(n_samples=point_num, noise=0.05, random_state=1)
# 添加两个噪声特征
# noise = np.random.rand(point_num, n_features-2)
# X_data = np.hstack((X_data, noise))

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)
# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def run_on_cpu(X_train, X_test, y_train, y_test):
    svm = SVM(
        n_epochs=args.n_epochs, step_size=args.step_size, lambda_param=args.lambda_param
    )
    y_train = jnp.where(y_train <= 0, -1, 1)
    print(X_train.shape)
    w, b = jax.jit(svm.fit)(X_train, y_train)
    print(w, b)

    y_test = jnp.where(y_test <= 0, -1, 1)
    score = metrics.accuracy_score(y_test, predict(X_test, w, b))
    print("AUC(cpu)={}".format(score))
    # 测试精确率
    score = metrics.precision_score(y_test, predict(X_test, w, b))
    print("Precision(cpu)={}".format(score))
    # 测试召回率
    score = metrics.recall_score(y_test, predict(X_test, w, b))
    print("Recall(cpu)={}".format(score))


def run_on_spu(X_train, X_test, y_train, y_test):
    @ppd.device("SPU")
    def train(x1, x2, y):
        x = jnp.concatenate((x1, x2), axis=1)
        svm = SVM(args.n_epochs, args.step_size, args.lambda_param)
        y = jnp.where(y <= 0, -1, 1)
        return svm.fit(x, y)

    y_train = jnp.where(y_train <= 0, -1, 1)
    x1 = ppd.device("P1")(lambda x: x)(X_train[:, : (n_features // 2)])
    y = ppd.device("P1")(lambda x: x)(y_train[:train_point])
    x2 = ppd.device("P2")(lambda x: x)(X_train[:, (n_features // 2) :])
    w, b = train(x1, x2, y)
    w_r, b_r = ppd.get(w), ppd.get(b)
    print(w_r, b_r)

    y_test = jnp.where(y_test <= 0, -1, 1)
    # 测试准确率
    score = metrics.accuracy_score(y_test, predict(X_test, w_r, b_r))
    print("AUC(spu)={}".format(score))
    # 测试精确率
    score = metrics.precision_score(y_test, predict(X_test, w_r, b_r))
    print("Precision(spu)={}".format(score))
    # 测试召回率
    score = metrics.recall_score(y_test, predict(X_test, w_r, b_r))
    print("Recall(spu)={}".format(score))


if __name__ == "__main__":
    # 测量CPU时间
    start = time.time()
    print("Run on CPU\n------\n")
    run_on_cpu(X_train, X_test, y_train, y_test)
    end = time.time()
    print("CPU Time cost: {}s".format(end - start))
    # 测量SPU时间
    start = time.time()
    print("Run on SPU\n------\n")
    run_on_spu(X_train, X_test, y_train, y_test)
    end = time.time()
    print("SPU Time cost: {}s".format(end - start))
