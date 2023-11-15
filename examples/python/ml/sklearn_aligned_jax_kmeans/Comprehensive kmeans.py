import argparse
import json
import jax
from jax import random, jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import spu.binding.util.distributed as ppd
import os
import math

# os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/foo"
plt.switch_backend('agg')
seed = 20
key = random.PRNGKey(seed)
parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="/home/ndbtjay/Desktop/PPC/SecretFlow/spu/examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

def distance(data, centroids): #考虑并行化
    return jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)

def cluster_assignment(distances):
    return jnp.argmin(distances, axis=-1)

def update_centroids(data, cluster_assignments, k):
    new_centroids = jnp.zeros((k, data.shape[1]))
    # fori_loop模式
    def loop_for_update(cluster_idx, new_centroids):
        # 利用mask选出属于当前聚类的点，不会影响Array的形状。可能需要考虑一下没有属于该聚类的点的情况
        q = cluster_assignments == cluster_idx
        mask = q.astype(jnp.int32)
        # c为属于当前聚类的点的个数，s为它们对应点的坐标之和
        c = jnp.sum(mask)
        s = jnp.sum(data * mask[:, None], axis=0)
        new_centroids = new_centroids.at[cluster_idx].set(s/c)
        return new_centroids

    new_centroids = jax.lax.fori_loop(0, k, loop_for_update, new_centroids)
    return new_centroids
    # for循环模式
    # for cluster_idx in range(k):
    #     # 利用mask选出属于当前聚类的点，不会影响Array的形状。可能需要考虑一下没有属于该聚类的点的情况
    #     q = cluster_assignments == cluster_idx
    #     mask = q.astype(jnp.int32)
    #     # c为属于当前聚类的点的个数，s为它们对应点的坐标之和
    #     c = jnp.sum(mask)
    #     s = jnp.sum(data * mask[:, None], axis=0)
    #     new_centroids = new_centroids.at[cluster_idx].set(s/c)
    # return new_centroids

class KMeans:
    def __init__(self, n_points = 100, k = 3, max_epochs=50, early_stop_threshold=0.01, seed=None, comp_mode="full", init_mode="rand"):
        self.n_points = n_points
        self.k = k
        self.max_epochs = max_epochs
        self.early_stop_threshold = early_stop_threshold
        self.seed = seed
        self.comp_mode = comp_mode
        self.init_mode = init_mode

    # 普通初始化方法，随机重排后选择前k个点
    def init_centroids(self, data):
        temp = random.permutation(key, data, axis=0)
        return temp[:self.k]

    # kmeans++初始化方法，尽量选择与已选的点距离较远的点，目前只实现了jit上运行
    def init_centroids_kmeanspp(self, data):
        centroids = jnp.zeros((self.k, 2), dtype=jnp.int32)
        # mask_dist是用来标记已选择的点的，如果已选，置为0，方便算距离的时候将距离置为0
        mask_dist = jnp.ones(self.n_points, dtype=jnp.int32)
        c0_idx = random.randint(key, shape=(1,), minval=0, maxval=self.n_points - 1)
        centroids = centroids.at[0].set(data[c0_idx[0]])
        # 对于mask，将第一个选定点的索引置为0
        mask_dist = mask_dist.at[c0_idx[0]].set(0)
        
        for i in range(1, self.k):
            dist = jnp.sum((data[:, None, :] - centroids[None, :i, :]) ** 2, axis=-1)
            # 找出各点离已选的聚类中心的最短距离
            dist_min = jnp.min(dist, axis=1)
            # 利用mask_dist将已选的点的距离置为0
            dist_min = dist_min * mask_dist[:]
            # 在最短距离中找最大值，找到最远的点
            ci_idx = jnp.argmax(dist_min, axis=0)
            mask_dist = mask_dist.at[ci_idx].set(0)
            centroids = centroids.at[i].set(data[ci_idx])
        return centroids

    def fit(self, data):
        # 初始化聚类中心
        if self.init_mode == "rand":
            # 普通初始化
            centroids = self.init_centroids(data)
            # print("init_centroids1: ", centroids)
        elif self.init_mode == "kmeans++":
            # K-means++初始化
            centroids = self.init_centroids_kmeanspp(data)
            # print("init_centroids2: ", centroids)
        # 转换成float类型，与后面计算一致，避免报类型不同的错误
        centroids = centroids.astype(float)
        cluster_assignments = jnp.zeros(data.shape[0], dtype=jnp.int32)

        # 算距离，找归属，更新聚类中心
        # while_loop循环
        if self.comp_mode == "full": # 经典模式，计算距离平方和，比较距离大小
            def loop_step(stat):
                epochs, centroids, cluster_assignments, diff = stat
                distances = distance(data, centroids)
                cluster_assignments = cluster_assignment(distances)
                new_centroids = update_centroids(data, cluster_assignments, self.k)
                diff = jnp.sum(jnp.square((new_centroids - centroids)))
                return (epochs+1, new_centroids, cluster_assignments, diff)
            def end_cond(stat):
                epochs, centroids, cluster_assignments, diff = stat
                #return (epochs < self.max_epochs) & (diff > self.early_stop_threshold)
                return epochs < self.max_epochs
            (epochs, centroids, cluster_assignments, diff) = jax.lax.while_loop(end_cond,  loop_step, (0, centroids, cluster_assignments, 100))
            return centroids, cluster_assignments, epochs, diff

        # # for循环
        # if self.comp_mode == "full": # 经典模式，计算距离平方和，比较距离大小
        #     for epochs in range(self.max_epochs):
        #         distances = distance(data, centroids)
        #         cluster_assignments = cluster_assignment(distances)
        #         new_centroids = update_centroids(data, cluster_assignments, self.k)
        #         diff = jnp.sum(jnp.square((new_centroids - centroids)))
        #         centroids = new_centroids
        #     return centroids, cluster_assignments, epochs+1, diff

        # # elkan目前只实现了python上运行
        # elif self.comp_mode == "elkan": # elkan模式，利用三角不等式减少距离平方和的计算次数，但并行性差
        #     for i in range(self.max_epochs):
        #         # 计算各聚类中心之间距离的平方cen_dist的1/4，等下可以与各点与聚类中心距离的平方point_dist进行比较
        #         cen_dist = distance(centroids, centroids) / 4
        #         cluster_assignments = jnp.zeros(self.n_points)
        #         # cen_dist = jax.lax.div(cen_dist, 4)
        #         # 如果point_dist < 1/4 cen_dist，说明2a<c，则b>a，则不需要进一步计算b的距离
        #         for j in range(self.n_points):
        #             # 先计算出点与第一个聚类的距离
        #             min_dist = jnp.sum((data[j, None, :] - centroids[None, 0, :]) ** 2, axis=-1)[0]
        #             min_label = 0

        #             for l in range(1, self.k):
        #                 # 如果point_dist > 1/4 cen_dist，说明需要进一步计算b，否则不继续计算
        #                 if min_dist > cen_dist[min_label, l]:
        #                     this_dist = jnp.sum((data[j, None, :] - centroids[None, l, :]) ** 2, axis=-1)[0]
        #                     if this_dist < min_dist:
        #                         min_dist = this_dist
        #                         min_label = l
        #             cluster_assignments = cluster_assignments.at[j].set(min_label)
        #         centroids = update_centroids(data, cluster_assignments, self.k)
        #         print(i)
        #     return centroids, cluster_assignments

# Bisecting-kmeans，目前只实现了python上运行
# 实现难点在于需要将SSE最大的聚类提取出来，这涉及到动态切片之类的操作，难以在jit和SPU上实现
class Bisecting_KMeans:
    def __init__(self, n_points = 100, k = 3, max_epochs=50, early_stop_threshold=0.01, seed=None, comp_mode="full", init_mode="rand"):
        self.n_points = n_points
        self.k = k
        self.max_epochs = max_epochs
        self.early_stop_threshold = early_stop_threshold
        self.seed = seed
        self.bicount = 0
        self.comp_mode = comp_mode
        self.init_mode = init_mode

    def init_centroids(self, data):
        centroids = jnp.zeros((self.k, 2))
        cluster_assignments = jnp.zeros(shape=(self.n_points), dtype=jnp.int32)
        c0_idx = random.randint(key, shape=(1,), minval=0, maxval=self.n_points - 1)
        centroids = centroids.at[0].set(data[c0_idx[0]])
        self.bicount = 1
        return centroids, cluster_assignments

    def fit(self, data):
        # 初始化第一批点
        centroids, cluster_assignments = self.init_centroids(data)
        tmp = data
        while self.bicount <= self.k:
            ## 找到内部SSE最大的，作为要切分的部分
            SSE = jnp.zeros(self.bicount)
            for i in range(self.bicount):
                # 提取出属于当前聚类的点作为tmpdata
                tmpdata = data[cluster_assignments == i]
                # mask = cluster_assignments == i
                # tmpdata = jnp.zeros(shape = (self.n_points, 2))
                # k = 0
                # for j in range(self.n_points):
                #     if mask[i] == 1:
                #         tmpdata = tmpdata.at[k].set(data[j])
                #         k += 1

                tmpSSE = jnp.sum((tmpdata[:] - centroids[i]) ** 2)
                SSE = SSE.at[i].set(tmpSSE)
            # SSE = jnp.sum(distance(data, centroids[:self.bicount]), axis=0)
            idx = jnp.argmax(SSE, axis=0)
            
            tmp_data = data[cluster_assignments == idx] # 取出SSE最大部分的数据
            print("tmp_data: ", len(tmp_data))
            # print(tmp_data)
            split_kmeans = KMeans(len(tmp_data), 2)
            tmp_centroids, tmp_cluster_assignments, epochs, diff = split_kmeans.fit(tmp_data)
            print("tmp_cluster_assignments: ", len(tmp_cluster_assignments))
            draw(2, tmp_data, tmp_cluster_assignments, tmp_centroids, str(self.bicount))
            centroids = centroids.at[idx].set(tmp_centroids[0])
            centroids = centroids.at[self.bicount].set(tmp_centroids[1])# 多出来的一个分类放到最后
            j = 0
            k = 0
            for c in cluster_assignments:
                if c == idx:
                    if tmp_cluster_assignments[k] == 0:
                        cluster_assignments = cluster_assignments.at[j].set(idx)
                    else:
                        cluster_assignments = cluster_assignments.at[j].set(self.bicount)
                    k += 1
                j += 1
            self.bicount += 1
        # 此处epochs和diff无意义，只是为了接口统一
        return centroids, cluster_assignments, epochs, diff

class MiniBatch_Kmeans:
    def __init__(self, n_points = 100, k = 3, max_epochs=50, batch_size=1000, early_stop_threshold=0.01, seed=None, comp_mode="full", init_mode="rand"):
        self.n_points = n_points
        self.k = k
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stop_threshold = early_stop_threshold
        self.seed = seed
        self.comp_mode = comp_mode
        self.init_mode = init_mode
        self.key = jax.random.PRNGKey(seed)
    
    # 普通初始化方法，随机重排后选择前k个点
    def init_centroids(self, data):
        temp = random.permutation(key, data, axis=0)
        return temp[:self.k]

    # kmeans++初始化方法，尽量选择与已选的点距离较远的点，目前只实现了jit上运行
    def init_centroids_kmeanspp(self, data):
        centroids = jnp.zeros((self.k, 2), dtype=jnp.int32)
        # mask_dist是用来标记已选择的点的，如果已选，置为0，方便算距离的时候将距离置为0
        mask_dist = jnp.ones(self.n_points, dtype=jnp.int32)
        c0_idx = random.randint(key, shape=(1,), minval=0, maxval=self.n_points - 1)
        centroids = centroids.at[0].set(data[c0_idx[0]])
        # 对于mask，将第一个选定点的索引置为0
        mask_dist = mask_dist.at[c0_idx[0]].set(0)
        
        for i in range(1, self.k):
            dist = jnp.sum((data[:, None, :] - centroids[None, :i, :]) ** 2, axis=-1)
            # 找出各点离已选的聚类中心的最短距离
            dist_min = jnp.min(dist, axis=1)
            # 利用mask_dist将已选的点的距离置为0
            dist_min = dist_min * mask_dist[:]
            # 在最短距离中找最大值，找到最远的点
            ci_idx = jnp.argmax(dist_min, axis=0)
            mask_dist = mask_dist.at[ci_idx].set(0)
            centroids = centroids.at[i].set(data[ci_idx])
        return centroids

    def fit(self, data):
        # 初始化聚类中心
        if self.init_mode == "rand":
            # 普通初始化
            centroids = self.init_centroids(data)
            # print("init_centroids1: ", centroids)
        elif self.init_mode == "kmeans++":
            # K-means++初始化
            centroids = self.init_centroids_kmeanspp(data)
            # print("init_centroids2: ", centroids)
        # 转换成float类型，与后面计算一致，避免报类型不同的错误
        centroids = centroids.astype(float)
        cluster_assignments = jnp.zeros(data.shape[0], dtype=jnp.int32)

        # while_loop形式
        def loop_step(stat):
            (epochs, centroids, cluster_assignments, diff, key) = stat
            key, subkey = jax.random.split(key, 2)
            # 随机选择batch_size大小的样本
            batch = jax.random.choice(key=subkey, a=data, shape=(self.batch_size,), replace=False)
            distances = distance(batch, centroids)
            cluster_assignments = cluster_assignment(distances)
            new_centroids = update_centroids(batch, cluster_assignments, self.k)
            diff = jnp.sum(jnp.square((new_centroids - centroids)))
            return (epochs+1, new_centroids, cluster_assignments, diff, key)
        def end_cond(stat):
            (epochs, centroids, cluster_assignments, diff, key) = stat
            #return (epochs < self.max_epochs) & (diff > self.early_stop_threshold)
            return epochs < self.max_epochs
        (epochs, centroids, cluster_assignments, diff, _) = jax.lax.while_loop(end_cond, loop_step, (0, centroids, cluster_assignments, 100, self.key))
        distances = distance(data, centroids)
        cluster_assignments = cluster_assignment(distances)
        return centroids, cluster_assignments, epochs, diff

        # # for循环形式
        # for epochs in range(self.max_epochs):
        #     self.key, subkey = jax.random.split(self.key)
        #     batch = jax.random.choice(key=subkey, a=data, shape=(self.batch_size,), replace=False)
        #     distances = distance(batch, centroids)
        #     cluster_assignments = cluster_assignment(distances)
        #     new_centroids = update_centroids(batch, cluster_assignments, self.k)
        #     # 可以判断new_centroids的变化量是否小于threshold，如果是则提前结束循环
        #     diff = jnp.sum(jnp.square((new_centroids - centroids)))
        #     centroids = new_centroids
        # # 用centroids计算归属，得到最终结果
        # distances = distance(data, centroids)
        # cluster_assignments = cluster_assignment(distances)
        # return centroids, cluster_assignments, epochs+1, diff

# 画图并保存图片
def draw(k, data, cluster_assignments, centroids, file_name):
    col = ['HotPink', 'Aqua', 'Chartreuse', 'yellow', 'LightSalmon']
    for i in range(k):
        # 选择属于该聚类的点， 需要考虑一下没有属于该聚类的点的情况
        points = data[cluster_assignments == i]
        for p in points:
            plt.scatter(p[0], p[1], s=10, color=col[i])
        plt.scatter(centroids[i][0], centroids[i][1], s=25, color='Black')
    # plt.show()
    plt.savefig(file_name + ".jpg")
    plt.clf()

def get_data(data):
    return data

def create_data(n_points=100):
    # 生成二维随机坐标
    # arr是一个数组，每个元素都是一个二元组，代表着一个坐标
    # arr形如：[ (x1, y1), (x2, y2), (x3, y3) ... ]
    # 随机选取数据
    return(random.randint(key, shape=(n_points, 2), minval=1, maxval=100))

def create_data_split(n_points=1000, k=3, r=50):
    # n_points表示点的个数
    # k表示聚类个数
    # r表示生成随机数的方框长度
    # 先随机生成k个中心
    cen = random.randint(key, shape=(k, 2), minval=1+r, maxval=1000-r)
    print(cen)
    data = jnp.zeros(shape=(n_points, 2))
    ran = random.randint(key, shape=(n_points, 2), minval=1, maxval=r)
    # 将n_points分成k份
    step = math.ceil(n_points / k)
    for i in range(k):
        for j in range(step):
            data = data.at[i*step+j].set(cen[i]+ran[i*step+j])
        if i == k-1:
            for j in range(n_points % step):
                data = data.at[(i+1)*step+j].set(cen[(i+1)]+ran[i*step+j])
    return data

def run_on_cpu(data, n_points = 100, k = 3, iter = 50, comp_mode = "full", init_mode = "kmeans++"):
    kmeans = KMeans(n_points, k, iter, comp_mode="full", init_mode=init_mode)
    # kmeans = Bisecting_KMeans(n_points, k, iter, comp_mode="full", init_mode=init_mode)
    # kmeans = MiniBatch_Kmeans(n_points, k, iter, seed=seed, comp_mode="full", init_mode=init_mode)

    # jit模式
    final_centroids, final_cluster_assignments, epochs, diff = jit(kmeans.fit)(data)

    # python模式，可以看到内部值
    # final_centroids, final_cluster_assignments, epochs, diff = kmeans.fit(data)
    print(final_centroids)
    print(final_cluster_assignments)
    print("end at ", epochs, " epochs")
    print("diff: ", diff)
    draw(k, data, final_cluster_assignments, final_centroids, str(1))


def run_on_spu(data1, data2, n_points = 100, k = 3, iter = 50, comp_mode = "full", init_mode = "kmeans++"):
    def cluster(data1, data2):
        data = jnp.concatenate((data1, data2), axis=0) # 横向联邦
        kmeans = KMeans(n_points, k, iter, comp_mode=comp_mode, init_mode=init_mode)
        # kmeans = Bisecting_KMeans(n_points, k, iter, comp_mode="full", init_mode=init_mode)
        # kmeans = MiniBatch_Kmeans(n_points, k, iter, seed=seed, comp_mode="full", init_mode=init_mode)
        return data, kmeans.fit(data)

    data1 = ppd.device("P1")(get_data)(data1)
    data2 = ppd.device("P2")(get_data)(data2)
    data, (final_centroids_s, final_cluster_assignments_s, epochs, diff) = ppd.device("SPU")(cluster)(data1, data2)
    final_centroids = ppd.get(final_centroids_s)
    final_cluster_assignments = ppd.get(final_cluster_assignments_s)
    data = ppd.get(data)
    epochs = ppd.get(epochs)
    diff = ppd.get(diff)
    print(final_centroids)
    print(final_cluster_assignments)
    print("end at ", epochs, " epochs")
    print("diff: ", diff)
    draw(k, data, final_cluster_assignments, final_centroids, str(2))

if __name__=="__main__":
    n_points = 1000
    k = 5
    r = 50
    iter = 50
    # data = create_data(n_points)
    data = create_data_split(n_points, k, r)
    data1 = data[:int(n_points/2)]
    data2 = data[int(n_points/2):]
    print('Run on CPU\n------\n')
    run_on_cpu(data, n_points, k, iter)
    print('Run on SPU\n------\n')
    run_on_spu(data1, data2, n_points, k, iter)
