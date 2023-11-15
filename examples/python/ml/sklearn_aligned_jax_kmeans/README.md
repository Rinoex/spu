在Jax平台上初步实现了对应于Sklearn库已封装的Kmeans、Kmeans++、Mini_batch算法，并初步确保部分算法能在SPU上运行。

Bisecting-Kmeans无法动态切片以计算SSE的问题仍需解决。

本算法主要在算法层实现隐私计算，对涉及可能暴露数据形状的策略进行了平衡，最小化暴露信息。

![kmeans2.jpg](https://s2.loli.net/2023/11/15/rc7jbmQau4lkKEV.jpg)

参考文献：
Efficient Privacy-Preserving K-Means Clustering from Secret-Sharing-Based Secure Three-Party Computation
