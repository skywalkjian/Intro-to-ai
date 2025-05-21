from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 10     # 树的数量
ratio_data = 0.8   # 采样的数据比例
ratio_feat = 0.5 # 采样的特征比例
hyperparams = {
    "depth":10, 
    "purity_bound":1e-3,
    "gainfunc": gain
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    n, d = X.shape
    trees = []
    for _ in range(num_tree):
        ssize = int(n * ratio_data)
        indices = np.random.choice(n,ssize,replace=True)
        Xs = X[indices]
        Ys = Y[indices]
        
        # 特征扰动 - 随机选择部分特征
        feat_size = int(d*ratio_feat)
        feat_indices = np.random.choice(d,feat_size,replace=False)
        unused = list(feat_indices)
        
        # 构建决策树
        tree = buildTree(Xs,Ys,unused,hyperparams["depth"],hyperparams["purity_bound"],hyperparams["gainfunc"])
        trees.append(tree)
        
    return trees

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
