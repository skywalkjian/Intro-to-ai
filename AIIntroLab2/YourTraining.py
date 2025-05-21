'''
Softmax 回归。计算accuracy。
'''
from answerMultiLayerPerceptron import buildGraph, lr, wd1, wd2, batchsize
import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed
from scipy.ndimage import rotate, shift # 导入用于数据增强的库

from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import * # 确保导入所有需要的节点

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/My.npy" 

# --- 数据增强函数 ---
def augment_data(images, labels, rotation_range=10, shift_range=2):
    """对图像数据进行增强"""
    augmented_images = []
    augmented_labels = []

    for img, lbl in zip(images, labels):
        # 原始图像
        augmented_images.append(img)
        augmented_labels.append(lbl)

        # 增强后的图像
        img_2d = img.reshape(28, 28)

        angle = np.random.uniform(-rotation_range, rotation_range)
        rotated_img = rotate(img_2d, angle, reshape=False, mode='nearest')
        augmented_images.append(rotated_img.flatten())
        augmented_labels.append(lbl) # 标签不变

    return np.array(augmented_images), np.array(augmented_labels)

# --- 加载并合并训练集和验证集 ---
X_train_orig = mnist.trn_X
Y_train_orig = mnist.trn_Y
X_val_orig = mnist.val_X
Y_val_orig = mnist.val_Y

X_combined_orig = np.concatenate((X_train_orig, X_val_orig), axis=0)
Y_combined_orig = np.concatenate((Y_train_orig, Y_val_orig), axis=0)

X, Y = augment_data(X_combined_orig, Y_combined_orig)
# TODO: You can change the hyperparameters here
lr = 5e-4  # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-5# L2正则化
batchsize = 128
epochs = 30 

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    nodes = [BatchNorm(mnist.num_feat),
            Linear(mnist.num_feat,mnist.num_feat//2),BatchNorm(mnist.num_feat//2),relu(),Dropout(0.2),
            Linear(mnist.num_feat//2,mnist.num_feat//4),BatchNorm(mnist.num_feat//4),relu(),Dropout(0.5),
            Linear(mnist.num_feat//4,mnist.num_feat//8),BatchNorm(mnist.num_feat//8),relu(),Dropout(0.5),
            Linear(mnist.num_feat//8, mnist.num_class), LogSoftmax(), NLLLoss(Y)]
    graph=Graph(nodes)
    return graph

if __name__ == "__main__":
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, epochs + 1): 
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i}/{epochs} loss {loss:.3e} train_acc_on_combined_augmented {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)

    with open(save_path, "rb") as f:
        graph = pickle.load(f)
    graph.eval()
    graph.flush()
    pred = graph.forward(mnist.val_X, removelossnode=1)[-1]
    haty = np.argmax(pred, axis=1)
    print("valid acc", np.average(haty==mnist.val_Y))

