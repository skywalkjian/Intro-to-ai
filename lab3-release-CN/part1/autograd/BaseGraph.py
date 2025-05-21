from .BaseNode import *
from typing import List
import numpy as np
class Graph(List):
    '''
    计算图类
    '''
    def __init__(self, nodes: List[Node]):
        super().__init__()
        for node in nodes:
            self.append(node) # 按照前向传播的顺序添加节点。注意，self本身即为存储节点的list。

    def eval(self):
        for node in self:
            node.eval()
    
    def train(self):
        for node in self:
            node.train()

    def flush(self):        
        for node in self:
            node.flush()

    def forward(self, X, debug=False, removelossnode: int = 0):
        """
        正向传播
        @param X: n*d 输入样本
        @param debug: 用于debug, print输入和输出数据的shape
        @param removelossnode: 训练时设为0, 测试时设为1, 不使用最后的loss节点
        @return: 计算图中各个节点的输出
        """
        ret = []
        if debug:
            print("forward debug start")
        if removelossnode > 0:
            nlist = self[:-removelossnode]
        else:
            nlist = self
        for n in nlist:
            X = n.forward(X, debug)
            ret.append(X)
        if debug:
            print("forward debug end")
        return ret

    def backward(self, grad=1.0, debug=False):
        """
        反向传播
        @param grad: 1, 从最后一层开始反传的梯度值
        @param debug: 用于debug, print上游和下游梯度的shape
        @return: 反传结束得到的梯度（损失函数对输入的偏导）
        """
        if debug:
            print("backward debug start")
            
        # TODO: YOUR CODE HERE
        for n in reversed(self):
            grad = n.backward(grad, debug)
        #上传时候需要删掉
            
        if debug:
            print("backward debug end")
        return grad
        
        raise NotImplementedError
    
    def optimstep(self, lr, wd1, wd2):
        """
        利用计算好的梯度对参数进行更新
        @param lr: 超参数，学习率
        @param wd1: 超参数, L1正则化。选做，可不实现。
        @param wd2: 超参数, L2正则化
        @return: 不需要返回值
        """  
        # TODO: YOUR CODE HERE
        for n in self:
            for i, param in enumerate(n.params):
                #param=np.expand_dims(param, axis=0)
                n.params[i] =param- lr * (n.grad[i] + 2 * wd2 * param + wd1 * np.sign(param))
                #param=np.squeeze(param, axis=0)
                
                #print(n.grad[i][0])
        return
        #上传时候需要删掉
        #raise NotImplementedError

    def parameters(self):
        """
        返回当前计算图中的所有节点的参数
        @return:
        """
        ret = []
        for n in self:
            for param in n.params:
                ret.append(param)
        return ret

    def grads(self):
        """
        返回当前计算图中的所有节点的梯度
        @return:
        """
        ret = []
        for n in self:
            for grad in n.grad:
                ret.append(grad)
        return ret