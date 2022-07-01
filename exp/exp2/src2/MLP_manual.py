import torch
import numpy as np
from matplotlib import pyplot as plt
import math
import sys

def tanh_vec(x):
    '''
    x: 一个一维的ndarray
    对x的每个分量xi, 计算yi = tanh(xi)
    返回y
    '''
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = math.tanh(x[i])
    return y

def softmax(x):
    denominator = 0.0
    for xi in x:
        denominator += math.exp(xi)
    y = np.empty(x.shape)
    for i in range(len(y)):
        y[i] = math.exp(x[i]) / denominator
    return y

class MLP:
    def __init__(self):
        # layer size = [10, 8, 8, 4]
        # 初始化所需参数   
        self.x_size = 10
        self.h1_size = 10
        self.h2_size = 8
        self.h3_size = 8
        self.y_size = 4
        self.W1 = np.random.rand(self.h1_size, self.x_size)
        self.W2 = np.random.rand(self.h2_size, self.h1_size)
        self.W3 = np.random.rand(self.h3_size, self.h2_size)
        self.W4 = np.random.rand(self.y_size, self.h3_size)
        self.b1 = np.random.rand(self.h1_size)
        self.b2 = np.random.rand(self.h2_size)
        self.b3 = np.random.rand(self.h3_size)
        self.b4 = np.random.rand(self.y_size)
        self.s1 = tanh_vec
        self.s2 = tanh_vec
        self.s3 = tanh_vec
        self.s4 = softmax
    
    def forward(self, x):
        # 前向传播
        '''
        输入: x: 一个样本
        输出: forward_pack: 各层的计算结果
        '''
        forward_pack = {}
        h1 = np.zeros(self.h1_size)
        h2 = np.zeros(self.h2_size)
        h3 = np.zeros(self.h3_size)
        y_predict = np.zeros(self.y_size)
        
        h1 = self.s1(np.dot(self.W1, x) + self.b1)
        h2 = self.s2(np.dot(self.W2, h1) + self.b2)
        h3 = self.s3(np.dot(self.W3, h2) + self.b3)
        y_predict = self.s4(np.dot(self.W4, h3) + self.b4)
    
        forward_pack['h1'] = h1
        forward_pack['h2'] = h2
        forward_pack['h3'] = h3
        forward_pack['y_predict'] = y_predict
        return forward_pack

    def backward(self, x, forward_pack, label, lr): # 自行确定参数表
        # 反向传播
        '''
        输入: 一个样本的各层计算结果和实际label, 以及学习率lr
        计算Wi和bi的梯度, 然后做一次梯度下降调整参数
        '''
        t = np.argmax(label)
        h1 = forward_pack['h1']
        h2 = forward_pack['h2']
        h3 = forward_pack['h3']
        y_predict = forward_pack['y_predict']
        loss = -math.log(y_predict[t])
        # print('    loss: %f'%loss)
        
        # TODO: pytorch求导
        
        
        # 手动计算梯度
        gs1 = np.zeros(h1.shape)
        gs2 = np.zeros(h2.shape)
        gs3 = np.zeros(h3.shape)
        for i in range(len(gs1)):
            gs1[i] = 1 - h1[i] ** 2
        for i in range(len(gs2)):
            gs2[i] = 1 - h2[i] ** 2
        for i in range(len(gs3)):
            gs3[i] = 1 - h3[i] ** 2
        gb4 = np.zeros(label.shape) # gb4 stands for gradient of b4, equals to l's4'
        for i in range(len(label)):
            if i == t:
                gb4[i] = y_predict[i] - 1
            else:
                gb4[i] = y_predict[i]
        gb4 = gb4.reshape(-1, 1) # 将数组转化成列向量
        gW4 = np.dot(gb4, h3.reshape(1, -1)) # 注意h3转化成行向量
        gb3 = np.dot(self.W4.T, gb4) * gs3.reshape(-1, 1)
        gW3 = np.dot(gb3, h2.reshape(1, -1))
        gb2 = np.dot(self.W3.T, gb3) * gs2.reshape(-1, 1)
        gW2 = np.dot(gb2, h1.reshape(1, -1))
        gb1 = np.dot(self.W2.T, gb2) * gs1.reshape(-1, 1)
        gW1 = np.dot(gb1, x.reshape(1, -1))
        
        # 梯度下降调整Wi和bi
        self.W1 -= lr * gW1
        self.b1 -= lr * gb1.reshape(-1)
        self.W2 -= lr * gW2
        self.b2 -= lr * gb2.reshape(-1)
        self.W3 -= lr * gW3
        self.b3 -= lr * gb3.reshape(-1)
        self.W4 -= lr * gW4
        self.b4 -= lr * gb4.reshape(-1)
        
        return loss 
    
    def print_params(self):
        print('W1:')
        print(self.W1)
        print('W2:')
        print(self.W2)
        print('W3:')
        print(self.W3)
        print('W4:')
        print(self.W4)
        print('b1:')
        print(self.b1)
        print('b2:')
        print(self.b2)
        print('b3:')
        print(self.b3)
        print('b4:')
        print(self.b4)
  
def train(mlp: MLP, epochs, lr, inputs, labels):
    '''
        mlp: 传入实例化的MLP模型
        epochs: 训练轮数
        lr: 学习率
        inputs: 生成的随机数据
        labels: 生成的one-hot标签
    '''
    # 一轮训练过程: 每个样本做一次前向传播和反向传播
    xpoints = np.zeros(epochs, dtype=np.int)
    ypoints = np.zeros(epochs)
    for e in range(epochs):
        avg_loss = 0.0
        for i in range(len(inputs)):
            forward_pack = mlp.forward(inputs[i])
            avg_loss += mlp.backward(inputs[i], forward_pack, labels[i], lr)
        avg_loss /= len(inputs)
        xpoints[e] = e
        ypoints[e] = avg_loss
        print('epoch %d: average loss: %f' %(e, avg_loss))
        
    # 绘制loss曲线    
    plt.plot(xpoints, ypoints)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
 

if __name__ == '__main__':
    # 设置随机种子,保证结果的可复现性
    np.random.seed(1)
    mlp = MLP()
    # 生成数据
    inputs = np.random.randn(100, 10)
    

    # 生成one-hot标签
    labels = np.eye(4)[np.random.randint(0, 4, size=(1, 100))].reshape(100, 4)

    # 训练
    epochs = 50
    lr = 0.1
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
        lr = float(sys.argv[2])
    train(mlp, epochs, lr, inputs, labels)
    
    