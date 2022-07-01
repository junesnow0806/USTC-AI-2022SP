import numpy as np
import cvxpy as cvx
import copy

class SupportVectorMachine:
    def __init__(self, C=1, kernel='Linear', epsilon=1e-4):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel

        # Hint: 你可以在训练后保存这些参数用于预测
        # SV即Support Vector，表示支持向量，SV_alpha为优化问题解出的alpha值，
        # SV_label表示支持向量样本的标签。
        self.SV = []
        self.SV_alpha = []
        self.SV_label = []
        self.b = None

    def KERNEL(self, x1, x2, d=2, sigma=1):
        #d for Poly, sigma for Gauss
        if self.kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif self.kernel == 'Linear':
            K = np.dot(x1,x2)
        elif self.kernel == 'Poly':
            K = (np.dot(x1,x2) + 1) ** d
        else:
            raise NotImplementedError()
        return K
    
    def cal_yyT(self, y):
        '''
        输入一个列向量y, 计算y与yT(转置)的乘积, 返回一个yyT矩阵
        '''
        n = y.shape[0]
        yyT = np.empty([n, n])
        for i in range(n):
            for j in range(n):
                yyT[i][j] = y[i] * y[j]
        return yyT
    
    def cal_K(self, X):
        '''
        输入样本集X, X是一个二维向量, X[i]代表一个样本, 一个样本为一个向量
        计算它们的核函数矩阵K
        '''
        n = X.shape[0]
        K = np.empty([n, n])
        for i in range(n):
            for j in range(n):
                K[i][j] = self.KERNEL(X[i], X[j])
        return K
    
    def debur(self, alpha):
        '''
        将小于epsilon的值置成0
        '''
        for i in range(len(alpha)):
            if alpha[i] < self.epsilon:
                alpha[i] = 0.0
      
    def fit(self, train_data, train_label):
        '''
        TODO: 实现软间隔SVM训练算法
        train_data：训练数据，是(N, 7)的numpy二维数组，每一行为一个样本
        train_label：训练数据标签，是(N,)的numpy数组，和train_data按行对应
        '''
        n = train_data.shape[0]
        alpha = cvx.Variable(n, pos=True)
        yyT = self.cal_yyT(train_label)
        K = self.cal_K(train_data)
        G = yyT * K # 星乘, 矩阵内各位置的元素相乘
        obj = cvx.Maximize(cvx.sum(alpha) - (1/2)*cvx.quad_form(alpha, G))
        constraints = [alpha <= self.C,
                       cvx.sum(cvx.multiply(alpha, train_label)) == 0]
        prob = cvx.Problem(obj, constraints)
        prob.solve(adaptive_rho=False)
        self.debur(alpha.value)
        for i in range(len(alpha.value)):
            if alpha.value[i] > 0:
                self.SV_alpha.append(alpha.value[i])
                self.SV.append(train_data[i])
                self.SV_label.append(train_label[i])
        
    def predict_single(self, x):
        predict_y = 0.0
        for j in range(len(self.SV)):
            alpha_j = self.SV_alpha[j]
            yj = self.SV_label[j]
            xj = self.SV[j]
            predict_y += alpha_j * yj * self.KERNEL(xj, x)
        predict_y += self.b
        if predict_y >= 0:
            return 1
        else:
            return -1

    def predict(self, test_data):
        '''
        TODO: 实现软间隔SVM预测算法
        test_data：测试数据，是(M, 7)的numpy二维数组，每一行为一个样本
        必须返回一个(M,)的numpy数组，对应每个输入预测的标签，取值为1或-1表示正负例
        '''
        i = np.argmax(self.SV_alpha)
        xi = self.SV[i]
        yi = self.SV_label[i]
        b = 0.0
        assert len(self.SV) == len(self.SV_alpha)
        assert len(self.SV) == len(self.SV_label)
        for j in range(len(self.SV)):
            alpha_j = self.SV_alpha[j]
            yj = self.SV_label[j]
            xj = self.SV[j]
            b += alpha_j * yj * self.KERNEL(xi, xj)
        b = yi - b
        self.b = b
        n = len(test_data)
        predict_labels = np.zeros(n, dtype=int)
        for i in range(n):
            predict_labels[i] = self.predict_single(test_data[i])
        return predict_labels