import numpy as np
import math

from sympy import cartes
import copy

class treenode:
    def __init__(self, attr=None, category=None):
        self.attr = attr         
        self.category = category # 类别: 约定叶节点类别为0/1, 非叶节点为2
        self.children = {}       # 属性值作为字典索引, 一个孩子节点作为字典值
   
    def set_attr(self, attr):
        self.attr = attr
   
    def set_category(self, category):
        self.category = category
        
    def get_attr(self):
        return self.attr
        
    def get_category(self):
        return self.category

    def append_child(self, val, child):
        self.children[val] = child
        
    def get_child(self, val):
        return self.children[val]
        
    def get_branches(self):
        return self.children.keys()

class DecisionTree:
    def __init__(self):
        self.root = treenode()
    
    def cal_entropy(self, p):
        if p == 1.0 or p == 0.0:
            return 0.0
        return -(p * math.log2(p) + (1-p) * math.log2(1-p))
    
    def cal_info_gain(self, train_features, train_labels):
        '''
        计算每个特征的信息增益
        返回一个列表info_gains
        info_gains[i]表示第i个特征的信息增益, i = 0, 1, ..., 8
        '''
        # 统计每个特征每种取值的出现频率
        # 每个特征有一个字典, 键为该特征的取值, 值为该特征取值出现的次数
        feature_num = train_features.shape[1]
        sample_num = train_features.shape[0]
        frequency_set = []
        # freqecy_set[i]记录第i个特征的相关信息
        for i in range(feature_num):
            frequency_set.append({})
        for r in range(sample_num):
            for i in range(feature_num):
                feature_value = train_features[r][i]
                if feature_value not in frequency_set[i].keys():
                    frequency_set[i][feature_value] = {}
                    frequency_set[i][feature_value]['p'] = 0
                    frequency_set[i][feature_value]['n'] = 0
                if train_labels[r] == 1:
                    frequency_set[i][feature_value]['p'] += 1
                elif train_labels[r] == 0:
                    frequency_set[i][feature_value]['n'] += 1
            
        # 计算训练集label的信息熵
        pos = 0
        neg = 0
        for label in train_labels:
            if label == 1:
                pos += 1
            elif label == 0:
                neg += 1
        entropy = self.cal_entropy(pos / (pos+neg))
        
        # 计算每个特征的信息增益
        info_gains = []
        for i in range(feature_num):
            remainder = 0.0
            for key in frequency_set[i].keys():
                fvalue_pos = frequency_set[i][key]['p'] # 该特征取值下的正样例数
                fvalue_neg = frequency_set[i][key]['n']
                fvalue_all = fvalue_pos + fvalue_neg
                remainder += self.cal_entropy(fvalue_pos / fvalue_all) * (fvalue_all / sample_num)
            info_gains.append(entropy - remainder)
        return info_gains
    
    def choose_attr(self, attrs, info_gains):
        '''
        属性集: A = {attrs剩余属性编号, info_gains}
        从属性集中挑选信息增益最大的属性, 返回该属性编号
        '''
        max_info_gain = 0.0
        max_ig_attr = -1
        for i in attrs:
            if i < len(info_gains) and info_gains[i] > max_info_gain:
                max_info_gain = info_gains[i]
                max_ig_attr = i
        return max_ig_attr
    
    def is_all_one(self, samples, train_labels):
        for i in samples:
            if train_labels[i] != 1:
                return False
        return True
    
    def is_all_zero(self, samples, train_labels):
        for i in samples:
            if train_labels[i] != 0:
                return False
        return True
    
    def get_category(self, samples, train_labels):
        '''
        判断样本集samples是否都属于同一类别
        '''
        if self.is_all_one(samples, train_labels):
            return 1
        elif self.is_all_zero(samples, train_labels):
            return 0
        else:
            return 2
    
    def same_on_attrs(self, samples, train_features, attrs):
        '''
        样本集: samples
        属性集: attrs
        判断样本集是否在属性集上的取值都一样
        '''
        features_samples = []
        for sample in samples:
            tmp = []
            for attr in attrs:
                tmp.append(train_features[sample][attr])
            features_samples.append(tmp)
        if len(features_samples) > 1:
            for i in range(1, len(features_samples)):
                if features_samples[i] != features_samples[0]:
                    return False
        return True
    
    def get_most_category(self, samples, train_labels):
        one_count = 0
        zero_count = 0
        for i in samples:
            if train_labels[i] == 1:
                one_count += 1
            elif train_labels[i] == 0:
                zero_count += 1
        if one_count > zero_count:
            return 1
        else:
            return 0
        
    def get_values(self, attr, samples, train_features):
        '''
        samples: 样本编号的集合
        返回样本集samples在属性attr上的取值范围
        '''
        values = set()
        for sample in samples:
            values.add(train_features[sample][attr])
        return values
     
    def get_subsamples(self, attr, value, samples, train_features):
        '''
        返回样本集samples在属性attr上取值为value的子集
        '''
        subsamples = []
        for sample in samples:
            if train_features[sample][attr] == value:
                subsamples.append(sample)
        return subsamples
     
    def values_on_attrs(self, train_features):
        values = {}
        for i in range(train_features.shape[1]):
            values[i] = set()
        for i in range(train_features.shape[0]):
             for j in range(train_features.shape[1]):
                 values[j].add(train_features[i][j])
        return values
     
    def generate_tree(self, root, samples, attrs, values_attrs, train_features, train_labels, info_gains):
        '''
        样本集: D = {samples剩余样本编号, train_features, train_labels}
        属性集: A = {attrs剩余属性编号, info_gains}
        values_atrrs: 各属性的全局取值范围
        生成一个树节点
        '''
        category = self.get_category(samples, train_labels)
        if category == 1 or category == 0:
            # 如果samples中所有样本全属于同一类别C, 将节点标记为C类节点
            root.category = category
            return
        elif len(attrs) == 0 or self.same_on_attrs(samples, train_features, attrs):
            # 如果剩余属性集attrs为空
            # 或者samples中样本在attrs上取值相同(意味着属性集取值相同却存在不同类别)
            # 将节点类别标记为samples中样本数最多的类别
            root.category = self.get_most_category(samples, train_labels)
            return
        
        # 选择最优划分属性a*, 为该属性的每个值创建一个分支
        best_attr = self.choose_attr(attrs, info_gains)
        root.set_attr(best_attr)
        root.set_category(2) # 设置为非叶节点
        # values = self.get_values(best_attr, samples, train_features)
        values = values_attrs[best_attr]
        for value in values:
            subsamples = self.get_subsamples(best_attr, value, samples, train_features)
            if len(subsamples) == 0:
                # 没有在best_attr上该取值的样本
                # 将分支节点标记为叶节点, 其类别标记为samples中样本最多的类
                child = treenode()
                child.set_category(self.get_most_category(samples, train_labels))
                root.append_child(value, child)
            else:
                # 以generate_tree(Dv, A\{a*})为分支节点
                subattrs = copy.deepcopy(attrs)
                subattrs.remove(best_attr)
                child = treenode()
                self.generate_tree(child, subsamples, subattrs, values_attrs, train_features, train_labels, info_gains)
                root.append_child(value, child)

    def predict_single(self, test_feature):
        '''
        输入一个样本的feature向量, 输出对于该样本的预测结果label
        '''
        now = self.root
        while now.get_category() != 0 and now.get_category() != 1:
            attr = now.get_attr()
            val = test_feature[attr]
            # !!如果出现了训练过程中没有出现过的属性取值
            values = now.get_branches()
            assert val in values
            now = now.get_child(val)
        return now.get_category()

    def fit(self, train_features, train_labels):
        '''
        TODO: 实现决策树学习算法.
        train_features是维度为(训练样本数,属性数)的numpy数组
        train_labels是维度为(训练样本数, )的numpy数组
        '''
        info_gains = self.cal_info_gain(train_features, train_labels)
        samples = [i for i in range(0, train_features.shape[0])]
        attrs = [i for i in range(0, train_features.shape[1])]
        values_attrs = self.values_on_attrs(train_features)
        self.generate_tree(self.root, samples, attrs, values_attrs, train_features, train_labels, info_gains)

    def predict(self, test_features):
        '''
        TODO: 实现决策树预测.
        test_features是维度为(测试样本数,属性数)的numpy数组
        该函数需要返回预测标签，返回一个维度为(测试样本数, )的numpy数组
        '''
        predict_labels = []
        for i in range(test_features.shape[0]):
            predict_labels.append(self.predict_single(test_features[i]))
        return np.array(predict_labels)
    
# treenode: [attr, feat[attr] == 1, feat[attr] == 2, feat[attr] == 3]
