from utils import *
from DecisionTree import DecisionTree
from SVM import SupportVectorMachine


def test_decisiontree():
    train_features, train_labels, test_features, test_labels = load_decisiontree_dataset()   
    
    '''
    print(train_features)
    print(train_labels)
    model = DecisionTree()
    info_gains = model.cal_info_gain(train_features, train_labels)
    print(info_gains)
    attr = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    print(model.choose_attr(attr, info_gains))
    model.fit(train_features, train_labels)
    print('--调试中')
    return
    '''
    
    model = DecisionTree()
    model.fit(train_features, train_labels)
    results = model.predict(test_features)
    # results = np.random.randint(2, size=56)
    print('DecisionTree acc: {:.2f}%'.format(get_acc(results, test_labels) * 100))

def test_svm(C=1, kernel='Linear', epsilon = 1e-4):
    train_features, train_labels, test_features, test_labels = load_svm_dataset()
    
    '''
    model = SupportVectorMachine(C, kernel, epsilon)
    model.fit(train_features, train_labels)
    print('--debuging...')
    '''
    
    model = SupportVectorMachine(C, kernel, epsilon)
    model.fit(train_features, train_labels)
    pred = model.predict(test_features)
    print('SVM({} kernel) acc: {:.2f}%'.format(kernel, get_acc(pred, test_labels.reshape(-1,)) * 100))
    

if __name__=='__main__':
    test_decisiontree()
    test_svm(1, 'Linear', 1e-4)
    test_svm(1, 'Poly', 1e-4)
    test_svm(1, 'Gauss', 1e-4)
    
