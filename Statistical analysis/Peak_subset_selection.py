import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from data_util import *
from ROCandAUC import compute_auc

# 导入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

def peak_subset_selection(X, y, max_feature=6, test_size=0.5, random_state=0):
    n_samples, n_features = X.shape
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    ## 初始化一维特征数据
    cur_aucs = dict()  # {features1 : auc1, features2 : auc2, ...}, features: (1, 2, ...)
    for num_f in range(0, n_features):
        temp = list()
        temp.append(num_f)
        cur_aucs[tuple(temp)] = 0

    best_features = ()
    best_score = 0

    ## 外层for循环，最多考虑几个特征
    for i in range(2, max_feature):
        new_aucs = dict()
        ## 遍历当前特征列表
        for features in cur_aucs.keys():
            ## 内层for循环，加入 n_features 个新的特征并计算auc
            for num_f in range(0, n_features):
                if num_f in features:
                    continue
                features_l = list(features)
                features_l.append(num_f)
                features_l.sort()
                new_features = tuple(features_l)
                ## 先查重，再计算，减少计算量
                if new_aucs.get(new_features, -1) != -1:
                    continue
                ## 使用添加后的features计算auc
                try:
                    cur_auc = compute_auc(X_train, X_test, y_train, y_test, features_l)
                    new_aucs[new_features] = cur_auc
                    ## 记录最佳组合
                    if cur_auc > best_score:
                        best_features = new_features
                        best_score = cur_auc
                except:
                    print("************ new_features: ", new_features)
                    X_train_s = np.array([X_train[:, i] for i in features_l]).T
                    print("X_train_s: ", X_train_s)
                    X_test_s = np.array([X_test[:, i] for i in features_l]).T
                    print("X_test_s: ", X_test_s)

        ## 排序，输出前n_features项
        sorted_list = list(new_aucs.items())
        sorted_list.sort(key=lambda a : a[1], reverse=True)
        sorted_list = sorted_list[0:n_features]

        ## 写入文件
        write_list("features"+str(i)+".txt", sorted_list)

        ## 更新数据
        cur_aucs.clear()
        cur_aucs = dict(sorted_list)

        print("already completed: ", i)

    return best_score, best_features

# peak_subset_selection(X, y, max_feature=6)