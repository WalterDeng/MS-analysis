from sklearn import discriminant_analysis
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.utils import shuffle
import numpy as np
from sklearn import metrics
from plot_lda_qda import plot_lda_result, plot_time_series

# 评估LDA分数
def LDA_evaluation(X, y):
    # 随机打乱
    # X, y = shuffle(X, y, random_state=0)

    # 分割数据集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.4, random_state = 0)

    lda = discriminant_analysis.LinearDiscriminantAnalysis()

    lda.fit(X_train, y_train)

    # score函数获取各子集平均正确率
    scores = lda.score(X_test, y_test)

    print("LDA_evaluation scores: ", scores)

    return scores

# 交叉评估LDA分数
def LDA_cross_evaluation(X, y):
    # 随机打乱
    # X, y = shuffle(X, y, random_state=0)

    lda = discriminant_analysis.LinearDiscriminantAnalysis()

    # 评分默认采用各评估器自带score函数，也可用scoring指定
    scores = model_selection.cross_val_score(lda, X, y, cv=5)
    print("LDA_cross_evaluation scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # scores_f1 = model_selection.cross_val_score(lda, X, y, cv=10, scoring="f1_macro")
    # print("LDA_cross_evaluation f1_macro scores: ", scores_f1)
    # print("f1_macro Accuracy: %0.2f (+/- %0.2f)" % (scores_f1.mean(), scores_f1.std() * 2))

    return scores

def evaluation_func(X, Y):
    # 随机打乱
    X, y = shuffle(X, Y, random_state=0)

    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    score = metrics.recall_score(average='weighted')
    scores = model_selection.cross_validate(lda, X, y, cv=5, scoring=score)
    print(scores)
    # for score in scoring:
    #
    #     print("LDA_cross_evaluation scores: ", scores)
    #     print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def compute_score(X, y, scoreing='lda'):
    evaluator = discriminant_analysis.LinearDiscriminantAnalysis()
    if scoreing == 'qda':
        evaluator = discriminant_analysis.QuadraticDiscriminantAnalysis()

    # 评分默认采用各评估器自带score函数，也可用scoring指定
    scores = model_selection.cross_val_score(evaluator, X, y, cv=5)
    score = scores.mean()

    del evaluator
    return score

def test_evaluation(X, y):
    evaluator = discriminant_analysis.LinearDiscriminantAnalysis()
    evaluator.fit(X[1:10,:], y[1:10])
    print(evaluator.score(X, y))
    evaluator = discriminant_analysis.LinearDiscriminantAnalysis()
    evaluator.fit(X, y)
    print(evaluator.score(X, y))
    evaluator = discriminant_analysis.LinearDiscriminantAnalysis()
    evaluator.fit(X[1:10,:], y[1:10])
    print(evaluator.score(X, y))

# 交叉评估SVM分数
def SVM_cross_evaluation(X, y):
    # 随机打乱
    random_state = 0
    X, y = shuffle(X, y, random_state=random_state)

    svmm = svm.SVC(kernel="linear", probability=True, random_state=random_state)

    # 评分默认采用各评估器自带score函数，也可用scoring指定
    scores = model_selection.cross_val_score(svmm, X, y, cv=5)
    print("SVM_cross_evaluation scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # scores_f1 = model_selection.cross_val_score(lda, X, y, cv=10, scoring="f1_macro")
    # print("LDA_cross_evaluation f1_macro scores: ", scores_f1)
    # print("f1_macro Accuracy: %0.2f (+/- %0.2f)" % (scores_f1.mean(), scores_f1.std() * 2))

    return scores

# PCA及结果绘制
def myPCA(X, y, target_names, isnumber=False, n_components=2):
    pca = decomposition.PCA(n_components=n_components)
    X_r = pca.fit(X).transform(X)

    plt.figure()
    colors = ['navy', 'turquoise']
    # colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    if isnumber:
        for i in range(X_r.shape[0]):
            plt.text(X_r[i, 0], X_r[i, 1], str(i))

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of dataset')
    plt.show()

# PCA及结果绘制
def myPCAResult(X, y, n_components=2):
    pca = decomposition.PCA(n_components=n_components)
    X_r = pca.fit(X).transform(X)
    return X_r

# 多分类LDA及结果绘制
def myLDAForMul(X, y, target_names, isnumber=False, tag_false=False):
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    scores = lda.decision_function(X)
    predict_y = lda.predict(X)

    # plt.figure()
    # colors = ['navy', 'turquoise']
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    plt.figure()
    # 数字标记
    if isnumber:
        for i in range(X_r2.shape[0]):
            plt.text(X_r2[i, 0], X_r2[i, 1], str(i))
    # 错误标记
    if tag_false:
        for i in range(X_r2.shape[0]):
            if(predict_y[i] != y[i]):
                plt.text(X_r2[i, 0], X_r2[i, 1], str(i), color='red')

    # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    #     plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
    #                 label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Thoracic Dataset')
    # # 画一下决策边界
    # plot_decision_boundary(X_r2, y)
    plot_lda_result(X_r2, y, target_names)
    plt.show()

# 多分类LDA及结果绘制
def myLDAForTwo(X, y, target_names, title, isnumber=False, tag_false=False):
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    predict_y = lda.predict(X)

    plt.figure()
    # # 数字标记
    # if isnumber:
    #     for i in range(X_r2.shape[0]):
    #         plt.text(X_r2[i, 0], X_r2[i, 1], str(i))
    # # 错误标记
    # if tag_false:
    #     for i in range(X_r2.shape[0]):
    #         if(predict_y[i] != y[i]):
    #             plt.text(X_r2[i, 0], X_r2[i, 1], str(i), color='red')
    #
    # plt.title('LDA of Thoracic Dataset')
    # # 画一下决策边界
    # plot_decision_boundary(X_r2, y)
    plot_lda_result(X_r2[:-3, :], y[:-3], target_names, title)
    plt.show()

# 二分类时序图
def binaryTimeSeries(X, y, before_series, after_series, target_names, title, isnumber=False, tag_false=False):
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_r2 = lda.transform(X)
    before_series_r2 = lda.transform(before_series)
    after_series_r2 = lda.transform(after_series)

    plt.figure()
    plot_time_series(X_r2[:-3, :], y[:-3], before_series_r2, after_series_r2, target_names, title)
    plt.show()


# 二分类LDA及结果绘制
def myLDA(X, y, target_names, isnumber=False):
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
    X_r2 = lda.fit(X, y).transform(X)
    scores = lda.decision_function(X)
    predict_y = lda.predict(X)

    # plt.figure()
    # colors = ['navy', 'turquoise']
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    plt.figure()
    # 数字标记
    if isnumber:
        for i in range(X_r2.shape[0]):
            if(predict_y[i] == y[i]):
                plt.text(X_r2[i, 0], X_r2[i, 0], str(i))
            else:
                plt.text(X_r2[i, 0], X_r2[i, 0], str(i), color='red')

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 0], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Thoracic Dataset')
    # # 画一下决策边界
    # plot_decision_boundary(X, y)
    # plot_lda_result(X_r2, y, target_names)
    plt.show()

# 绘制决策边界(二分类可用）
def plot_decision_boundary(X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    # 用预测函数预测一下
    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)



def single_case_time_series(X, y, indices, target_names):
    caseX = X[indices, :]
    caseY = y[indices]

    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_r2 = lda.transform(X)
    caseX_r2 = lda.transform(caseX)
    scores = lda.decision_function(X)
    predict_y = lda.predict(X)

    # plt.figure()
    # colors = ['navy', 'turquoise']
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    plt.figure()
    # 数字标记

    for i in range(X_r2.shape[0]):
        if(predict_y[i] == y[i]):
            plt.text(X_r2[i, 0], X_r2[i, 1], str(i))
        else:
            plt.text(X_r2[i, 0], X_r2[i, 1], str(i), color='red')

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    # 画时间序列样本点
    plt.scatter(caseX_r2[:, 0], caseX_r2[:, 1], color='red', label='target')
    # 连虚线
    plt.plot(caseX_r2[:, 0], caseX_r2[:, 1], color='red', linestyle='--')


    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Thoracic Dataset')
    # # 画一下决策边界
    # plot_decision_boundary(X, y, lambda x: lda.predict(x))
    plt.show()
