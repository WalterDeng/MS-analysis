import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from scipy.interpolate import interp1d
# Import some data to play with

def compute_auc(X_train, X_test, y_train, y_test, features_l):
    ## 切片数据集特征
    X_train_s = np.array([X_train[:, i] for i in features_l]).T
    # print("X_train_s.shape: ", X_train_s.shape)
    X_test_s = np.array([X_test[:, i] for i in features_l]).T
    # print("X_test_s.shape: ", X_test_s.shape)

    ## 计算y_score
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_r2 = lda.fit(X_train_s, y_train).transform(X_test_s)
    y_score = lda.decision_function(X_test_s)
    # print("y_score: ", y_score.shape)

    ## 转换y_test格式
    # y_test = label_binarize(y_test, classes=[0, 1, 2])
    # n_classes = y_test.shape[1]

    ## auc
    # fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    # auc = metrics.auc(fpr, tpr)
    auc = metrics.roc_auc_score(y_test, y_score)
    # print("features_l：", str(features_l), "y_score: ", y_score)

    return auc

def get_roc():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target



    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Learn to predict each class against the other
    # classifier = OneVsRestClassifier(
    #     # svm.SVC(kernel="linear", probability=True, random_state=random_state)
    #     LinearDiscriminantAnalysis(n_components=2)
    # )
    #
    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X_train, y_train).transform(X_test)
    y_score = lda.decision_function(X_test)

    # for i in range(2, 4):
    #     X_train, X_test, y_train, y_test = np.delete(X_train, i, axis=1), X_test, y_train, y_test
    #     lda = LinearDiscriminantAnalysis(n_components=2)
    #     X_r2 = lda.fit(X_train, y_train).transform(X_test)
    #     y_score = lda.decision_function(X_test)

    # Binarize the output
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ######################### 特定类的 ROC 曲线图
    plt.figure()
    lw = 2
    plt.plot(
        fpr[2],
        tpr[2],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


######################################## 为多类问题绘制 ROC 曲线
def draw_roc_for_mul(fpr, tpr, roc_auc, n_classes, lw):
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp1d(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()