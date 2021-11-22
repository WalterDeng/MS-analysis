import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import learning_curve
from preprocessGC import *

X = np.log2(X)
y = y
target_names = target_names
## ------------------------------------------- PCA/LDA --------------------------------------------------------
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)



# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y==i, 0], X_r[y==i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')


# lda = LinearDiscriminantAnalysis(n_components=2)
# X_r2 = lda.fit(X, y).transform(X)
# plt.figure()
# for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#     plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('LDA of IRIS dataset')

plt.show()

## ------------------------------------------------------------ SVM ------------------------------------------
# pca = PCA(n_components=6)
# X = pca.fit(X).transform(X)
# 拟合模型，并且为了展示作用，并不进行标准化
# clf = svm.SVC(kernel='linear')
# clf.fit(X, y)
#
# plt.scatter(X[:, 0], X[:, 1], c=y, s=225, cmap=plt.cm.Paired)

# 绘制decision function的结果
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# # 创造网格来评估模型
# xx = np.linspace(xlim[0], xlim[1], 225)
# yy = np.linspace(ylim[0], ylim[1], 225)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)
# 绘制决策边界和边际
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# 绘制支持向量
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
# plt.show()