"""
====================================================================
Linear and Quadratic Discriminant Analysis with covariance ellipsoid
====================================================================

This example plots the covariance ellipsoids of each class and
decision boundary learned by LDA and QDA. The ellipsoids display
the double standard deviation for each class. With LDA, the
standard deviation is the same for all the classes, while each
class has its own standard deviation with QDA.

"""

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# #############################################################################
# Colormap
cmap = colors.LinearSegmentedColormap(
    "red_blue_classes",
    {
        "blue": [(0, 0.7, 0.7), (1, 1, 1)],
        "green": [(0, 0.7, 0.7), (1, 0.7, 0.7)],
        "red": [(0, 1, 1), (1, 0.7, 0.7)],
    },
)
plt.cm.register_cmap(cmap=cmap)


# #############################################################################
# Generate datasets
def dataset_fixed_cov():
    """Generate 2 Gaussians samples with the same covariance matrix"""
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0.0, -0.23], [0.83, 0.23]])
    X = np.r_[
        np.dot(np.random.randn(n, dim), C),
        np.dot(np.random.randn(n, dim), C) + np.array([1, 1]),
    ]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def dataset_cov():
    """Generate 2 Gaussians samples with different covariance matrices"""
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
    X = np.r_[
        np.dot(np.random.randn(n, dim), C),
        np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4]),
    ]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


# #############################################################################
# Plot functions
def plot_data(lda, X, y, y_pred, target_names, title, is_plot_false=True):

    plt.title(title)
    plt.ylabel("Data with\n fixed covariance")

    tp = y == y_pred  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    false_names = [x + "_false" for x in target_names]

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker=".", color="red", label=target_names[1])
    if is_plot_false:
        plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker="x", s=20, color="#990000", label=false_names[1])  # dark blue

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color="blue", label=target_names[0])
    if is_plot_false:
        plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker="x", s=20, color="#000099", label=false_names[0])  # dark red

    plt.legend(loc='best', shadow=False, scatterpoints=1)

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = 1-Z
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(
        xx, yy, Z, cmap="red_blue_classes", norm=colors.Normalize(0.0, 1.0), zorder=0
    )
    plt.contour(xx, yy, Z, [0.5], linewidths=2.0, colors="white")

    # means
    plt.plot(
        lda.means_[0][0],
        lda.means_[0][1],
        "*",
        color="yellow",
        markersize=15,
        markeredgecolor="grey",
    )
    plt.plot(
        lda.means_[1][0],
        lda.means_[1][1],
        "*",
        color="yellow",
        markersize=15,
        markeredgecolor="grey",
    )

    return

def plot_time_series_data(lda, X, y, y_pred, before_series, after_series, target_names, title):

    plt.title(title)
    plt.ylabel("Data with\n fixed covariance")

    tp = y == y_pred  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    time_series = np.concatenate((before_series, after_series), axis=0)
    plt.plot(time_series[:, 0], time_series[:, 1], linestyle="--", color="grey")

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker=".", color="red", label=target_names[1])
    plt.scatter(before_series[:, 0], before_series[:, 1], marker="*", color="red", label='before_operation')

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color="blue", label=target_names[0])
    plt.scatter(after_series[:, 0], after_series[:, 1], marker="*", color="blue", label='after_operation')



    for i in range(0, time_series.shape[0]):
        plt.text(time_series[i, 0], time_series[i, 1], s=str(i+1), color="black")

    plt.legend(loc='best', shadow=False, scatterpoints=1)

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = 1 - Z
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(
        xx, yy, Z, cmap="red_blue_classes", norm=colors.Normalize(0.0, 1.0), zorder=0
    )
    plt.contour(xx, yy, Z, [0.5], linewidths=2.0, colors="white")

    # means
    # plt.plot(
    #     lda.means_[0][0],
    #     lda.means_[0][1],
    #     "*",
    #     color="yellow",
    #     markersize=15,
    #     markeredgecolor="grey",
    # )
    # plt.plot(
    #     lda.means_[1][0],
    #     lda.means_[1][1],
    #     "*",
    #     color="yellow",
    #     markersize=15,
    #     markeredgecolor="grey",
    # )

    return

# Plot functions
def plot_data_for_mul(lda, X, y, y_pred, target_names, title):

    plt.title(title)
    plt.ylabel("Data with\n fixed covariance")

    u, counts = np.unique(y, return_counts=True)
    true_color = ["blue", 'red', 'green']
    false_colors = ["#990000", "#000099", "#009900"]
    tp = y == y_pred  # True Positive
    false_names = [x + "_false" for x in target_names]
    for i in u:
        tpi = tp[y == i]
        Xi = X[y == i]
        X0_tp, X0_fp = Xi[tpi], Xi[~tpi]

        plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color=true_color[i], label=target_names[i])
        plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker="x", s=20, color=false_colors[i], label=false_names[i])

        # means
        plt.plot(
            lda.means_[i][0],
            lda.means_[i][1],
            "*",
            color="yellow",
            markersize=15,
            markeredgecolor="grey",
        )
    plt.legend(loc='best', shadow=False, scatterpoints=1)

    # areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

    # Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z[:, i].reshape(xx.shape)
    cm_light = mpl.colors.ListedColormap(['#FFA0A0', '#A0A0FF', '#A0FFA0'])
    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = 1 - Z
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(
        xx, yy, Z, cmap="red_blue_classes", norm=colors.Normalize(0.0, 5.0), zorder=0
        # xx, yy, Z, cmap=cm_light, zorder=0
    )
    plt.contour(xx, yy, Z, colors=["white", "white", "white"], zorder=0)

    return


def plot_ellipse(mean, cov, color):

    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(
        mean,
        2 * v[0] ** 0.5,
        2 * v[1] ** 0.5,
        180 + angle,
        facecolor=color,
        edgecolor="black",
        linewidth=2,
    )
    axes = plt.axes()
    ell.set_clip_box(axes.bbox)
    ell.set_alpha(0.2)
    axes.add_artist(ell)
    # axes.set_xticks(())
    # axes.set_yticks(())


def plot_lda_cov(lda):
    plot_ellipse(lda.means_[0], lda.covariance_, "blue")
    plot_ellipse(lda.means_[1], lda.covariance_, "red")


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariance_[0], "blue")
    plot_ellipse(splot, qda.means_[1], qda.covariance_[1], "red")

def plot_lda_result(X, y, target_names, title):
    u = np.unique(y)
    if u.shape[0] > 2:
        plot_lda_result_for_mul(X, y, target_names, title)
    else:
        plot_lda_result_for_two(X, y, target_names, title)
    plt.show()

def plot_time_series(X, y, before_series, after_series, target_names, title):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    plot_time_series_data(lda, X, y, y_pred, before_series, after_series, target_names, title)

def plot_lda_result_for_two(X, y, target_names, title):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    plot_data(lda, X, y, y_pred, target_names, title)
    plot_lda_cov(lda)


def plot_lda_result_for_mul(X, y, target_names, title):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    plot_data_for_mul(lda, X, y, y_pred, target_names, title)
    # plot_lda_cov(lda)

# plt.figure(figsize=(10, 8), facecolor="white")
# plt.suptitle(
#     "Linear Discriminant Analysis vs Quadratic Discriminant Analysis",
#     y=0.98,
#     fontsize=15,
# )
# for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
#     # Linear Discriminant Analysis
#     lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
#     y_pred = lda.fit(X, y).predict(X)
#     splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
#     plot_lda_cov(lda, splot)
#     plt.axis("tight")
#
#     # Quadratic Discriminant Analysis
#     qda = QuadraticDiscriminantAnalysis(store_covariance=True)
#     y_pred = qda.fit(X, y).predict(X)
#     splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
#     plot_qda_cov(qda, splot)
#     plt.axis("tight")
# plt.tight_layout()
# plt.subplots_adjust(top=0.92)
# plt.show()
