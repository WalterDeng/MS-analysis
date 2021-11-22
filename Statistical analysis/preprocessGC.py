from sklearn import preprocessing
from data_util import *
from testPCAandLDA import colonPCA, colonLDA
from Peak_subset_selection import peak_subset_selection
from diomenReduction import diomensionReduction

AreaY, XPeak, YPeak = load_colon()

Time_list, Area_list, Area_del_0 = Bining3(AreaY, XPeak, YPeak, width1=2, width2=4, ratio_zero=0.7)
# write_list("Time_list.txt", Time_list)

# 归一化
Area_list = normalizeArea(Area_list)
Area_del_0 = normalizeArea(Area_del_0)
#
y = np.array([0,0,0,0,1,1,1,1,1])
#
target_names = ['colon', 'health']

colonPCA(Area_del_0, y, target_names)
colonLDA(Area_del_0, y, target_names)
# best_score, best_features = peak_subset_selection(Area_del_0, y, max_feature=10)

diomensionReduction(Area_del_0, y)
#
# X, y, target_names = load_data()
#
# ## scale标准化
# X_scaled = preprocessing.scale(X)
# X_mean = X_scaled.mean(axis=0)
# X_std = X_scaled.std(axis=0)
# print(X_std, X_mean)
#
# ## StandardScaler生成可对测试数据同样放缩的放缩器
# scaler = preprocessing.StandardScaler().fit(X)
# # print("scaler.mean_:", scaler.mean_)
# # print("scaler.scale_:", scaler.scale_)
# X_scaled2 = scaler.transform(X)
# # print("X_scaled:", X_scaled)
#
# ## 映射到高斯分布
# pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
# X_lognormal = pt.fit_transform(X)
# # print("X_lognormal:", X_lognormal)
#
# ## 归一化
# X_normalized = preprocessing.normalize(X_scaled, norm='l2')
# # print("X_normalized:", X_normalized)

