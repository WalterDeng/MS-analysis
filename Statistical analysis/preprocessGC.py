import numpy as np
from sklearn import preprocessing
from data_util import *
from myPreprocessing import *
from myEvaluation import *
from Peak_subset_selection import peak_subset_selection
from diomenReduction import diomensionReduction
from evaluationModel import BEST_FEATURES

# BEST_FEATURES_HEALTHY_LUNG = [0, 51, 84, 122, 129, 153, 206, 235, 244, 300]

# 肺癌病人时序变化
def lung_time_series():
    X, Y, target_names, Area, lung_series_dict = load_lung_series()
    test_evaluation(X, Y)
    #
    # X = normalizeArea(X)
    # X = X[:, BEST_FEATURES.get("lung_healthy")]
    #
    # Area = normalizeArea(Area)
    # Area = Area[:, BEST_FEATURES.get("lung_healthy")]
    #
    # for person_name, lung_indices in lung_series_dict.items():
    #     before_series = Area[lung_indices["before"], :]
    #     after_series = Area[lung_indices["after"], :]
    #     print("person_name: ",person_name, ", lung_indices: ", lung_indices)
    #     binaryTimeSeries(X, Y, before_series, after_series, target_names, title="Time Series Of "+person_name, isnumber=False, tag_false=True)
    # return

# 肺癌与对照
def healthy_and_lung():
    X, Y, target_names = load_healthy_lung()
    X = X[:, BEST_FEATURES.get("lung_healthy")]
    X = normalizeArea(X)
    # myPCA(X, Y, target_names)
    myLDAForTwo(X, Y, target_names, title="LDA oF healthy_and_lung", isnumber=False, tag_false=True)
    return

# 胸腺瘤与对照
def healthy_and_thoracic():
    X, Y, target_names = load_healthy_lung()
    # X = X[:, BEST_FEATURES_HEALTHY_LUNG]
    X = normalizeArea(X)
    myLDAForTwo(X, Y, target_names, title="LDA oF healthy_and_lung", isnumber=False, tag_false=True)
    return


# healthy_and_lung()

lung_time_series()



# AreaY, XPeak, YPeak = load_colon()

# Time_list, Area_list, Area_del_0 = Bining3(AreaY, XPeak, YPeak, width1=2, width2=4, ratio_zero=0.7)
# write_list("Time_list.txt", Time_list)
def test():

    Height, Area, Y, diseaseDict = load_samples()

    lung_before_indices = diseaseDict.get('lung_before_indices')
    Thoracic_before_indices = diseaseDict.get('Thoracic_before_indices')
    lung_after_indices = diseaseDict.get('lung_after_indices')
    Thoracic_after_indices = diseaseDict.get('Thoracic_after_indices')

    X = Area
    target_names = ['health', 'sick', 'operate']
    # 根据选出的三分类最优子集切片
    # best_features=[4, 36, 39, 79, 83, 91, 96, 99, 130, 146, 183, 187, 231, 243, 256, 264, 282, 292, 326]
    # X = X[:, best_features]

    # Y[Y >= 1] = 1
    # target_names = ['health', 'sick']

    # 清除0列
    # X_del_0 = del_0(X, 0.1)

    # 归一化
    X = normalizeArea(X)

    # LDA_evaluation(X, Y)
    # LDA_cross_evaluation(X, Y)

    # myPCA(X, Y, target_names, isnumber=False)
    # myLDA(X, Y, target_names, isnumber=True)
    # myLDAForMul(X, Y, target_names, isnumber=False, tag_false=True)

    # 迭代峰值子集
    # best_score, best_features = peak_subset_selection(X, Y, max_feature=10)

    # diomensionReduction(X, Y)

