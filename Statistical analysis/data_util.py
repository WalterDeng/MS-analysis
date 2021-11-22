import numpy as np
import csv
import scipy.io as io
import pandas as pd

# 加载数据
def load_data():
    with open(r'C:\Users\DELL\Documents\R\R-projects\areaData_del.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        target_names = ["ko", "wt"]
        X = np.array([row[1:] for row in reader], np.float32)
        y = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
        return X, y, target_names

# 写入文件
def write_list(fileName, landmarks):
    fp = open(fileName, 'w+')
    fp.write(
        "version: 1" + '\n'
                       "n_points: 68" + '\n'
                                        "{" + '\n'
    )
    for i in range(len(landmarks)):
        fp.write(str(landmarks[i][0]))
        fp.write(" ")
        fp.write(str(landmarks[i][1]) + '\n')

    fp.write("}")
    fp.close()
    return True

## 加载matlab处理后的conlon数据
def load_colon():
    # path = "D:\D1+D2 lookfor peaks\colon.mat"
    path = "D:\DATA\colon\colon_9.mat"
    colonMat = io.loadmat(path)
    print(colonMat.keys())
    AreaY = colonMat.get('AreaY')
    XPeak = colonMat.get('XPeak')
    YPeak = colonMat.get('YPeak')

    ## 去除冗余维度
    AreaY = np.squeeze(AreaY)
    for i in range(len(AreaY)):
        AreaY[i] = np.squeeze(AreaY[i])
    XPeak = np.squeeze(XPeak)
    for i in range(len(XPeak)):
        XPeak[i] = np.squeeze(XPeak[i])
    YPeak = np.squeeze(YPeak)
    for i in range(len(YPeak)):
        YPeak[i] = np.squeeze(YPeak[i])

    AreaY = AreaY.transpose()
    XPeak = XPeak.transpose()
    YPeak = YPeak.transpose()

    return AreaY, XPeak, YPeak

## 对数据进行分箱处理
def Bining2(AreaY, XPeak, YPeak, width1=1, width2=2):
    """
       动态规划思想处理并返回分箱好的数据.

       Parameters
       ----------
       AreaY : array_like
           样本的面积列表.
       XPeak : array_like
           样本检测峰的时间.
       YPeak : array_like
           样本检测峰值.
       padding : int
           样本峰数较少时填充值.
       width1 : int
           峰值检测的半宽度.
       width2 : int
           峰值检测的全宽度.

       Returns
       -------
       Time_list : ndarray. (sample_num, 2)
       Area_list : ndarray. (sample_num, features)

    """
    Time_list = list()
    Area_list = list()
    left = 0
    right = 0
    sample_num = XPeak.shape[0]
    i_to_j = np.zeros(sample_num, int)
    j_max = [XPeak[i].shape[0] - 1 for i in range(sample_num)]
    # print("j_max: ", j_max)
    while(True):
        # print("i_to_j: ", i_to_j)
        # print("left: ", left, "right: ", right)
        # 过滤没有数据的样本
        arg_index = np.squeeze(np.argwhere(i_to_j <= j_max))
        # print("arg_index: ", arg_index)
        arg_num = arg_index.size
        if arg_num == 1: arg_index = np.array([arg_index])
        # print("arg_num: ", arg_num)

        # 当所有样本都已遍历，跳出循环
        if arg_num == 0: break

        # 取出该列数据
        cur_time = np.zeros(sample_num)
        for i in arg_index:
            j = i_to_j[i]
            cur_time[i] = XPeak[i][j]

        # 获取该列参数
        mean = np.sum(cur_time) / arg_num
        min_val = np.min(cur_time[np.where(cur_time > 0)])
        max_val = np.max(cur_time)

        # 参数检验
        if min_val < right:
            print("error, min_val: ", min_val, "right: ", right, "arg_index: ", arg_index)
            break

        # 确定左右边界

        # 初步定义边界，左边界为当前列最小值， 后边界为下一列最小值（下一列最小值为当前列最大值）
        new_left = min_val
        new_right = min(new_left+width2, mean+width1)  # 取较小值，确保峰集中在固定范围内
        # 进一步确定边界
        left_num = 0
        is_over = False
        for i in range(sample_num):
            rt = cur_time[i]
            # 仅考虑在边界左边的样本
            if rt <= new_right:
                left_num += 1
                if i_to_j[i] + 1 <= j_max[i] and new_right > XPeak[i][i_to_j[i] + 1]:
                    new_right = XPeak[i][i_to_j[i] + 1]
                    is_over = True
        # # 如果没有越界且
        # if not is_over and left_num < arg_num / 2:

        Time_list.append((new_left, new_right))

        # 根据边界分组
        cur_A_list = list()
        for i in range(sample_num):

            rt = cur_time[i]
            # 仅处理数据在边界内的样本
            if new_left <= rt <= new_right:
                cur_A_list.append(AreaY[i][i_to_j[i]])

                # 处理i_to_j数组
                i_to_j[i] += 1
            # 其余补0
            else:
                cur_A_list.append(0)
        # 分组数据放入最终结果
        Area_list.append(cur_A_list)

        # 更新边界
        left = new_left
        right = new_right
    Area_list = np.transpose(np.array(Area_list))
    Time_list = np.transpose(np.array(Time_list))
    # print("Time_list: ", Time_list)
    # print("Area_list: ", Area_list)
    # print("Area_list.shape: ", Area_list.shape)

    return Time_list, Area_list

## 对数据进行分箱处理
def Bining3(AreaY, XPeak, YPeak, width1=1, width2=2, ratio_zero=0.5):
    """
       动态规划思想处理并返回分箱好的数据.

       Parameters
       ----------
       AreaY : array_like
           样本的面积列表.
       XPeak : array_like
           样本检测峰的时间.
       YPeak : array_like
           样本检测峰值.
       padding : int
           样本峰数较少时填充值.
       width1 : int
           峰值检测的半宽度.
       width2 : int
           峰值检测的全宽度.

       Returns
       -------
       Time_list : ndarray. (sample_num, 2)
       Area_list : ndarray. (sample_num, features)

    """
    print(XPeak.shape)
    Time_list = list()
    Area_list = list()
    left = 0
    right = 0
    sample_num = XPeak.shape[0]
    i_to_j = np.zeros(sample_num, int)
    j_max = [XPeak[i].shape[0] - 1 for i in range(sample_num)]
    # print("j_max: ", j_max)
    while(True):
        # print("i_to_j: ", i_to_j)
        # print("left: ", left, "right: ", right)
        # 过滤没有数据的样本
        arg_index = np.squeeze(np.argwhere(i_to_j <= j_max))
        # print("arg_index: ", arg_index)
        arg_num = arg_index.size
        if arg_num == 1: arg_index = np.array([arg_index])
        # print("arg_num: ", arg_num)

        # 当所有样本都已遍历，跳出循环
        if arg_num == 0: break

        # 取出该列数据
        cur_time = np.zeros(sample_num)
        for i in arg_index:
            j = i_to_j[i]
            cur_time[i] = XPeak[i][j]

        # 获取该列参数
        mean = np.sum(cur_time) / arg_num
        min_val = np.min(cur_time[arg_index])
        max_val = np.max(cur_time)

        # 参数检验
        if min_val < right:
            print("error, min_val: ", min_val, "right: ", right, "arg_index: ", arg_index)
            break

        # 确定左右边界

        # 初步定义边界，左边界为当前列最小值， 后边界为下一列最小值（下一列最小值为当前列最大值）
        new_left = min_val
        new_right = get_col_min(i_to_j+1, j_max, sample_num, XPeak, min(new_left+width2, mean+width1))
        # new_right = min(new_left+width2, mean+width1)  # 取较小值，确保峰集中在固定范围内
        # # 进一步确定边界
        # left_num = 0
        # left_max_rt = 0
        # for i in range(arg_index):
        #     rt = cur_time[i]
        #     # 仅考虑在边界左边的样本
        #     if rt <= new_right:
        #         left_num += 1
        #
        # if left_num > arg_num / 2:

        Time_list.append((new_left, new_right))

        # 根据边界分组
        cur_A_list = list()
        for i in range(sample_num):

            rt = cur_time[i]
            # 仅处理数据在边界内的样本
            if new_left <= rt <= new_right:
                cur_A_list.append(AreaY[i][i_to_j[i]])

                # 处理i_to_j数组
                i_to_j[i] += 1
            # 其余补0
            else:
                cur_A_list.append(0)
        # 分组数据放入最终结果
        Area_list.append(cur_A_list)

        # 更新边界
        left = new_left
        right = new_right
    Area_list = np.transpose(np.array(Area_list))
    # Time_list = np.transpose(np.array(Time_list))
    # print("Time_list: ", Time_list)
    # print("Area_list: ", Area_list)
    print("Area_list.shape: ", Area_list.shape)

    extra_index = []
    for i in range(Area_list.shape[1]):
        notZeroNum = np.size(np.where(Area_list[:, i] > 0))
        if notZeroNum > sample_num * ratio_zero:
            extra_index.append(i)
    Area_del_0 = Area_list[:, extra_index]
    print("Area_del_0.shape: ", Area_del_0.shape)

    return Time_list, Area_list, Area_del_0

def get_col_min(i_to_j, j_max, sample_num, XPeak, default):
    arg_index = np.squeeze(np.argwhere(i_to_j <= j_max))
    # print("arg_index: ", arg_index)
    arg_num = arg_index.size
    if arg_num == 1: arg_index = np.array([arg_index])
    # print("arg_num: ", arg_num)

    # 当前所有样本都已遍历
    if arg_num == 0: return default

    # 取出该列数据
    cur_time = np.zeros(sample_num)
    for i in arg_index:
        j = i_to_j[i]
        cur_time[i] = XPeak[i][j]

    # 获取该列参数
    min_val = np.min(cur_time[arg_index])
    return min_val

# load_colon()

def normalizeArea(X):
    A_sum = np.sum(X, axis=1)
    for i in range(X.shape[0]):
        X[i, :] = X[i, :] / A_sum[i]
    return X

## 对数据进行分箱处理
def Bining(AreaY, XPeak, YPeak, padding, width=0.5, ratio=0.2):
    """
       返回分箱好的数据.

       Parameters
       ----------
       AreaY : array_like
           样本的面积列表.
       XPeak : array_like
           样本检测峰的时间.
       YPeak : array_like
           样本检测峰值.
       padding : int
           样本峰数较少时填充值.
       width : int
           峰值检测的宽度.
       ratio : float
           比例.

       Returns
       -------
       Time_list : ndarray.
       Area_list : ndarray.

    """

    max_time = np.max(XPeak)
    sample_num = np.shape(XPeak)[0]

    cur_col = 1
    tolal_col = 1
    Time_list = list()
    Area_list = list()
    Area_index = 0
    Time_index = 0
    i_to_j = np.zeros(sample_num)
    while(True):
        # 取出该列数据
        cur_time = np.zeros(sample_num)
        for i in range(sample_num):
            cur_time[i] = (XPeak[i, i_to_j[i]])
        # 求该列均值 （有填充要处理）
        mean = np.mean(cur_time)

        min_val = mean - width
        min_args = np.squeeze(np.argwhere(cur_time < min_val))
        min_num = min_args.shape[0]

        max_val = mean + width
        max_args = np.squeeze(np.argwhere(cur_time > min_val))
        max_num = max_args.shape[0]

        # 先判断有无前峰
        if min_num / sample_num < ratio:
            # 少的单独成列
            for i in range(sample_num):
                if(i in min_args):
                    Area_list.append(AreaY[i, i_to_j[i]])  ### 有j超出上限的可能
                else:
                    Area_list.append(0)

            i_to_j[min_args] += 1

        elif min_num / sample_num < ratio:
            return
