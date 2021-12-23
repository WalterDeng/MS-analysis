import numpy as np
import csv
import scipy.io as sio
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
    # fp.write(
    #     "version: 1" + '\n'
    #                    "n_points: 68" + '\n'
    #                                     "{" + '\n'
    # )
    for i in range(len(landmarks)):
        fp.write(str(landmarks[i][0]))
        fp.write(" ")
        fp.write(str(landmarks[i][1]) + '\n')

    # fp.write("}")
    fp.close()
    return True

## 加载matlab处理后的conlon数据
def load_colon():
    # path = "D:\D1+D2 lookfor peaks\colon.mat"
    path = "D:\DATA\colon\colon_9.mat"
    colonMat = sio.loadmat(path)
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

## 加载matlab处理后的conlon数据
def load_colon2():
    # path = "D:\D1+D2 lookfor peaks\colon.mat"
    Hpath = "D:\chromx\Height.mat"
    HDict = sio.loadmat(Hpath)
    Ypath = "D:\chromx\Tag.mat"
    YDict = sio.loadmat(Ypath)
    Apath = "D:\chromx\Area.mat"
    ADict = sio.loadmat(Apath)
    Height = HDict.get('Hr')
    Area = ADict.get('Ar')
    Y = np.squeeze(YDict.get('ActTag'))
    return Height, Area, Y
# load_colon2()

def load_samples():
    # path = "D:\D1+D2 lookfor peaks\colon.mat"
    Hpath = "D:\chromx\Height.mat"
    HDict = sio.loadmat(Hpath)
    Ypath = "D:\chromx\Tag.mat"
    YDict = sio.loadmat(Ypath)
    Apath = "D:\chromx\Area.mat"
    ADict = sio.loadmat(Apath)

    Height = HDict.get('Hr')
    Area = ADict.get('Ar')
    Y = np.squeeze(YDict.get('ActTag'))

    fileList = pd.read_excel(io="D:\chromx\\fileList.xlsx", header=None).T
    diseaseTag = pd.read_excel(io="D:\DATA\data_analysis\diseaseTag.xlsx", sheet_name=None)
    lung = diseaseTag.get("lung")
    Thoracic = diseaseTag.get("thoracic")
    lung_before = lung["before"]
    lung_after = lung["after"]
    Thoracic_before = Thoracic["before"]
    Thoracic_after = Thoracic["after"]
    lung_before_indices = []
    Thoracic_before_indices = []
    lung_after_indices = []
    Thoracic_after_indices = []
    i = 0
    for f in fileList[0]:
        if True in lung_before.str.contains(f, na=False).array:
            lung_before_indices.append(i)
        elif True in lung_after.str.contains(f, na=False).array:
            lung_after_indices.append(i)
        elif True in Thoracic_before.str.contains(f, na=False).array:
            Thoracic_before_indices.append(i)
        elif True in Thoracic_after.str.contains(f, na=False).array:
            Thoracic_after_indices.append(i)
        i = i + 1
    diseaseDict = {'lung_before_indices':lung_before_indices, 'thoracic_before_indices':Thoracic_before_indices,
                   'lung_after_indices':lung_after_indices, 'thoracic_after_indices':Thoracic_after_indices}

    lung_series = pd.read_excel(io="D:\DATA\data_analysis\\lung_series.xlsx", sheet_name=None)
    lung_series_dict = dict()
    for name, lung in lung_series.items():

        before_indices = []
        for f in lung["before"][lung["before"].notna()]:
            f_index = fileList[0][(fileList[0].str.contains(f, na=False))].index.tolist()
            if len(f_index)==0:
                print(f + " is empty")
            else:
                before_indices.append(f_index[0])

        after_indices = []
        for f in lung["after"][lung["after"].notna()]:
            f_index = fileList[0][(fileList[0].str.contains(f, na=False))].index.tolist()
            if len(f_index)==0:
                print(f + " is empty")
            else:
                after_indices.append(f_index[0])
        indices_dic = {"before":before_indices, "after":after_indices}
        lung_series_dict[name] = indices_dic

    return Height, Area, Y, diseaseDict, lung_series_dict

def load_healthy_lung_thoracic():
    Height, Area, Y, diseaseDict, _ = load_samples()
    lung_before_indices = diseaseDict.get('lung_before_indices')
    Thoracic_before_indices = diseaseDict.get('Thoracic_before_indices')
    lung_after_indices = diseaseDict.get('lung_after_indices')
    Thoracic_after_indices = diseaseDict.get('Thoracic_after_indices')

    X = Area

    healthy = X[Y == 0, :]
    lung = X[lung_before_indices, :]
    Thoracic = X[Thoracic_before_indices, :]
    target_names = ['healthy', 'lung', 'Thoracic']

    n_healthy = 20
    n_lung_before = 20
    n_Thoracic_before = 10

    train_X = np.concatenate((healthy[0:n_healthy, :], lung[0:n_lung_before, :], Thoracic[0:n_Thoracic_before, :]),
                             axis=0)
    test_X = np.concatenate((healthy[n_healthy:, :], lung[n_lung_before:, :], Thoracic[n_Thoracic_before:, :]), axis=0)

    train_Y = np.zeros((n_healthy + n_lung_before + n_Thoracic_before, 1), dtype='int')
    train_Y[n_healthy:n_healthy + n_lung_before, :] += 1
    train_Y[n_healthy + n_lung_before:, :] += 2
    train_Y = np.squeeze(train_Y)

    left_healthy = healthy.shape[0] - n_healthy
    left_lung_before = len(lung_before_indices) - n_lung_before
    left_Thoracic_before = len(Thoracic_before_indices) - n_Thoracic_before

    test_Y = np.zeros((left_healthy + left_lung_before + left_Thoracic_before, 1), dtype='int')
    test_Y[left_healthy:left_healthy + left_lung_before, :] += 1
    test_Y[left_healthy + left_lung_before:, :] += 2
    test_Y = np.squeeze(test_Y)

    return train_X, train_Y, test_X, test_Y, target_names

def load_healthy_lung():
    Height, Area, Y, diseaseDict, lung_series_dict = load_samples()
    lung_before_indices = diseaseDict.get('lung_before_indices')
    lung_after_indices = diseaseDict.get('lung_after_indices')[0:3]

    X = Area

    healthy = X[Y == 0, :]
    lung = X[lung_before_indices, :]
    lung_after = X[lung_after_indices, :]
    X = np.concatenate((healthy, lung, lung_after), axis=0)

    target_names = ['healthy', 'lung']

    n_healthy = healthy.shape[0]
    n_lung_before = len(lung_before_indices)
    n_lung_after = len(lung_after_indices)

    Y = get_Y({0:n_healthy, 1:n_lung_before, 2:n_lung_after})

    return X, Y, target_names

def load_healthy_thoracic():
    Height, Area, Y, diseaseDict, lung_series_dict = load_samples()
    thoracic_before_indices = diseaseDict.get('thoracic_before_indices')
    thoracic_after_indices = diseaseDict.get('thoracic_after_indices')[0:3]

    X = Area

    healthy = X[Y == 0, :]
    thoracic = X[thoracic_before_indices, :]
    thoracic_after = X[thoracic_after_indices, :]
    X = np.concatenate((healthy, thoracic), axis=0)

    target_names = ['healthy', 'thoracic']

    n_healthy = healthy.shape[0]
    n_thoracic_before = len(thoracic_before_indices)
    n_thoracic_after = len(thoracic_after_indices)

    Y = get_Y({0:n_healthy, 1:n_thoracic_before})

    return X, Y, target_names

def load_lung_series():
    Height, Area, Y, diseaseDict, lung_series_dict = load_samples()
    lung_before_indices = diseaseDict.get('lung_before_indices')
    lung_after_indices = diseaseDict.get('lung_after_indices')[0:3]

    X = Area

    healthy = X[Y == 0, :]
    lung = X[lung_before_indices, :]
    lung_after = X[lung_after_indices, :]
    X = np.concatenate((healthy, lung, lung_after), axis=0)

    target_names = ['healthy', 'lung']

    n_healthy = healthy.shape[0]
    n_lung_before = len(lung_before_indices)
    n_lung_after = len(lung_after_indices)

    Y = get_Y({0: n_healthy, 1: n_lung_before, 2: n_lung_after})
    # Y = np.zeros((n_healthy + n_lung_before + n_lung_after, 1), dtype='int')
    # Y[n_healthy:, :] += 1
    # Y[n_healthy + n_lung_before:, :] += 1
    # Y = np.squeeze(Y)

    return X, Y, target_names, Area, lung_series_dict

def get_Y(dict):
    Y = np.zeros(0, dtype=int)
    for y1, num in dict.items():
        Y1 = np.ones(num, dtype=int) * y1
        Y = np.concatenate((Y, Y1))
    return Y

def load_lung_thoracic():
    Height, Area, Y, diseaseDict, lung_series_dict = load_samples()
    lung_before_indices = diseaseDict.get('lung_before_indices')
    lung_after_indices = diseaseDict.get('lung_after_indices')[0:3]

    X = Area

    healthy = X[Y == 0, :]
    lung = X[lung_before_indices, :]
    lung_after = X[lung_after_indices, :]
    X = np.concatenate((healthy, lung, lung_after), axis=0)

    target_names = ['healthy', 'lung']

    n_healthy = healthy.shape[0]
    n_lung_before = len(lung_before_indices)
    n_lung_after = len(lung_after_indices)

    Y = np.zeros((n_healthy + n_lung_before + n_lung_after, 1), dtype='int')
    Y[n_healthy:, :] += 1
    Y[n_healthy + n_lung_before:, :] += 1
    Y = np.squeeze(Y)

    return X, Y, target_names

def print_sample_num(Y):
    uniques, counts = np.unique(Y, return_counts=True)
    for i in range(uniques.shape[0]):
        y = uniques[i]
        count = counts[i]
        print("sample ", y, " count ", count)

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

    Area_del_0 = del_0(Area_list, ratio_zero)

    return Time_list, Area_list, Area_del_0

# 删除data中0占比未达到p的特征列
def del_0(data, p):
    sample_num = data.shape[0]
    feature_num = data.shape[1]

    extra_index = []
    for i in range(feature_num):
        notZeroNum = np.size(np.where(data[:, i] > 0))
        if notZeroNum > sample_num * p:
            extra_index.append(i)
    Area_del_0 = data[:, extra_index]
    print("Area_del_0.shape: ", Area_del_0.shape)
    return Area_del_0

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
