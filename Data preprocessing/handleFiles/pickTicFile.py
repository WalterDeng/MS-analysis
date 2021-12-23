import sys
import os
import shutil


root = sys.argv[1]
targetDir = sys.argv[2]
print("root: " ,root)
print("targetDir: " ,targetDir)

targetFile = "tic_front.csv"

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + "\\" + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + "\\" + fname))

# 挑选出root下所有的targetFile，并统一复制到targetDir中
def pickTicFile(root, targetFile, targetDir):
    for filepath, dirnames, filenames in os.walk(root):
        if targetFile in filenames:
            newName = targetDir + "\\" + filepath.split('\\')[-2] + "_" + filepath.split('\\')[-1] + ".csv"
            oldName = filepath + "\\" + targetFile
            shutil.copy(oldName, newName)

pickTicFile(root, targetFile, targetDir)
