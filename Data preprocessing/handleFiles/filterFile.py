"""
python filterFile.py "D:\DATA\data_analysis\allTic"
"""



import sys
import os



root = sys.argv[1]
print("root: " ,root)

def filterFile(root):
    for filepath, dirnames, filenames in os.walk(root):
        for fileName in filenames:
            if "-M-" in fileName or "-Z-" in fileName or "-TJ-" in fileName:
                continue;
            else:
                oldName = filepath + "\\" + fileName
                os.remove(oldName)

filterFile(root)