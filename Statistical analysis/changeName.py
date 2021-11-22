import os
import sys
import re
import chardet

def get_files(root):
    files = list()
    for filepath,dirnames,filenames in os.walk(root):
        for filename in filenames:
            files.append(os.path.join(filepath,filename))
    return files

def changeName(root, origWord, targetWord):

    files = get_files(root)
    cur_path = os.getcwd()
    fLog = open(cur_path + "\\changeLog.txt", 'w')
    for file in files:
        try:
            encode = 'utf-8'
            with open(file, 'rb') as f:
                text = f.read()
                dicta = chardet.detect(text)
                encode1 = dicta.get('encoding')
                if encode1: encode = encode1
            f = open(file, 'r', encoding=encode)
            alllines = f.readlines()
            f.close()
            f = open(file, 'w+', encoding=encode)
            for eachline in alllines:
                if re.search(origWord, eachline):
                    a = re.sub(origWord, targetWord, eachline)
                    fLog.write("file_name: " + file)
                    fLog.write("eachline: " + eachline)
                    f.writelines(a)
                else:
                    f.writelines(eachline)
            f.close()
        except:
            fLog.write("file_name: " + file + " read fail!!! \n")
            # fLog.write("exec: " + str(exec) + "\n")
    fLog.close()

changeName(r"D:\20211117-tube1-250sccm-4min.D", r"GCMS", r"DYX")
