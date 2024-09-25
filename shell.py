import random
import os
import glob


num = 6
listl = []
listr = []
for i in range(num):
    list_l = glob.glob(os.path.join("D:\AllDataSet\me", str(i), "l", "l", "*.jpg"))
    list_r = glob.glob(os.path.join("D:\AllDataSet\me", str(i), "r", "r", "*.jpg"))
    listl.append(list_l)
    listr.append(list_r)
    random.shuffle(list_l)
    random.shuffle(list_r)


