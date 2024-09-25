import os.path

from PIL import Image
import torch
from torch.utils.data import Dataset
import csv


# 读取头部姿态文件
def readcsv():
    num = 6
    path = {}
    for i in range(num):
        # 添加不同人的头部姿态数据集
        pathwl = os.path.join("/successFUL/MobileVit_Data/SuccessData", str(i), "ll.csv")
        pathwr = os.path.join("/successFUL/MobileVit_Data/SuccessData", str(i), "rr.csv")

        pathgl = os.path.join("/successFUL/MobileVit_Data/SuccessData", str(i), "gl.csv")
        pathgr = os.path.join("/successFUL/MobileVit_Data/SuccessData", str(i), "gr.csv")

        pathdl = os.path.join("/successFUL/MobileVit_Data/SuccessData", str(i), "dl.csv")
        pathdr = os.path.join("/successFUL/MobileVit_Data/SuccessData", str(i), "dr.csv")

        pathxl = os.path.join("/successFUL/MobileVit_Data/SuccessData", str(i), "xl.csv")
        pathxr = os.path.join("/successFUL/MobileVit_Data/SuccessData", str(i), "xr.csv")

        # 每加载一个csv，都要处理一遍
        with open(pathwl, "r") as file:
            rows = file.readlines()
            for row in rows:
                if row[0] != "I":
                    a = row.split(",")
                    path[a[0]] = (a[1], a[2], a[3])
        with open(pathwr, "r") as file:
            rows = file.readlines()
            for row in rows:
                if row[0] != "I":
                    a = row.split(",")
                    path[a[0]] = (a[1], a[2], a[3])
        #
        with open(pathgl, "r") as file:
            rows = file.readlines()
            for row in rows:
                if row[0] != "I":
                    a = row.split(",")
                    path[a[0]] = (a[1], a[2], a[3])
        with open(pathgr, "r") as file:
            rows = file.readlines()
            for row in rows:
                if row[0] != "I":
                    a = row.split(",")
                    path[a[0]] = (a[1], a[2], a[3])

        with open(pathdl, "r") as file:
            rows = file.readlines()
            for row in rows:
                if row[0] != "I":
                    a = row.split(",")
                    path[a[0]] = (a[1], a[2], a[3])
        with open(pathdr, "r") as file:
            rows = file.readlines()
            for row in rows:
                if row[0] != "I":
                    a = row.split(",")
                    path[a[0]] = (a[1], a[2], a[3])

        with open(pathxl, "r") as file:
            rows = file.readlines()
            for row in rows:
                if row[0] != "I":
                    a = row.split(",")
                    path[a[0]] = (a[1], a[2], a[3])
        with open(pathxr, "r") as file:
            rows = file.readlines()
            for row in rows:
                if row[0] != "I":
                    a = row.split(",")
                    path[a[0]] = (a[1], a[2], a[3])

    return path


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.path = readcsv()

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        path = os.path.basename(self.images_path[item])
        pitch, roll, yaw = float(self.path[path][0]), float(self.path[path][1]), float(self.path[path][2])
        hplabel = torch.tensor([pitch, roll, yaw])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, hplabel

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, hplabels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        hplabels = torch.stack(hplabels, dim=0)
        return images, labels, hplabels
