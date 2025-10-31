import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from PIL import Image


class IQA_dataloader(Dataset):
    def __init__(self, data_dir, csv_path, transform=None, database=None):
        self.database = database
        self.transform = transform
        if self.database == 'Koniq10k':
            column_names = ['image_name','c1','c2','c3','c4','c5','c_total','MOS','SD','MOS_zscore']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['image_name']]
            self.Y_train = tmp_df['MOS_zscore']

        elif self.database == 'FLIVE' or  self.database == 'FLIVE_patch':
            column_names = ['name','mos']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['name']]
            self.Y_train = tmp_df['mos']

        elif self.database == 'LIVE_challenge':
            column_names = ['image','mos','std']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['image']]
            self.Y_train = tmp_df['mos']

        elif self.database == 'SPAQ':
            column_names = ['name','mos','brightness','colorfulness','contrast','noisiness','sharpness']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['name']]
            self.Y_train = tmp_df['mos']
            
        elif self.database == 'BID':
            column_names = ['name','mos']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['name']]
            self.Y_train = tmp_df['mos']

        self.data_dir = data_dir
        self.transform = transform
        self.length = len(self.X_train)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.X_train.iloc[index, 0])

        img = Image.open(path)
        img = img.convert('RGB')
        # 根据 database 的值设置 cropped_image_path
        if self.database == 'Koniq10k':
            cropped_image_path = "E:/shujuji/koniq10k_1024x768/Cropped_Images/"
        elif self.database == 'LIVE_challenge':
            #cropped_image_path = 'E:/shujuji/ChallengeDB_release/ChallengeDB_release/Cropped_Images2/'

            ###0606试验尝试运用U2net裁剪的图片
            cropped_image_path = 'E:/shujuji/ChallengeDB_release/ChallengeDB_release/Cropped_Images_U2Net/'
        else:
            # 可以根据需要添加其他数据库的路径，或者设置默认路径
            cropped_image_path = ''

        # cropped_image_path = 'E:/shujuji/ChallengeDB_release/ChallengeDB_release/Cropped_Images2/'
        file_name = os.path.basename(path)
        cropped_image_path += file_name
        cropped_img = Image.open(cropped_image_path)
        cropped_img = cropped_img.convert('RGB')

        # 应用转换函数
        if self.transform is not None:
            # 处理元组情况
            if isinstance(self.transform, tuple) and len(self.transform) == 2:
                img = self.transform[0](img)  # 对原始图像应用第一个转换函数
                cropped_img = self.transform[1](cropped_img)  # 对裁剪图像应用第二个转换函数
            else:
                # 处理单个转换函数的情况
                img = self.transform(img)
                cropped_img = self.transform(cropped_img)

        y_mos = self.Y_train.iloc[index]
        if self.database == 'BID':
            y_label = torch.FloatTensor(np.array(float(y_mos * 20)))
        elif self.database == 'FLIVE' or self.database == 'FLIVE_patch':
            y_label = torch.FloatTensor(np.array(float(y_mos - 50) * 2))
        else:
            y_label = torch.FloatTensor(np.array(float(y_mos)))

        return img, cropped_img, y_label ##使用显著性检测
        #return img, y_label  ##不使用显著性检测

    # def __getitem__(self, index):
    #     path = os.path.join(self.data_dir,self.X_train.iloc[index,0])
    #
    #     img = Image.open(path)
    #     img = img.convert('RGB')
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     y_mos = self.Y_train.iloc[index]
    #     if self.database == 'BID':
    #         y_label = torch.FloatTensor(np.array(float(y_mos*20)))
    #     elif self.database == 'FLIVE' or self.database == 'FLIVE_patch':
    #         y_label = torch.FloatTensor(np.array(float(y_mos-50)*2))
    #     else:
    #         y_label = torch.FloatTensor(np.array(float(y_mos)))
    #
    #
    #     #return img, y_label
    #     # 假设 cropped_images 可以从某个路径加载
    #     cropped_image_path = 'E:/shujuji/ChallengeDB_release/ChallengeDB_release/Cropped_Images/'
    #
    #     file_name = os.path.basename(path)
    #     # print(file_name)
    #     cropped_image_path += file_name
    #     cropped_img = Image.open(cropped_image_path)
    #     cropped_img = cropped_img.convert('RGB')
    #
    #
    #     if self.transform is not None:
    #         cropped_img = self.transform(cropped_img)
    #
    #     return img, cropped_img, y_label


    def __len__(self):
        return self.length
