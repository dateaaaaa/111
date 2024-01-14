import h5py
from matplotlib import axis
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import MinMaxScaler
import sys
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm
import torch.distributed as dist
import logging
import math
from torch.optim.lr_scheduler import LambdaLR


#筛选掉多余的emg信号
def emg_select(emg_data, id_data):
    idx = np.searchsorted(emg_data[:,0],id_data[:,0])
    emg_data = emg_data[idx]
    return emg_data

#剔除空值Nan,emg,id,ik
def remove_nan(emg, id, ik):
    idx = np.where(~np.isnan(id))[0]
    emg = emg[idx]
    id = id[idx]
    ik = ik[idx]
    return emg,id,ik


def getData():
    data1 = "E:\\paper\\test\\06\\1_treadmill_01_01.csv"
    data2 = "E:\\paper\\test\\06\\4_treadmill_01_01.csv"
    data3 = "E:\\paper\\test\\06\\2_treadmill_01_01.csv"
    data4 = "E:\\paper\\test\\06\\3_treadmill_01_01.csv"


    data1 = pd.read_csv(data1, header=0)
    data2 = pd.read_csv(data2, header=0)
    data3 = pd.read_csv(data3, header=0)
    data4 = pd.read_csv(data4, header=0)


    # data1 = data1.iloc[1:506, :]
    # data2 = data2.iloc[1:506, :]
    # data3 = data3.iloc[103:204, :]
    # data4 = data4.iloc[204:305, :]
    # print(data1.shape)
    # print(data4.shape)

    idData_HipFE_1 = data1.iloc[:, 1:2]
    idData_HipFE_2 = data2.iloc[:, 1:2]
    idData_HipFE_3 = data3.iloc[:, 1:2]
    idData_HipFE_4 = data4.iloc[:, 1:2]


    idData_HipAA_1 = data1.iloc[:, 2:3]
    idData_HipAA_2 = data2.iloc[:, 2:3]
    idData_HipAA_3 = data3.iloc[:, 2:3]
    idData_HipAA_4 = data4.iloc[:, 2:3]


    # Hip FE
    Data_HipFE_1 = np.concatenate([data1.iloc[:, 14:15], data1.iloc[:, 5:8],
                                   data1.iloc[:, 12:13], data1.iloc[:, 5:8],
                                   data1.iloc[:, 14:15], data1.iloc[:, 5:8],
                                   data1.iloc[:, 15:16], data1.iloc[:, 5:7]], axis=1)

    Data_HipFE_2 = np.concatenate([data2.iloc[:, 14:15], data2.iloc[:, 5:8],
                                   data2.iloc[:, 12:13], data2.iloc[:, 5:8],
                                   data2.iloc[:, 14:15], data2.iloc[:, 5:8],
                                   data2.iloc[:, 15:16], data2.iloc[:, 5:7]], axis=1)

    Data_HipFE_3 = np.concatenate([data3.iloc[:, 14:15], data3.iloc[:, 5:8],
                                   data3.iloc[:, 12:13], data3.iloc[:, 5:8],
                                   data3.iloc[:, 14:15], data3.iloc[:, 5:8],
                                   data3.iloc[:, 15:16], data3.iloc[:, 5:7]], axis=1)

    Data_HipFE_4 = np.concatenate([data4.iloc[:, 14:15], data4.iloc[:, 5:8],
                                   data4.iloc[:, 12:13], data4.iloc[:, 5:8],
                                   data4.iloc[:, 14:15], data4.iloc[:, 5:8],
                                   data4.iloc[:, 15:16], data4.iloc[:, 5:7]], axis=1)



    # Hip AA
    Data_HipAA_1 = np.concatenate([data1.iloc[:, 14:15], data1.iloc[:, 5:8],
                                   data1.iloc[:, 12:13], data1.iloc[:, 5:8],
                                   data1.iloc[:, 14:15], data1.iloc[:, 5:8],
                                   data1.iloc[:, 15:16], data1.iloc[:, 5:7]], axis=1)

    Data_HipAA_2 = np.concatenate([data2.iloc[:, 14:15], data2.iloc[:, 5:8],
                                   data2.iloc[:, 12:13], data2.iloc[:, 5:8],
                                   data2.iloc[:, 14:15], data2.iloc[:, 5:8],
                                   data2.iloc[:, 15:16], data2.iloc[:, 5:7]], axis=1)

    Data_HipAA_3 = np.concatenate([data3.iloc[:, 14:15], data3.iloc[:, 5:8],
                                   data3.iloc[:, 12:13], data3.iloc[:, 5:8],
                                   data3.iloc[:, 14:15], data3.iloc[:, 5:8],
                                   data3.iloc[:, 15:16], data3.iloc[:, 5:7]], axis=1)

    Data_HipAA_4 = np.concatenate([data4.iloc[:, 14:15], data4.iloc[:, 5:8],
                                   data4.iloc[:, 12:13], data4.iloc[:, 5:8],
                                   data4.iloc[:, 14:15], data4.iloc[:, 5:8],
                                   data4.iloc[:, 15:16], data4.iloc[:, 5:7]], axis=1)

    # FE
    Data_HipFE_1.astype("float")
    idData_HipFE_1.astype("float")
    Data_HipFE_2.astype("float")
    idData_HipFE_2.astype("float")
    Data_HipFE_3.astype("float")
    idData_HipFE_3.astype("float")
    Data_HipFE_4.astype("float")
    idData_HipFE_4.astype("float")


    # AA
    Data_HipAA_1.astype("float")
    idData_HipAA_1.astype("float")
    Data_HipAA_2.astype("float")
    idData_HipAA_2.astype("float")
    Data_HipAA_3.astype("float")
    idData_HipAA_3.astype("float")
    Data_HipAA_4.astype("float")
    idData_HipAA_4.astype("float")


    return Data_HipFE_1, idData_HipFE_1, Data_HipFE_2, idData_HipFE_2, \
           Data_HipFE_3, idData_HipFE_3, Data_HipFE_4, idData_HipFE_4,\
           Data_HipAA_1,idData_HipAA_1,Data_HipAA_2,idData_HipAA_2,\
           Data_HipAA_3,idData_HipAA_3,Data_HipAA_4,idData_HipAA_4

    # return Data_HipAA_1,idData_HipAA_1,Data_HipAA_2,idData_HipAA_2,\
    #        Data_HipAA_3,idData_HipAA_3,Data_HipAA_4,idData_HipAA_4