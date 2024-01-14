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
import h5py
import numpy as np
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

    datac = "E:\\1\\joint_ moment\\joint_ moment\\07_levelground.csv"
    datac_ramp = "E:\\1\\joint_ moment\\joint_ moment\\07_ramp.csv"



    datac = pd.read_csv(datac, header=0)
    datac_ramp = pd.read_csv(datac_ramp, header=0)


    idData_HipFE = datac.iloc[:, 1:2]
    idData_HipAA = datac.iloc[:, 2:3]

    idData_HipFE_ramp = datac_ramp.iloc[:, 1:2]
    idData_HipAA_ramp = datac_ramp.iloc[:, 2:3]


    # Hip FE
    Data_HipFE = np.concatenate([datac.iloc[:, 14:15], datac.iloc[:, 5:8],
                                 datac.iloc[:, 12:13], datac.iloc[:, 5:8],
                                 datac.iloc[:, 14:15], datac.iloc[:, 5:8],
                                 datac.iloc[:, 15:16], datac.iloc[:, 5:7]], axis=1)

    Data_HipFE_ramp = np.concatenate([datac_ramp.iloc[:, 14:15], datac_ramp.iloc[:, 5:8],
                                      datac_ramp.iloc[:, 12:13], datac_ramp.iloc[:, 5:8],
                                      datac_ramp.iloc[:, 14:15], datac_ramp.iloc[:, 5:8],
                                      datac_ramp.iloc[:, 15:16], datac_ramp.iloc[:, 5:7]], axis=1)

    # Hip AA
    Data_HipAA = np.concatenate([datac.iloc[:, 14:15], datac.iloc[:, 5:8],
                                 datac.iloc[:, 12:13], datac.iloc[:, 5:8],
                                 datac.iloc[:, 14:15], datac.iloc[:, 5:8],
                                 datac.iloc[:, 15:16], datac.iloc[:, 5:7]], axis=1)

    Data_HipAA_ramp = np.concatenate([datac_ramp.iloc[:, 14:15], datac_ramp.iloc[:, 5:8],
                                      datac_ramp.iloc[:, 12:13], datac_ramp.iloc[:, 5:8],
                                      datac_ramp.iloc[:, 14:15], datac_ramp.iloc[:, 5:8],
                                      datac_ramp.iloc[:, 15:16], datac_ramp.iloc[:, 5:7]], axis=1)


    Data_HipFE.astype("float")
    Data_HipAA.astype("float")
    Data_HipFE_ramp.astype("float")
    Data_HipAA_ramp.astype("float")

    idData_HipFE.astype("float")
    idData_HipAA.astype("float")
    idData_HipFE_ramp.astype("float")
    idData_HipAA_ramp.astype("float")


    return Data_HipFE,idData_HipFE, Data_HipAA,idData_HipAA, \
           Data_HipFE_ramp, idData_HipFE_ramp,\
           Data_HipAA_ramp, idData_HipAA_ramp
