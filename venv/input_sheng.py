import h5py
import numpy as np
import pandas as pd




def getData():
    data1 = "E:\\2\\data_sheng\\Sub07\\def\\Sub07_UPS_Data_2.csv"
    # data1 = "E:\\2\\data_sheng\\Sub07\\Sub07_KLFT_Data.csv"



    data1 = pd.read_csv(data1, header=0)



    # data1 = data1.iloc[1:506, :]
    # data2 = data2.iloc[1:506, :]
    # data3 = data3.iloc[103:204, :]
    # data4 = data4.iloc[204:305, :]
    # print(data1.shape)
    # print(data4.shape)

    idData_HipFE_r = data1.iloc[:, 1:2]
    idData_HipAA_r = data1.iloc[:, 2:3]
    idData_AnklePDF_r = data1.iloc[:, 4:5]
    idData_KneeFE_r = data1.iloc[:, 3:4]
    idData_HipFE_1 = data1.iloc[:, 5:6]
    idData_HipAA_1 = data1.iloc[:, 6:7]
    idData_AnklePDF_1 = data1.iloc[:, 7:8]
    idData_KneeFE_1 = data1.iloc[:, 8:9]






    # Left
    # 4 4
    Data_HipFE_1 = np.concatenate([data1.iloc[:, 18:19], data1.iloc[:, 13:15], data1.iloc[:, 16:17],
                                   data1.iloc[:, 20:21],data1.iloc[:, 13:15], data1.iloc[:, 16:17]], axis=1)

    # 4 4
    Data_HipAA_1 = np.concatenate([data1.iloc[:, 18:19], data1.iloc[:, 13:15], data1.iloc[:, 16:17],
                                   data1.iloc[:, 20:21], data1.iloc[:, 13:15], data1.iloc[:, 16:17]], axis=1)

    # 2 2 3 3 2
    Data_AnklePDF_1 = np.concatenate([data1.iloc[:, 21:22], data1.iloc[:, 15:16],
                                      data1.iloc[:, 22:23], data1.iloc[:, 15:16],
                                      data1.iloc[:, 23:24], data1.iloc[:, 15:17],
                                      data1.iloc[:, 24:25], data1.iloc[:, 15:17],
                                      data1.iloc[:, 25:26], data1.iloc[:, 15:16]], axis=1)


    # 4 2 4 3 3
    Data_KneeFE_1 = np.concatenate([data1.iloc[:, 18:19], data1.iloc[:, 13:15], data1.iloc[:, 16:17],
                                    data1.iloc[:, 19:20], data1.iloc[:, 16:17],
                                    data1.iloc[:, 20:21], data1.iloc[:, 13:15], data1.iloc[:, 16:17],
                                    data1.iloc[:, 23:24], data1.iloc[:, 15:17],
                                    data1.iloc[:, 24:25], data1.iloc[:, 15:17]], axis=1)

    # right
    Data_HipFE_r = np.concatenate([data1.iloc[:, 18:19], data1.iloc[:, 9:11], data1.iloc[:, 12:13],
                                   data1.iloc[:, 20:21], data1.iloc[:, 9:11], data1.iloc[:, 12:13]], axis=1)

    # 4 4
    Data_HipAA_r = np.concatenate([data1.iloc[:, 18:19], data1.iloc[:, 9:11], data1.iloc[:, 12:13],
                                   data1.iloc[:, 20:21], data1.iloc[:, 9:11], data1.iloc[:, 12:13]], axis=1)

    # 2 2 3 3 2
    Data_AnklePDF_r = np.concatenate([data1.iloc[:, 21:22], data1.iloc[:, 11:12],
                                      data1.iloc[:, 22:23], data1.iloc[:, 11:12],
                                      data1.iloc[:, 23:24], data1.iloc[:, 11:13],
                                      data1.iloc[:, 24:25], data1.iloc[:, 11:13],
                                      data1.iloc[:, 25:26], data1.iloc[:, 11:12]], axis=1)

    # 4 2 4 3 3
    Data_KneeFE_r = np.concatenate([data1.iloc[:, 18:19], data1.iloc[:, 9:11], data1.iloc[:, 12:13],
                                    data1.iloc[:, 19:20], data1.iloc[:, 12:13],
                                    data1.iloc[:, 20:21], data1.iloc[:, 9:11], data1.iloc[:, 12:13],
                                    data1.iloc[:, 23:24], data1.iloc[:, 11:13],
                                    data1.iloc[:, 24:25], data1.iloc[:, 11:13]], axis=1)



    # left
    Data_HipFE_1.astype("float")
    idData_HipFE_1.astype("float")
    Data_HipAA_1.astype("float")
    idData_HipAA_1.astype("float")
    Data_AnklePDF_1.astype("float")
    idData_AnklePDF_1.astype("float")
    Data_KneeFE_1.astype("float")
    idData_KneeFE_1.astype("float")

    # right
    Data_HipFE_r.astype("float")
    idData_HipFE_r.astype("float")
    Data_HipAA_r.astype("float")
    idData_HipAA_r.astype("float")
    Data_AnklePDF_r.astype("float")
    idData_AnklePDF_r.astype("float")
    Data_KneeFE_r.astype("float")
    idData_KneeFE_r.astype("float")





    # return Data_HipFE_1, idData_HipFE_1,\
    #        Data_HipAA_1, idData_HipAA_1,\
    #        Data_AnklePDF_1, idData_AnklePDF_1,\
    #        Data_KneeFE_1, idData_KneeFE_1

    return Data_HipFE_r, idData_HipFE_r, \
           Data_HipAA_r, idData_HipAA_r, \
           Data_AnklePDF_r, idData_AnklePDF_r, \
           Data_KneeFE_r, idData_KneeFE_r

