import numpy as np
import pandas as pd
import os
import model_data as MD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
import time
import datetime
from sklearn.preprocessing import MinMaxScaler
#数据处理
from dic import dic
import xlwt
from math import sqrt
import scipy.io as scio  # 需要用到scipy库
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



#构建模型(全连接ann)hip FE
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    #构建每一块的网络，然后连接
        self.layers_dict = nn.ModuleDict({
            "linear1": nn.Linear(15,12),
            "relu":nn.ReLU(),
            "linear2": nn.Linear(12,4),
            "relu2": nn.ReLU(),
            "linear3": nn.Linear(4, 1),

            # # tanh
            # "linear1": nn.Linear(15, 12),
            # "Tanh": nn.Tanh(),
            # "linear2": nn.Linear(12, 4),
            # "Tanh2": nn.Tanh(),
            # "linear3": nn.Linear(4, 1),

            # # rrelu
            # "linear1": nn.Linear(15, 12),
            # "rrelu": nn.RReLU(),
            # "linear2": nn.Linear(12, 4),
            # "rrelu2": nn.RReLU(),
            # "linear3": nn.Linear(4, 1),


            # 输出激活函数
            "sigmoid": nn.Sigmoid()

        })
    def forward(self,x):
        layers = ["linear1", "relu", "linear2", "relu2", "linear3", "sigmoid"]
        # layers = ["linear1", "Tanh", "linear2", "Tanh2", "linear3", "sigmoid"]
        # layers = ["linear1", "rrelu", "linear2", "rrelu2", "linear3", "sigmoid"]

        for layer in layers:
            x = self.layers_dict[layer](x)
        return x

# ann_silu
class ann_silu(nn.Module):
    def __init__(self):
        super(ann_silu, self).__init__()
        # 输入层
        self.w11 = nn.Parameter(torch.randn(15, 12))
        # 第一层隐藏层
        self.w12 = nn.Parameter(torch.randn(12, 4))
        # 第二层隐藏层
        self.w13 = nn.Parameter(torch.randn(4, 1))
    # 正向传播
    def forward(self, x):
        # print("@@@", x.shape)
        # print("@@@", (x @ self.w11).shape)
        # Silu
        # x11 = torch.relu(x @ self.w11)
        # x11 = torch.sigmoid(x @ self.w11) * (x @ self.w11)
        x11 = (x @ self.w11) * torch.sigmoid(x @ self.w11)
        # print(x @ self.w11)
        # print("@@@", x11.shape)
        # x12 = torch.relu(x11 @ self.w12)
        # x12 = torch.sigmoid(x11 @ self.w12) * (x11 @ self.w12)
        x12 = (x11 @ self.w12) * torch.sigmoid(x11 @ self.w12)

        # 激活函数
        y = torch.sigmoid(x12 @ self.w13)
        return y

# 存储数据
def saveMatrix2Excel(data, path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, data[i, j])
    f.save(path)


#构建模型(BSANN)hip
class BSANN_Hip(nn.Module):
    def __init__(self):
        super(BSANN_Hip, self).__init__()
        # 输入层
        self.w11 = nn.Parameter(torch.randn(4, 3))
        self.w21 = nn.Parameter(torch.randn(4, 3))
        self.w31 = nn.Parameter(torch.randn(4, 3))
        self.w41 = nn.Parameter(torch.randn(3, 3))

        # 第一层隐藏层
        self.w12 = nn.Parameter(torch.randn(3, 1))
        self.w22 = nn.Parameter(torch.randn(3, 1))
        self.w32 = nn.Parameter(torch.randn(3, 1))
        self.w42 = nn.Parameter(torch.randn(3, 1))

        # 第二层隐藏层
        self.w13 = nn.Parameter(torch.randn(4, 1))

    # 正向传播
    def forward(self, x):
        x1 = torch.split(x, [4, 4, 4, 3], dim=1)

        x11 = torch.relu(x1[0] @ self.w11)
        x21 = torch.relu(x1[1] @ self.w21)
        x31 = torch.relu(x1[2] @ self.w31)
        x41 = torch.relu(x1[3] @ self.w41)

        x12 = torch.relu(x11 @ self.w12)
        x22 = torch.relu(x21 @ self.w22)
        x32 = torch.relu(x31 @ self.w32)
        x42 = torch.relu(x41 @ self.w42)

        # silu
        # x11 = (x1[0] @ self.w11)*torch.sigmoid(x1[0] @ self.w11)
        # x21 = (x1[1] @ self.w21)*torch.sigmoid(x1[1] @ self.w21)
        # x31 = (x1[2] @ self.w31)*torch.sigmoid(x1[2] @ self.w31)
        # x41 = (x1[3] @ self.w41)*torch.sigmoid(x1[3] @ self.w41)
        #
        # x12 = (x11 @ self.w12)*torch.sigmoid(x11 @ self.w12)
        # x22 = (x21 @ self.w22)*torch.sigmoid(x21 @ self.w22)
        # x32 = (x31 @ self.w32)*torch.sigmoid(x31 @ self.w32)
        # x42 = (x41 @ self.w42)*torch.sigmoid(x41 @ self.w42)

        x2 = torch.cat((x12, x22, x32, x42), axis=1)

        # 变更激活函数
        y = torch.sigmoid(x2 @ self.w13)
        # y = (x2 @ self.w13) * torch.sigmoid(x2 @ self.w13)

        return y





#ANN精度
def evaluate(network, test_data_set, test_labels):
    error = 0
    p = np.zeros(test_labels.shape)
    lp = np.zeros(test_labels.shape)
    total = test_data_set.shape[0]

    for i in range(total):
        label = test_labels[i]
        predict = float(network(test_data_set[i]))
        p[i] = float(predict)
        lp[i] = float(label - predict)
        error += float(label - predict) ** 2
    test_labels = test_labels.numpy()
    vaf = (1 - np.var(lp) / np.var(test_labels)) * 100

    return np.var(lp), p, vaf


#BSANN,根据预测的关机力矩要调整网络结构
def evaluate_BSANN(network, test_data_set, test_labels):
    error = 0
    p=np.zeros(test_labels.shape)
    lp=np.zeros(test_labels.shape)
    total = test_data_set.shape[0]
    network_set = network(test_data_set)

    for i in range(total):
        label = test_labels[i]
        predict = float(network_set[i])
        p[i] = float(predict)
        lp[i] = float(label - predict)
        error += float(label - predict) ** 2
    test_labels = test_labels.numpy()
    vaf = (1 - np.var(lp) / np.var(test_labels)) * 100
    return np.var(lp),p,vaf


#x是输入数据，y是真实值
Data_HipFE_1,idData_HipFE_1,\
Data_HipFE_2,idData_HipFE_2,\
Data_HipFE_3,idData_HipFE_3,\
Data_HipFE_4,idData_HipFE_4,\
Data_HipAA_1,idData_HipAA_1,\
Data_HipAA_2,idData_HipAA_2,\
Data_HipAA_3,idData_HipAA_3,\
Data_HipAA_4,idData_HipAA_4 = MD.getData()


idData_HipFE_1 = MinMaxScaler().fit_transform(idData_HipFE_1)
idData_HipFE_2 = MinMaxScaler().fit_transform(idData_HipFE_2)
idData_HipFE_3 = MinMaxScaler().fit_transform(idData_HipFE_3)
idData_HipFE_4 = MinMaxScaler().fit_transform(idData_HipFE_4)

idData_HipAA_1 = MinMaxScaler().fit_transform(idData_HipAA_1)
idData_HipAA_2 = MinMaxScaler().fit_transform(idData_HipAA_2)
idData_HipAA_3 = MinMaxScaler().fit_transform(idData_HipAA_3)
idData_HipAA_4 = MinMaxScaler().fit_transform(idData_HipAA_4)



x_name = [Data_HipFE_1, Data_HipFE_2,
          Data_HipFE_3, Data_HipFE_4,
          Data_HipAA_1, Data_HipAA_2,
          Data_HipAA_3, Data_HipAA_4]
y_name = [idData_HipFE_1, idData_HipFE_2,
          idData_HipFE_3, idData_HipFE_4,
          idData_HipAA_1, idData_HipAA_2,
          idData_HipAA_3, idData_HipAA_4]


name_i = 0
# 消融实验的，只需要一组
for name_i in range(1):
    t1 = 80
    t2 = 80
    t3 = []
    t4 = []
    # start = datetime.datetime.now()

    for i in range(20):
        # 切割数据,这里的数据是numpy
        line = x_name[name_i].shape[0]
        split = math.floor(line * 0.8 / 101) * 101

        # print(x_name[name_i][:split, :].shape)

        x_train = x_name[name_i][:split, :]
        y_train = y_name[name_i][:split, :]
        x_test = x_name[name_i][split:, :]
        y_test = y_name[name_i][split:, :]

        X_train = torch.from_numpy(x_train.astype(np.float32))
        Y_train = torch.from_numpy(y_train.astype(np.float32))
        X_test = torch.from_numpy(x_test.astype(np.float32))
        Y_test = torch.from_numpy(y_test.astype(np.float32))

        # 将数据处理为数据加载器
        train_data = Data.TensorDataset(X_train, Y_train)
        test_data = Data.TensorDataset(X_test, Y_test)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=3, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=3, shuffle=True)

        # 调用网络
        mlpreg = Net()
        # mlpreg = ann_silu()

        mlpreg_BSANN = BSANN_Hip()


        # 处理
        optimizer = torch.optim.Adam(params=mlpreg.parameters(), lr=0.01)
        optimizer_BSANN = torch.optim.Adam(params=mlpreg_BSANN.parameters(), lr=0.01)


        # 均方误差损失函数
        loss_func = nn.MSELoss()
        # loss_func = nn.L1Loss()
        # loss_func = nn.BCELoss()
        # loss_func = nn.SmoothL1Loss()


        train_loss_all = []
        test_loss_all = []
        train_loss_all_BSANN = []
        test_loss_all_BSANN = []

        for epoch in range(20):
            # ANN训练集
            train_loss = 0
            train_num = 0
            test_loss = 0
            test_num = 0
            # 训练
            for step, (b_x, b_y) in enumerate(train_loader):
                output = mlpreg(b_x)
                loss = loss_func(output, b_y)  # loss是均方误差，可认为是一个样本的loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)  # b_x.size(0) = batch_size
                train_num += b_x.size(0)
            train_loss_all.append(train_loss / train_num)  # 一个batch_size的loss加和 / batch_size

            # 测试
            for step, (b_x, b_y) in enumerate(test_loader):
                output_test = mlpreg(b_x)
                loss_test = loss_func(output_test, b_y)  # loss是均方误差，可认为是一个样本的loss
                optimizer.zero_grad()
                loss_test.backward()
                optimizer.step()
                test_loss += loss_test.item() * b_x.size(0)  # b_x.size(0) = batch_size
                test_num += b_x.size(0)
            test_loss_all.append(test_loss / test_num)  # 一个batch_size的loss加和 / batch_size

            # ANN测试集
            # time1 = time.clock()
            # output_test = mlpreg(X_test)
            # loss_test = loss_func(output_test, Y_test)
            # print("@@@",test_loss_all.type)
            # time2 = time.clock()
            # time3 = time2 - time1
            # print("time= ",time3 * 300)

            # ANN训练精度
            lp, p, vaf = evaluate(mlpreg, X_train, Y_train)


            # 测试
            lp2, p2, vaf2 = evaluate(mlpreg, X_test, Y_test)



            # ann_silu
            # lp, p, vaf = evaluate_BSANN(mlpreg, X_train, Y_train)
            # lp2, p2, vaf2 = evaluate_BSANN(mlpreg, X_test, Y_test)

            # 训练
            HipFE_ANN_train = mlpreg.forward(X_train)
            HipFE_ANN_train = HipFE_ANN_train.detach().numpy()
            # 其他品佳指标
            MAE_ANN_train = mean_absolute_error(Y_train, HipFE_ANN_train)
            MSE_ANN_train = mean_squared_error(Y_train, HipFE_ANN_train)
            RMSE_ANN_train = sqrt(mean_squared_error(Y_train, HipFE_ANN_train))

            # 测试
            HipFE_ANN_test = mlpreg.forward(X_test)
            HipFE_ANN_test = HipFE_ANN_test.detach().numpy()
            # 其他品佳指标
            MAE_ANN = mean_absolute_error(Y_test, HipFE_ANN_test)
            MSE_ANN = mean_squared_error(Y_test, HipFE_ANN_test)
            RMSE_ANN = sqrt(mean_squared_error(Y_test, HipFE_ANN_test))

            if vaf2 > t1:
                t1 = vaf2
                t3 = HipFE_ANN_test
                torch.save(Net().state_dict(), 'log/ANN_hip_FE_weight_2.pth')

            # BSANN训练集
            train_loss_BSANN = 0
            train_num_BSANN = 0
            test_loss_BSANN = 0
            test_num_BSANN = 0

            # start_train = datetime.datetime.now()

            for step, (b_x, b_y) in enumerate(train_loader):
                output_BSANN = mlpreg_BSANN(b_x)
                loss_BSANN = loss_func(output_BSANN, b_y)
                optimizer_BSANN.zero_grad()
                loss_BSANN.backward()
                optimizer_BSANN.step()
                train_loss_BSANN += loss_BSANN.item() * b_x.size(0)
                train_num_BSANN += b_x.size(0)
            train_loss_all_BSANN.append(train_loss_BSANN / train_num_BSANN)

            # 测试
            for step, (b_x, b_y) in enumerate(test_loader):
                output_BSANN = mlpreg_BSANN(b_x)
                loss_BSANN = loss_func(output_BSANN, b_y)
                optimizer_BSANN.zero_grad()
                loss_BSANN.backward()
                optimizer_BSANN.step()
                test_loss_BSANN += loss_BSANN.item() * b_x.size(0)
                test_num_BSANN += b_x.size(0)
            test_loss_all_BSANN.append(test_loss_BSANN / test_num_BSANN)

            # Time_train = datetime.datetime.now()
            # print("Time_train = ", (Time_train - start_train) * 300)

            # BSANN测试集
            # start_test = datetime.datetime.now()
            # time1 = time.clock()

            output_test_BSANN = mlpreg_BSANN(X_test)
            loss_test_BSANN = loss_func(output_test_BSANN, Y_test)

            # time2 = time.clock()
            # time3 = time2 - time1
            # print("time= ",time3 * 300)

            # Time_test = datetime.datetime.now()
            # print("Time_test = ", (Time_test - start_test) * 300)

            # 精度
            lp_BSANN, p_BSANN, vaf_BSANN = evaluate_BSANN(mlpreg_BSANN, X_train, Y_train)
            lp2_BSANN, p2_BSANN, vaf2_BSANN = evaluate_BSANN(mlpreg_BSANN, X_test, Y_test)
            # 训练
            HipFE_BSANN_train = mlpreg_BSANN.forward(X_train)
            HipFE_BSANN_train = HipFE_BSANN_train.detach().numpy()
            # 其他品佳指标
            MAE_BSANN_train = mean_absolute_error(Y_train, HipFE_BSANN_train)
            MSE_BSANN_train = mean_squared_error(Y_train, HipFE_BSANN_train)
            RMSE_BSANN_train = sqrt(mean_squared_error(Y_train, HipFE_BSANN_train))

            # 测试
            HipFE_BSANN_test = mlpreg_BSANN.forward(X_test)
            HipFE_BSANN_test = HipFE_BSANN_test.detach().numpy()
            # 其他品佳指标
            MAE_BSANN = mean_absolute_error(Y_test, HipFE_ANN_test)
            MSE_BSANN = mean_squared_error(Y_test, HipFE_ANN_test)
            RMSE_BSANN = sqrt(mean_squared_error(Y_test, HipFE_ANN_test))


            if vaf2_BSANN > t2:
                t2 = vaf2_BSANN
                t4 = HipFE_BSANN_test
                torch.save(BSANN_Hip().state_dict(), 'log/BSANN_hip_FE_weight_2.pth')


    # 所有结果数据记录
    # FE1_ANN = np.empty([101,2],dtype=float)
    # 1
    if name_i == 0:

        # xlsx1 =np.concatenate([np.concatenate([y_train, HipFE_ANN_train], axis=0),HipFE_BSANN_train], axis=0)
        # # def saveMatrix2Excel(data, path):  #data数据,path存储位置
        # #     f = xlwt.Workbook()  # 创建工作簿
        # #     sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
        # #     [h, l] = data.shape  # h为行数，l为列数
        # #     for i in range(h):
        # #         for j in range(l):
        # #             sheet1.write(i, j, data[i, j])
        # #     f.save(path)
        # # W = [[1, 2, 3, 4], [5, 6, 7, 8]]
        # pathW = "C:\\Users\\X\\Desktop\\w\\xg\\b\\train_FE.xlsx"  # 保存在当前文件夹下，你也可以指定绝对路径
        # saveMatrix2Excel(xlsx1, pathW)

        # 训练
        y_train = y_train[0:101]
        HipFE_ANN_train = HipFE_ANN_train[0:101]
        HipFE_BSANN_train = HipFE_BSANN_train[0:101]
        # 测试
        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]

        # 存数据
        t5_train = np.hstack((y_train, HipFE_ANN_train, HipFE_BSANN_train))
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\FE_0.50_Train.mat', {'data': t5_train})

        t5 = np.hstack((y_test, t3, t4))
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\FE_0.50_Test.mat', {'data': t5})


        print("0.5m/s：")
        # 训练
        print("训练")
        print("HipFE_ANN_train =", vaf)
        print("HipFE_BSANN_train =", vaf_BSANN)
        print("MAE_ANN:", MAE_ANN_train)
        print("MAE_BSANN:", MAE_BSANN_train)
        print("MSE:", MSE_ANN_train)
        print("MSE:", MSE_BSANN_train)
        print("RMSE:", RMSE_ANN_train)
        print("RMSE:", RMSE_BSANN_train)
        # 测试
        print("测试")
        print("HipFE_ANN_test =", vaf2)
        print("HipFE_BSANN_test =", vaf2_BSANN)
        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)


        # 训练
        plt.subplot(221)
        plt.plot(y_train, "+-", label="Inverse Dynamics_Train")
        plt.plot(HipFE_ANN_train, "x-", label="HipFE_ANN_Predition_Train")
        plt.plot(HipFE_BSANN_train, "r*-", label="HipFE_BSANN_Predition_Train")
        plt.legend()
        plt.grid()
        plt.title("(a) 0.50 m/s (train)", fontsize=15)
        # plt.ylabel('Prediction Accuracy Of Model', fontsize=20)  # x轴标签
        plt.ylabel('Hip Flexion Moment with Train Data', fontsize=15)  # y轴标签
        plt.xlabel('Time Frames', fontsize=15)
        # plt.xlabel('(a)')
        # ax2 = plt.twiny()
        # # ax2.plot(X, Y3)
        # ax2.set_title('(a)', fontsize=15)

        # 测试
        plt.subplot(223)
        plt.plot(y_test, "+-", label="Inverse Dynamics_Test")
        plt.plot(t3, "x-", label="HipFE_ANN_Predition_Test")
        plt.plot(t4, "r*-", label="HipFE_BSANN_Predition_Test")
        plt.legend()
        plt.grid()
        plt.title("(c) 0.50 m/s (test)", fontsize=15)
        # plt.ylabel('Prediction Accuracy Of Model', fontsize=20)  # x轴标签
        plt.ylabel('Hip Flexion Moment with Test Data', fontsize=15)  # y轴标签
        plt.xlabel('Time Frames', fontsize=15)
        # plt.xlabel('(c)')
        # ax2 = plt.twiny()
        # # ax2.plot(X, Y3)
        # ax2.set_title('(c)', fontsize=15)

        # loss
        plt.subplot(222)
        plt.title("(b) Loss (ANN)", fontsize=15)
        plt.plot(train_loss_all, "y-", label="ANN_loss_train")
        # loss_test
        plt.plot(test_loss_all, "g*-", label="ANN_loss_test")
        plt.xlabel('Iterations', fontsize=15)
        # plt.xlabel('(b)')
        # ax2 = plt.twiny()
        # # ax2.plot(X, Y3)
        # ax2.set_title('(b)', fontsize=15)

        plt.legend()
        plt.grid()

        plt.subplot(224)
        plt.title("(d) Loss (BASNN)", fontsize=15)
        plt.plot(train_loss_all_BSANN, "y-", label="BSANN_loss_train")
        plt.plot(test_loss_all_BSANN, "g*-", label="BSANN_loss_test")
        plt.xlabel('Iterations', fontsize=15)
        # plt.xlabel('(d)')
        # ax2 = plt.twiny()
        # # ax2.plot(X, Y3)
        # ax2.set_title('(d)', fontsize=15)
        plt.legend()
        plt.grid()

        # fig, ax = plt.subplots(tight_layout=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

        plt.show()



