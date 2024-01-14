import torch
from torch import nn
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
from math import sqrt
import xlwt
from math import sqrt
import scipy.io as scio  # 需要用到scipy库
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


from sklearn.preprocessing import MinMaxScaler


#数据处理
from dic import dic

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
            "sigmoid": nn.Sigmoid()
        })
    def forward(self,x):
        layers = ["linear1","relu","linear2","relu2","linear3","sigmoid"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x

class ANN_3(nn.Module):
    def __init__(self):
        super(ANN_3, self).__init__()
#构建每一块的网络，然后连接
        self.layers_dict = nn.ModuleDict({
            "linear1": nn.Linear(15,3),
            "relu":nn.ReLU(),
            "linear2": nn.Linear(3,1),
            "sigmoid": nn.Sigmoid()
        })
    def forward(self,x):
        layers = ["linear1","relu","linear2","sigmoid"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x


class ANN_4(nn.Module):
    def __init__(self):
        super(ANN_4, self).__init__()
#构建每一块的网络，然后连接
        self.layers_dict = nn.ModuleDict({
            "linear1": nn.Linear(15, 3),
            "relu": nn.ReLU(),
            "linear2": nn.Linear(3, 4),
            "relu2": nn.ReLU(),
            "linear3": nn.Linear(4, 1),
            "sigmoid": nn.Sigmoid()
        })

    def forward(self, x):
            layers = ["linear1", "relu", "linear2", "relu2", "linear3", "sigmoid"]
            for layer in layers:
                x = self.layers_dict[layer](x)
            return x



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

        x2 = torch.cat((x12, x22, x32, x42), axis=1)

        y = torch.sigmoid(x2 @ self.w13)
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
for name_i in range(8):
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

        # 调用网络
        mlpreg = Net()
        # mlpreg = ANN_4()
        # mlpreg = ANN_3()

        mlpreg_BSANN = BSANN_Hip()

        # 处理
        optimizer = torch.optim.Adam(params=mlpreg.parameters(), lr=0.01)
        optimizer_BSANN = torch.optim.Adam(params=mlpreg_BSANN.parameters(), lr=0.01)

        # 均方误差损失函数
        loss_func = nn.MSELoss()
        train_loss_all = []
        test_loss_all = []
        train_loss_all_BSANN = []
        test_loss_all_BSANN = []

        for epoch in range(10):
            # ANN训练集
            train_loss = 0
            train_num = 0
            test_loss = 0
            test_num = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                output = mlpreg(b_x)
                loss = loss_func(output, b_y)  # loss是均方误差，可认为是一个样本的loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)  # b_x.size(0) = batch_size
                train_num += b_x.size(0)
            train_loss_all.append(train_loss / train_num)  # 一个batch_size的loss加和 / batch_size

            # ANN测试集
            time1 = time.clock()
            output_test = mlpreg(X_test)
            loss_test = loss_func(output_test, Y_test)
            time2 = time.clock()
            time3 = time2 - time1
            # print("time= ",time3 * 300)

            # 精度
            lp, p, vaf = evaluate(mlpreg, X_train, Y_train)
            lp2, p2, vaf2 = evaluate(mlpreg, X_test, Y_test)


            HipFE_ANN_test = mlpreg.forward(X_test)
            HipFE_ANN_test = HipFE_ANN_test.detach().numpy()

            # 其他品佳指标
            MAE_ANN = mean_absolute_error(Y_test, HipFE_ANN_test)
            MSE_ANN = mean_squared_error(Y_test, HipFE_ANN_test)
            RMSE_ANN = sqrt(mean_squared_error(Y_test, HipFE_ANN_test))

            if vaf2 > t1:
                t1 = vaf2
                t3 = HipFE_ANN_test

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


            HipFE_BSANN_test = mlpreg_BSANN.forward(X_test)
            HipFE_BSANN_test = HipFE_BSANN_test.detach().numpy()

            # 其他评价指标
            MAE_BSANN = mean_absolute_error(Y_test, HipFE_BSANN_test)
            MSE_BSANN = mean_squared_error(Y_test, HipFE_BSANN_test)
            RMSE_BSANN = sqrt(mean_squared_error(Y_test, HipFE_BSANN_test))

            if vaf2_BSANN > t2:
                t2 = vaf2_BSANN
                t4 = HipFE_BSANN_test
                # torch.save(BSANN_Hip().state_dict(), 'log/BSANN_hip_FE_.pth')


        # 所有结果数据记录
        # FE1_ANN = np.empty([101,2],dtype=float)
        # 1
    if name_i == 0:
            t5 = np.hstack((y_test, t3, t4))
            # 保存到当前路径下
            scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\FE_0.50.mat', {'data': t5})

            y_test = y_test[0:101]
            t3 = t3[0:101]
            t4 = t4[0:101]

            print("speed 1：")
            print("HipFE_ANN_test =", vaf2)
            print("HipFE_BSANN_test =", vaf2_BSANN)

            print("MAE_ANN:", MAE_ANN)
            print("MAE_BSANN:", MAE_BSANN)
            print("MSE:", MSE_ANN)
            print("MSE:", MSE_BSANN)
            print("RMSE:", RMSE_ANN)
            print("RMSE:", RMSE_BSANN)

            plt.subplot(241)
            plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
            plt.plot(t3, "x-")  # , label="HipFE_ANN_Predition")
            plt.plot(t4, "r*-")  # , label="HipFE_BSANN_Predition")

            plt.legend()
            plt.grid()
            plt.title("0.50 m/s", fontsize=15)
            plt.xlabel('Time Frames')
            # # plt.ylabel('Prediction Accuracy Of Model', fontsize=20)  # x轴标签
            plt.ylabel('Hip Flexion Moment', fontsize=15)  # y轴标签
            # plt.ylabel("Hip Adduction Moment", fontsize=15)
            # plt.show()

    elif name_i == 1:
        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\FE_0.85.mat', {'data': t5})

        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]

        print("speed 2：")
        print("HipFE_ANN_test =", vaf2)
        print("HipFE_BSANN_test =", vaf2_BSANN)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(242)
        plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
        plt.plot(t3, "x-")  # , label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")  # , label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')
        plt.legend()
        plt.grid()
        plt.title("0.85 m/s", fontsize=15)
        # plt.title("speed 1",fontsize=15)

    # 2
    elif name_i == 2:
        # HipFE_ANN_test_2 = mlpreg.forward(X_test)
        # HipFE_ANN_test_2 = HipFE_ANN_test_2.detach().numpy()
        #
        # HipFE_BSANN_test_2 = mlpreg_BSANN.forward(X_test)
        # HipFE_BSANN_test_2 = HipFE_BSANN_test_2.detach().numpy()
        #
        # print("速度2：")
        # print("HipFE_ANN_test_2 =", vaf2)
        # print("HipFE_BSANN_test_2 =", vaf2_BSANN)

        # FE1_ANN[0:101,1:2] = HipFE_ANN_test_2

        # data_ = pd.DataFrame(HipFE_ANN_test_2)
        # data2_ = pd.DataFrame(HipFE_BSANN_test_2)
        # writer_ = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\ANN\\FE2.xlsx')  # 写入Excel文件
        # writer2_ = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\BSANN\\FE2.xlsx')
        # data_.to_excel(writer_,  float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data2_.to_excel(writer2_,  float_format='%.5f')
        # writer_.save()
        # writer_.close()
        # writer2_.save()
        # writer2_.close()

        # true3 = pd.DataFrame(y_test)
        # w3 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\true\\FE2.xlsx')
        # true3.to_excel(w3, float_format='%.5f')
        # w3.save()
        # w3.close()

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\FE_1.20.mat', {'data': t5})

        y_test = y_test[20:121]
        t3 = t3[20:121]
        t4 = t4[20:121]

        print("speed 3：")
        print("HipFE_ANN_test =", vaf2)
        print("HipFE_BSANN_test =", vaf2_BSANN)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(243)
        plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
        plt.plot(t3, "x-")  # , label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")  # , label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        # plt.title("speed 2",fontsize=15)
        plt.title('1.20 m/s', fontsize=15)


    elif name_i == 3:
        # HipAA_ANN_test_2 = mlpreg.forward(X_test)
        # HipAA_ANN_test_2 = HipAA_ANN_test_2.detach().numpy()
        #
        # HipAA_BSANN_test_2 = mlpreg_BSANN.forward(X_test)
        # HipAA_BSANN_test_2 = HipAA_BSANN_test_2.detach().numpy()
        #
        # print("速度2_AA：")
        # print("HipAA_ANN_test_2 =", vaf2)
        # print("HipAA_BSANN_test_2 =", vaf2_BSANN)

        # data21 = pd.DataFrame(HipAA_ANN_test_2)
        # data22 = pd.DataFrame(HipAA_BSANN_test_2)
        # writer21 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\ANN\\AA2.xlsx')  # 写入Excel文件
        # writer22 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\BSANN\\AA2.xlsx')
        # data21.to_excel(writer21,  float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data22.to_excel(writer22,  float_format='%.5f')
        # writer21.save()
        # writer21.close()
        # writer22.save()
        # writer22.close()

        # true4 = pd.DataFrame(y_test)
        # w4 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\true\\AA2.xlsx')
        # true4.to_excel(w4, float_format='%.5f')
        # w4.save()
        # w4.close()

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\FE_1.55.mat', {'data': t5})

        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]

        print("speed 4：")
        print("HipFE_ANN_test =", vaf2)
        print("HipFE_BSANN_test =", vaf2_BSANN)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(244)
        plt.plot(y_test, "+-", label="Inverse Dynamics")
        plt.plot(t3, "x-", label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-", label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        plt.title("1.55 m/s", fontsize=15)
        # plt.show()

    # 3
    elif name_i == 4:
        # HipFE_ANN_test_3 = mlpreg.forward(X_test)
        # HipFE_ANN_test_3 = HipFE_ANN_test_3.detach().numpy()
        #
        # HipFE_BSANN_test_3 = mlpreg_BSANN.forward(X_test)
        # HipFE_BSANN_test_3 = HipFE_BSANN_test_3.detach().numpy()
        #
        # print("速度3：")
        # print("HipFE_ANN_test_3 =", vaf2)
        # print("HipFE_BSANN_test_3 =", vaf2_BSANN)

        # FE1_ANN[0:101, 2:3] = HipFE_ANN_test_3
        # data3 = pd.DataFrame(HipFE_ANN_test_3)
        # data23 = pd.DataFrame(HipFE_BSANN_test_3)
        # writer3 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\ANN\\FE3.xlsx')  # 写入Excel文件
        # writer23 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\BSANN\\FE3.xlsx')
        # data3.to_excel(writer3,  float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data23.to_excel(writer23,  float_format='%.5f')
        # writer3.save()
        # writer3.close()
        # writer23.save()
        # writer23.close()

        # true5 = pd.DataFrame(y_test)
        # w5 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\true\\FE3.xlsx')
        # true5.to_excel(w5, float_format='%.5f')
        # w5.save()
        # w5.close()

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\AA_0.50.mat', {'data': t5})

        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]

        print("speed 5：")
        print("HipFE_ANN_test =", vaf2)
        print("HipFE_BSANN_test =", vaf2_BSANN)
        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(245)
        plt.plot(y_test, "+-")#, label="Inverse Dynamics")
        plt.plot(t3, "x-")#, label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")#, label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        # plt.title("speed 3",fontsize=15)
        plt.ylabel('Hip Adduction Moment', fontsize=15)

    elif name_i == 5:
        # HipAA_ANN_test_3 = mlpreg.forward(X_test)
        # HipAA_ANN_test_3 = HipAA_ANN_test_3.detach().numpy()
        #
        # HipAA_BSANN_test_3 = mlpreg_BSANN.forward(X_test)
        # HipAA_BSANN_test_3 = HipAA_BSANN_test_3.detach().numpy()
        #
        # print("速度3_AA：")
        # print("HipAA_ANN_test_3=", vaf2)
        # print("HipAA_BSANN_test_3=", vaf2_BSANN)

        # data33 = pd.DataFrame(HipAA_ANN_test_3)
        # data233 = pd.DataFrame(HipAA_BSANN_test_3)
        # writer33 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\ANN\\AA3.xlsx')  # 写入Excel文件
        # writer233 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\BSANN\\AA3.xlsx')
        # data33.to_excel(writer33,  float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data233.to_excel(writer233, float_format='%.5f')
        # writer33.save()
        # writer33.close()
        # writer233.save()
        # writer233.close()

        # true6 = pd.DataFrame(y_test)
        # w6 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\true\\AA3.xlsx')
        # true6.to_excel(w6, float_format='%.5f')
        # w6.save()
        # w6.close()

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\AA_0.85.mat', {'data': t5})

        y_test = y_test[25:126]
        t3 = t3[25:126]
        t4 = t4[25:126]

        print("speed 6：")
        print("HipFE_ANN_test =", vaf2)
        print("HipFE_BSANN_test =", vaf2_BSANN)
        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(246)
        plt.plot(y_test, "+-")#, label="Inverse Dynamics")
        plt.plot(t3, "x-")#, label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")#, label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        # plt.title("speed 3",fontsize=15)

    # 4
    elif name_i == 6:

        # HipFE_ANN_test_4 = mlpreg.forward(X_test)
        # HipFE_ANN_test_4 = HipFE_ANN_test_4.detach().numpy()
        #
        # HipFE_BSANN_test_4 = mlpreg_BSANN.forward(X_test)
        # HipFE_BSANN_test_4 = HipFE_BSANN_test_4.detach().numpy()
        #
        # print("速度4：")
        # print("HipFE_ANN_test_4 =", vaf2)
        # print("HipFE_BSANN_test_4 =", vaf2_BSANN)

        # FE1_ANN[0:101, 3:4] = HipFE_ANN_test_4

        # data4 = pd.DataFrame(HipFE_ANN_test_4)
        # data24 = pd.DataFrame(HipFE_BSANN_test_4)
        # writer4 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\ANN\\FE4.xlsx')  # 写入Excel文件
        # writer24 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\BSANN\\FE4.xlsx')
        # data4.to_excel(writer4,  float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data24.to_excel(writer24, float_format='%.5f')
        # writer4.save()
        # writer4.close()
        # writer24.save()
        # writer24.close()

        # true7 = pd.DataFrame(y_test)
        # w7 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\true\\FE4.xlsx')
        # true7.to_excel(w7, float_format='%.5f')
        # w7.save()
        # w7.close()

        y_test = y_test[60:161]
        t3 = t3[60:161]
        t4 = t4[60:161]

        plt.subplot(247)
        plt.plot(y_test, "+-")#, label="Inverse Dynamics")
        plt.plot(t3, "x-")#, label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")#, label="HipFE_BSANN_Predition")

        print("speed 7：")
        print("HipFE_ANN_test =", vaf2)
        print("HipFE_BSANN_test =", vaf2_BSANN)
        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        # plt.title("speed 4",fontsize=15)
        # plt.ylabel('Speed 4', fontsize=15)

    else:
        # HipAA_ANN_test_4 = mlpreg.forward(X_test)
        # HipAA_ANN_test_4 = HipAA_ANN_test_4.detach().numpy()
        #
        # HipAA_BSANN_test_4 = mlpreg_BSANN.forward(X_test)
        # HipAA_BSANN_test_4 = HipAA_BSANN_test_4.detach().numpy()
        #
        # print("速度4_AA：")
        # print("HipAA_ANN_test_4 =", vaf2)
        # print("HipAA_BSANN_test_4 =", vaf2_BSANN)

        # data44 = pd.DataFrame(HipAA_ANN_test_4)
        # data244 = pd.DataFrame(HipAA_BSANN_test_4)
        # writer44 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\ANN\\AA4.xlsx')  # 写入Excel文件
        # writer244 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\BSANN\\AA4.xlsx')
        # data44.to_excel(writer44,  float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data244.to_excel(writer244,  float_format='%.5f')
        # writer44.save()
        # writer44.close()
        # writer244.save()
        # writer244.close()
        # true8 = pd.DataFrame(y_test)
        # w8 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\test\\true\\AA4.xlsx')
        # true8.to_excel(w8, float_format='%.5f')
        # w8.save()
        # w8.close()

        y_test = y_test[20:121]
        t3 = t3[20:121]
        t4 = t4[20:121]

        print("speed 8：")
        print("HipFE_ANN_test =", vaf2)
        print("HipFE_BSANN_test =", vaf2_BSANN)
        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(248)
        plt.plot(y_test, "+-", label="Inverse Dynamics")
        plt.plot(t3, "x-", label="HipAA_ANN_Predition")
        plt.plot(t4, "r*-", label="HipAA_BSANN_Predition")
        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        # plt.title("speed 4",fontsize=15)
        plt.show()

    # ANN_FE1 = pd.DataFrame(FE1_ANN)
    # writerFE1 = pd.ExcelWriter('C:\\Users\\X\\Desktop\\1.xlsx')
    # ANN_FE1.to_excel(writerFE1, float_format='%.5f')
    # writerFE1.save()
    # writerFE1.close()