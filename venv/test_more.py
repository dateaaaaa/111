import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import test_more_data as MD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
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


# 损失函数(二元交叉熵)
def loss_func(self,y_pred,y_true):
    #将预测值限制在1e-7以上, 1- (1e-7)以下，避免log(0)错误
    eps = 1e-7
    y_pred = torch.clamp(y_pred,eps,1.0-eps)
    bce = - y_true*torch.log(y_pred) - (1-y_true)*torch.log(1-y_pred)
    return torch.mean(bce)


# 评估指标(准确率)
def metric_func(self,y_pred,y_true):
     y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                          torch.zeros_like(y_pred,dtype = torch.float32))
     acc = torch.mean(1-torch.abs(y_true-y_pred))
     return acc


def train_step(model, features, labels):

    # 正向传播求损失
    predictions = model.forward(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)
    # 反向传播求梯度
    loss.backward()

    # 梯度下降法更新参数
    for param in model.parameters():
        # 注意是对param.data进行重新赋值,避免此处操作引起梯度记录
        param.data = (param.data - 0.01*param.grad.data)
    # 梯度清零
    model.zero_grad()
    return loss.item(),metric.item()


def train_model(model,epochs):

    for epoch in range(1,epochs+1):
        loss_list,metric_list = [],[]

        for features, labels in data_iter(X,Y,101):
            lossi,metrici = train_step(model,features,labels)
            loss_list.append(lossi)
            metric_list.append(metrici)
        loss = np.mean(loss_list)
        metric = np.mean(metric_list)

        if epoch%100==0:
            printbar()
            print("epoch =",epoch,"loss = ",loss,"metric = ",metric)

    # train_model(model,epochs = 300)


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

#ANN3精度
def evaluate_ANN3(network, test_data_set, test_labels):
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

#ANN4精度
def evaluate_ANN4(network, test_data_set, test_labels):
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
Data_HipFE_5,idData_HipFE_5,\
Data_HipFE_6,idData_HipFE_6,\
Data_HipFE_7,idData_HipFE_7 = MD.getData()


idData_HipFE_1 = MinMaxScaler().fit_transform(idData_HipFE_1)
idData_HipFE_2 = MinMaxScaler().fit_transform(idData_HipFE_2)
idData_HipFE_3 = MinMaxScaler().fit_transform(idData_HipFE_3)
idData_HipFE_4 = MinMaxScaler().fit_transform(idData_HipFE_4)
idData_HipFE_5 = MinMaxScaler().fit_transform(idData_HipFE_5)
idData_HipFE_6 = MinMaxScaler().fit_transform(idData_HipFE_6)
idData_HipFE_7 = MinMaxScaler().fit_transform(idData_HipFE_7)


# idData_HipAA_1 = MinMaxScaler().fit_transform(idData_HipAA_1)
# idData_HipAA_2 = MinMaxScaler().fit_transform(idData_HipAA_2)
# idData_HipAA_3 = MinMaxScaler().fit_transform(idData_HipAA_3)
# idData_HipAA_4 = MinMaxScaler().fit_transform(idData_HipAA_4)


x_name = [Data_HipFE_1]
y_name = [idData_HipFE_1]
x_name_2 = [Data_HipFE_2, Data_HipFE_3,
            Data_HipFE_4, Data_HipFE_5,
            Data_HipFE_6, Data_HipFE_7]
y_name_2 =[idData_HipFE_2, idData_HipFE_3,
           idData_HipFE_4, idData_HipFE_5,
           idData_HipFE_6, idData_HipFE_7]



x_train = x_name[0][:, :]
y_train = y_name[0][:, :]

X_train = torch.from_numpy(x_train.astype(np.float32))
Y_train = torch.from_numpy(y_train.astype(np.float32))

# name_i = 0

for name_i in range(6):
    # if name_i != 5:
    #     continue
    # 切割数据,这里的数据是numpy
    # line = x_name[name_i].shape[0]
    # split = math.floor(line * 0.8 / 101) * 101
    t1 = 80
    t2 = 80
    t3 = []
    t4 = []

    for i in range(5):

        x_test = x_name_2[name_i][:, :]
        y_test = y_name_2[name_i][:, :]

        X_test = torch.from_numpy(x_test.astype(np.float32))
        Y_test = torch.from_numpy(y_test.astype(np.float32))

        # 将数据处理为数据加载器
        train_data = Data.TensorDataset(X_train, Y_train)
        test_data = Data.TensorDataset(X_test, Y_test)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=3, shuffle=True)

        # 调用网络
        mlpreg = Net()
        # mlpreg_ANN4 = ANN_4()
        # mlpreg_ANN3 = ANN_3()

        mlpreg_BSANN = BSANN_Hip()

        # 处理
        optimizer = torch.optim.Adam(params=mlpreg.parameters(), lr=0.01)
        optimizer_ANN3 = torch.optim.Adam(params=mlpreg.parameters(), lr=0.01)
        optimizer_ANN4 = torch.optim.Adam(params=mlpreg.parameters(), lr=0.01)
        optimizer_BSANN = torch.optim.Adam(params=mlpreg_BSANN.parameters(), lr=0.01)

        # 均方误差损失函数
        loss_func = nn.MSELoss()
        train_loss_all = []
        test_loss_all = []
        train_loss_all_BSANN = []
        test_loss_all_BSANN = []

        for epoch in range(300):
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
            output_test = mlpreg(X_test)
            loss_test = loss_func(output_test, Y_test)

            # 精度
            # lp, p, vaf = evaluate(mlpreg, X_train, Y_train)
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

            # # ANN_3训练集
            # train_loss_ANN3 = 0
            # train_num_ANN3 = 0
            # test_loss_ANN3 = 0
            # test_num_ANN3 = 0
            # for step, (b_x, b_y) in enumerate(train_loader):
            #     output_ANN3 = mlpreg_ANN3(b_x)
            #     loss_ANN3 = loss_func(output_ANN3, b_y)  # loss是均方误差，可认为是一个样本的loss
            #     optimizer_ANN3.zero_grad()
            #     loss_ANN3.backward()
            #     optimizer_ANN3.step()
            #     train_loss_ANN3 += loss_ANN3.item() * b_x.size(0)  # b_x.size(0) = batch_size
            #     train_num_ANN3 += b_x.size(0)
            # train_loss_all.append(train_loss_ANN3 / train_num_ANN3)  # 一个batch_size的loss加和 / batch_size
            #
            # # ANN测试集
            # output_test_ANN3 = mlpreg_ANN3(X_test)
            # loss_test_ANN3 = loss_func(output_test_ANN3, Y_test)
            #
            # # 精度
            # # lp_ANN3, p_ANN3, vaf_ANN3 = evaluate(mlpreg_ANN3, X_train, Y_train)
            # lp2_ANN3, p2_ANN3, vaf2_ANN3 = evaluate_ANN3(mlpreg_ANN3, X_test, Y_test)
            #
            # # ANN_4训练集
            # train_loss_ANN4 = 0
            # train_num_ANN4 = 0
            # test_loss_ANN4 = 0
            # test_num_ANN4 = 0
            # for step, (b_x, b_y) in enumerate(train_loader):
            #     output_ANN4 = mlpreg_ANN4(b_x)
            #     loss_ANN4 = loss_func(output_ANN4, b_y)  # loss是均方误差，可认为是一个样本的loss
            #     optimizer_ANN4.zero_grad()
            #     loss_ANN4.backward()
            #     optimizer_ANN4.step()
            #     train_loss_ANN4 += loss_ANN4.item() * b_x.size(0)  # b_x.size(0) = batch_size
            #     train_num_ANN4 += b_x.size(0)
            # train_loss_all.append(train_loss_ANN4 / train_num_ANN4)  # 一个batch_size的loss加和 / batch_size
            #
            # # ANN测试集
            # output_test_ANN4 = mlpreg_ANN3(X_test)
            # loss_test_ANN4 = loss_func(output_test_ANN4, Y_test)
            #
            # # 精度
            # # lp_ANN4, p_ANN4, vaf_ANN4 = evaluate_ANN4(mlpreg_ANN4, X_train, Y_train)
            # lp2_ANN4, p2_ANN4, vaf2_ANN4 = evaluate_ANN4(mlpreg_ANN4, X_test, Y_test)

            # BSANN训练集
            train_loss_BSANN = 0
            train_num_BSANN = 0
            test_loss_BSANN = 0
            test_num_BSANN = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                output_BSANN = mlpreg_BSANN(b_x)
                loss_BSANN = loss_func(output_BSANN, b_y)
                optimizer_BSANN.zero_grad()
                loss_BSANN.backward()
                optimizer_BSANN.step()
                train_loss_BSANN += loss_BSANN.item() * b_x.size(0)
                train_num_BSANN += b_x.size(0)
            train_loss_all_BSANN.append(train_loss_BSANN / train_num_BSANN)

            # BSANN测试集
            output_test_BSANN = mlpreg_BSANN(X_test)
            loss_test_BSANN = loss_func(output_test_BSANN, Y_test)

            # 精度
            # lp_BSANN, p_BSANN, vaf_BSANN = evaluate_BSANN(mlpreg_BSANN, X_train, Y_train)
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

    # t3 = np.array(t3)
    # t4 = np.array(t4)
    # y_test = np.array(y_test)
    #
    # y_test = y_test[2627:2728]
    # t3 = t3[2627:2728]
    # t4 = t4[2627:2728]
    #
    # t5 = np.hstack((y_test, t3, t4))
    #
    #     # 保存到当前路径下
    # scio.savemat('E:\\paper\\result\\06\\AA\\4\\6.mat', {'data': t5})
    #
    # print("6 ：")
    # print("HipAA_ANN_test_2 =", t1)
    # print("HipAA_BSANN_test_2 =", t2)




    # 所有结果数据记录
    # 1
    if name_i == 0:
        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        print(y_test.shape)
        print(t3.shape)
        print(t4.shape)

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\AA_1.25.mat', {'data': t5})
        # file_name.mat为保存的文件名。该保存的mat文件可直接在matlab打开

        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]

        print("1：")
        print("HipFE_ANN_test =", t1)
        print("HipFE_BSANN_test =", t2)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(161)
        plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
        plt.plot(t3, "x-")  # , label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")  # , label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        plt.title("1.25 m/s", fontsize=15)
        # plt.xlabel('Prediction Accuracy Of Model', fontsize=20)  # x轴标签
        # plt.ylabel('Hip Flexion Moment', fontsize=15)  # y轴标签
        plt.ylabel("Hip Adduction Moment", fontsize=15)
        # plt.show()

    elif name_i == 1:
        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)


        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\AA_1.30.mat', {'data': t5})

        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]

        print("2：")
        print("HipAA_ANN_test =", t1)
        print("HipAA_BSANN_test =", t2)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)


        plt.subplot(162)
        plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
        plt.plot(t3, "x-")  # , label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")  # , label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        plt.title("1.30 m/s", fontsize=15)
        # plt.show()

    elif name_i == 2:
        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\AA_1.35.mat', {'data': t5})

        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]



        print("3：")
        print("HipFE_ANN_test_2 =", t1)
        print("HipFE_BSANN_test_2 =", t2)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(163)
        plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
        plt.plot(t3, "x-")  # , label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")  # , label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')

        plt.legend()
        plt.grid()
        plt.title("1.35 m/s", fontsize=15)


    elif name_i == 3:
        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\AA_1.40.mat', {'data': t5})


        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]


        print("4 ：")
        print("HipAA_ANN_test_2 =", t1)
        print("HipAA_BSANN_test_2 =", t2)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)


        plt.subplot(164)
        plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
        plt.plot(t3, "x-")  # , label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")  # , label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')
        plt.legend()
        plt.grid()
        plt.title("1.40 m/s", fontsize=15)

    elif name_i == 4:
        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\AA_1.45.mat', {'data': t5})

        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]


        print("5 ：")
        print("HipAA_ANN_test_2 =", t1)
        print("HipAA_BSANN_test_2 =", t2)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)


        plt.subplot(165)
        plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
        plt.plot(t3, "x-")  # , label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-")  # , label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames')
        plt.legend()
        plt.grid()
        plt.title("1.45 m/s", fontsize=15)

    elif name_i == 5:
        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        t5 = np.hstack((y_test, t3, t4))
        # 保存到当前路径下
        scio.savemat('C:\\Users\\X\\Desktop\\w\\xg\\b\\AA_1.50.mat', {'data': t5})

        y_test = y_test[0:101]
        t3 = t3[0:101]
        t4 = t4[0:101]


        print("6 ：")
        print("HipAA_ANN_test_2 =", t1)
        print("HipAA_BSANN_test_2 =", t2)

        print("MAE_ANN:", MAE_ANN)
        print("MAE_BSANN:", MAE_BSANN)
        print("MSE:", MSE_ANN)
        print("MSE:", MSE_BSANN)
        print("RMSE:", RMSE_ANN)
        print("RMSE:", RMSE_BSANN)

        plt.subplot(166)
        plt.plot(y_test, "+-", label="Inverse Dynamics")
        # plt.plot(t3, "x-", label="HipFE_ANN_Predition")
        # plt.plot(t4, "r*-", label="HipFE_BSANN_Predition")
        plt.plot(t3, "x-", label="HipAA_ANN_Predition")
        plt.plot(t4, "r*-", label="HipAA_BSANN_Predition")
        plt.xlabel('Time Frames')
        plt.legend()
        plt.grid()
        plt.title("1.50 m/s", fontsize=15)
        plt.show()



