import numpy as np
import pandas as pd
import os
import input_sheng as MD
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



#构建模型(全连接ann)hip FE hip AA
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#构建每一块的网络，然后连接
        self.layers_dict = nn.ModuleDict({
            "linear1": nn.Linear(8,6),
            "relu":nn.ReLU(),
            "linear2": nn.Linear(6,2),
            "relu2": nn.ReLU(),
            "linear3": nn.Linear(2, 1),
            "sigmoid": nn.Sigmoid()
        })
    def forward(self,x):
        layers = ["linear1","relu","linear2","relu2","linear3","sigmoid"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x

class ANN_Ankle(nn.Module):
    def __init__(self):
        super(ANN_Ankle, self).__init__()
#构建每一块的网络，然后连接
        self.layers_dict = nn.ModuleDict({
            "linear1": nn.Linear(12,15),
            "relu":nn.ReLU(),
            "linear2": nn.Linear(15,5),
            "relu2": nn.ReLU(),
            "linear3": nn.Linear(5, 1),
            "sigmoid": nn.Sigmoid()
        })
    def forward(self,x):
        layers = ["linear1","relu","linear2","relu2","linear3","sigmoid"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x


class ANN_Knee(nn.Module):
    def __init__(self):
        super(ANN_Knee, self).__init__()
#构建每一块的网络，然后连接
        self.layers_dict = nn.ModuleDict({
            "linear1": nn.Linear(16,15),
            "relu":nn.ReLU(),
            "linear2": nn.Linear(15,5),
            "relu2": nn.ReLU(),
            "linear3": nn.Linear(5, 1),
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

        # 第一层隐藏层
        self.w12 = nn.Parameter(torch.randn(3, 1))
        self.w22 = nn.Parameter(torch.randn(3, 1))

        # 第二层隐藏层
        self.w13 = nn.Parameter(torch.randn(2, 1))

    # 正向传播
    def forward(self, x):
        x1 = torch.split(x, [4, 4], dim=1)

        x11 = torch.relu(x1[0] @ self.w11)
        x21 = torch.relu(x1[1] @ self.w21)

        x12 = torch.relu(x11 @ self.w12)
        x22 = torch.relu(x21 @ self.w22)

        x2 = torch.cat((x12, x22), axis=1)

        y = torch.sigmoid(x2 @ self.w13)
        return y

class BSANN_Ankle(nn.Module):
    def __init__(self):
        super(BSANN_Ankle, self).__init__()
        # 输入层
        self.w11 = nn.Parameter(torch.randn(2, 3))
        self.w21 = nn.Parameter(torch.randn(2, 3))
        self.w31 = nn.Parameter(torch.randn(3, 3))
        self.w41 = nn.Parameter(torch.randn(3, 3))
        self.w51 = nn.Parameter(torch.randn(2, 3))

        # 第一层隐藏层
        self.w12 = nn.Parameter(torch.randn(3, 1))
        self.w22 = nn.Parameter(torch.randn(3, 1))
        self.w32 = nn.Parameter(torch.randn(3, 1))
        self.w42 = nn.Parameter(torch.randn(3, 1))
        self.w52 = nn.Parameter(torch.randn(3, 1))


        # 第二层隐藏层
        self.w13 = nn.Parameter(torch.randn(5, 1))

    # 正向传播
    def forward(self, x):
        x1 = torch.split(x, [2, 2, 3, 3, 2], dim=1)

        x11 = torch.relu(x1[0] @ self.w11)
        x21 = torch.relu(x1[1] @ self.w21)
        x31 = torch.relu(x1[2] @ self.w31)
        x41 = torch.relu(x1[3] @ self.w41)
        x51 = torch.relu(x1[4] @ self.w51)

        x12 = torch.relu(x11 @ self.w12)
        x22 = torch.relu(x21 @ self.w22)
        x32 = torch.relu(x31 @ self.w32)
        x42 = torch.relu(x41 @ self.w42)
        x52 = torch.relu(x51 @ self.w52)

        x2 = torch.cat((x12, x22, x32, x42, x52), axis=1)

        y = torch.sigmoid(x2 @ self.w13)
        return y


class BSANN_Knee(nn.Module):
    def __init__(self):
        super(BSANN_Knee, self).__init__()
        # 输入层
        self.w11 = nn.Parameter(torch.randn(4, 3))
        self.w21 = nn.Parameter(torch.randn(2, 3))
        self.w31 = nn.Parameter(torch.randn(4, 3))
        self.w41 = nn.Parameter(torch.randn(3, 3))
        self.w51 = nn.Parameter(torch.randn(3, 3))

        # 第一层隐藏层
        self.w12 = nn.Parameter(torch.randn(3, 1))
        self.w22 = nn.Parameter(torch.randn(3, 1))
        self.w32 = nn.Parameter(torch.randn(3, 1))
        self.w42 = nn.Parameter(torch.randn(3, 1))
        self.w52 = nn.Parameter(torch.randn(3, 1))


        # 第二层隐藏层
        self.w13 = nn.Parameter(torch.randn(5, 1))

    # 正向传播
    def forward(self, x):
        x1 = torch.split(x, [4, 2, 4, 3, 3], dim=1)

        x11 = torch.relu(x1[0] @ self.w11)
        x21 = torch.relu(x1[1] @ self.w21)
        x31 = torch.relu(x1[2] @ self.w31)
        x41 = torch.relu(x1[3] @ self.w41)
        x51 = torch.relu(x1[4] @ self.w51)

        x12 = torch.relu(x11 @ self.w12)
        x22 = torch.relu(x21 @ self.w22)
        x32 = torch.relu(x31 @ self.w32)
        x42 = torch.relu(x41 @ self.w42)
        x52 = torch.relu(x51 @ self.w52)

        x2 = torch.cat((x12, x22, x32, x42, x52), axis=1)

        y = torch.sigmoid(x2 @ self.w13)
        return y


# 损失函数(二元交叉熵)
# def loss_func(self,y_pred,y_true):
def loss_func(y_pred, y_true):
    #将预测值限制在1e-7以上, 1- (1e-7)以下，避免log(0)错误
    eps = 1e-7
    y_pred = torch.clamp(y_pred,eps,1.0-eps)
    bce = - y_true*torch.log(y_pred) - (1-y_true)*torch.log(1-y_pred)
    print("@@@")
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

    # print("loss = ",loss_list)
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
Data_HipFE_1, idData_HipFE_1,\
Data_HipAA_1, idData_HipAA_1,\
Data_AnklePDF_1, idData_AnklePDF_1,\
Data_KneeFE_1, idData_KneeFE_1 = MD.getData()

idData_HipFE_1 = MinMaxScaler().fit_transform(idData_HipFE_1)
idData_HipAA_1 = MinMaxScaler().fit_transform(idData_HipAA_1)
idData_AnklePDF_1 = MinMaxScaler().fit_transform(idData_AnklePDF_1)
idData_KneeFE_1 = MinMaxScaler().fit_transform(idData_KneeFE_1)

x_name = [Data_HipFE_1,
          Data_HipAA_1,
          Data_AnklePDF_1,
          Data_KneeFE_1]
y_name = [idData_HipFE_1,
          idData_HipAA_1,
          idData_AnklePDF_1,
          idData_KneeFE_1]


name_i = 0
for name_i in range(4):
    if name_i == 2:
        t1 = 80
        t2 = 80
        t3 = []
        t4 = []

        # 切割数据,这里的数据是numpy
        line = x_name[name_i].shape[0]
        split = math.floor(line * 0.75 / 101) * 101

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
        mlpreg_Ankle = ANN_Ankle()
        mlpreg_BSANN_Ankle = BSANN_Ankle()

        # 处理
        optimizer_ANN_Ankle = torch.optim.Adam(params=mlpreg_Ankle.parameters(), lr=0.01)
        optimizer_BSANN_Ankle = torch.optim.Adam(params=mlpreg_BSANN_Ankle.parameters(), lr=0.01)

        # 均方误差损失函数
        loss_func = nn.MSELoss()
        train_loss_all = []
        test_loss_all = []
        train_loss_all_BSANN = []
        test_loss_all_BSANN = []

        for epoch in range(500):
            # ANN训练集
            train_loss = 0
            train_num = 0
            test_loss = 0
            test_num = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                output = mlpreg_Ankle(b_x)
                loss = loss_func(output, b_y)  # loss是均方误差，可认为是一个样本的loss
                optimizer_ANN_Ankle.zero_grad()
                loss.backward()
                optimizer_ANN_Ankle.step()
                train_loss += loss.item() * b_x.size(0)  # b_x.size(0) = batch_size
                train_num += b_x.size(0)
            train_loss_all.append(train_loss / train_num)  # 一个batch_size的loss加和 / batch_size

            # ANN测试集
            output_test = mlpreg_Ankle(X_test)
            # print(output_test.shape)
            loss_test = loss_func(output_test, Y_test)

            # 精度
            lp, p, vaf = evaluate(mlpreg_Ankle, X_train, Y_train)
            lp2, p2, vaf2 = evaluate(mlpreg_Ankle, X_test, Y_test)

            Ankle_ANN_test = mlpreg_Ankle.forward(X_test)
            # print(Ankle_ANN_test.shape)
            Ankle_ANN_test = Ankle_ANN_test.detach().numpy()
            # print(Ankle_ANN_test.shape)
            # loss
            # loss_ANN_test = loss_test.detach().numpy()
            # print(loss_test)
            # print(loss_test.shape)
            # print(loss_ANN_test.shape)

            if vaf2 > t1:
                t1 = vaf2
                t3 = Ankle_ANN_test

            # BSANN训练集
            train_loss_BSANN = 0
            train_num_BSANN = 0
            test_loss_BSANN = 0
            test_num_BSANN = 0


            for step, (b_x, b_y) in enumerate(train_loader):
                output_BSANN = mlpreg_BSANN_Ankle(b_x)
                loss_BSANN = loss_func(output_BSANN, b_y)
                optimizer_BSANN_Ankle.zero_grad()
                loss_BSANN.backward()
                optimizer_BSANN_Ankle.step()
                train_loss_BSANN += loss_BSANN.item() * b_x.size(0)
                train_num_BSANN += b_x.size(0)
            train_loss_all_BSANN.append(train_loss_BSANN / train_num_BSANN)


            output_test_BSANN = mlpreg_BSANN_Ankle(X_test)
            loss_test_BSANN = loss_func(output_test_BSANN, Y_test)


            # 精度
            lp_BSANN, p_BSANN, vaf_BSANN = evaluate_BSANN(mlpreg_BSANN_Ankle, X_train, Y_train)
            lp2_BSANN, p2_BSANN, vaf2_BSANN = evaluate_BSANN(mlpreg_BSANN_Ankle, X_test, Y_test)

            Ankle_BSANN_test = mlpreg_BSANN_Ankle.forward(X_test)
            Ankle_BSANN_test = Ankle_BSANN_test.detach().numpy()
            # loss
            # loss_BSANN_test = loss_test_BSANN.detach().numpy()

            if vaf2_BSANN > t2:
                t2 = vaf2_BSANN
                t4 = Ankle_BSANN_test

        # 图
        if name_i == 2:
            y_test = y_test[0:101]
            Ankle_ANN_test = Ankle_ANN_test[0:101]
            Ankle_BSANN_test = Ankle_BSANN_test[0:101]

            print("Ankle_ANN_test =", vaf2)
            print("Ankle_BSANN_test =", vaf2_BSANN)

            plt.plot(y_test, "+-", label="Inverse Dynamics")
            plt.plot(Ankle_ANN_test, "x-", label="Ankle_ANN_Prediction")
            plt.plot(Ankle_BSANN_test, "r*-", label="Ankle_BSANN_Prediction")
            # plt.plot(loss_ANN_test, "x*-", label="loss_ANN_test")
            # plt.plot(loss_test_BS, "r*-", label="loss_test_BS")

            plt.legend(loc=4)
            plt.grid()
            plt.title("UPS_R", fontsize=15)
            plt.show()



    # if name_i == 3:
    #     t1 = 80
    #     t2 = 80
    #     t3 = []
    #     t4 = []
    #
    #     # 切割数据,这里的数据是numpy
    #     line = x_name[name_i].shape[0]
    #     split = math.floor(line * 0.75 / 101) * 101
    #
    #     x_train = x_name[name_i][:split, :]
    #     y_train = y_name[name_i][:split, :]
    #     x_test = x_name[name_i][split:, :]
    #     y_test = y_name[name_i][split:, :]
    #
    #     X_train = torch.from_numpy(x_train.astype(np.float32))
    #     Y_train = torch.from_numpy(y_train.astype(np.float32))
    #     X_test = torch.from_numpy(x_test.astype(np.float32))
    #     Y_test = torch.from_numpy(y_test.astype(np.float32))
    #
    #     # 将数据处理为数据加载器
    #     train_data = Data.TensorDataset(X_train, Y_train)
    #     test_data = Data.TensorDataset(X_test, Y_test)
    #     train_loader = Data.DataLoader(dataset=train_data, batch_size=3, shuffle=True)
    #
    #     # 调用网络
    #     mlpreg_Knee = ANN_Knee()
    #
    #     mlpreg_BSANN_Knee = BSANN_Knee()
    #
    #     # 处理
    #     optimizer_ANN_Knee = torch.optim.Adam(params=mlpreg_Knee.parameters(), lr=0.01)
    #     optimizer_BSANN_Knee = torch.optim.Adam(params=mlpreg_BSANN_Knee.parameters(), lr=0.01)
    #
    #     # 均方误差损失函数
    #     loss_func = nn.MSELoss()
    #     train_loss_all = []
    #     test_loss_all = []
    #     train_loss_all_BSANN = []
    #     test_loss_all_BSANN = []
    #
    #     for epoch in range(100):
    #         # ANN训练集
    #         train_loss = 0
    #         train_num = 0
    #         test_loss = 0
    #         test_num = 0
    #         for step, (b_x, b_y) in enumerate(train_loader):
    #             output = mlpreg_Knee(b_x)
    #             loss = loss_func(output, b_y)  # loss是均方误差，可认为是一个样本的loss
    #             optimizer_ANN_Knee.zero_grad()
    #             loss.backward()
    #             optimizer_ANN_Knee.step()
    #             train_loss += loss.item() * b_x.size(0)  # b_x.size(0) = batch_size
    #             train_num += b_x.size(0)
    #         train_loss_all.append(train_loss / train_num)  # 一个batch_size的loss加和 / batch_size
    #
    #         # ANN测试集
    #         output_test = mlpreg_Knee(X_test)
    #         loss_test = loss_func(output_test, Y_test)
    #
    #         # 精度
    #         lp, p, vaf = evaluate(mlpreg_Knee, X_train, Y_train)
    #         lp2, p2, vaf2 = evaluate(mlpreg_Knee, X_test, Y_test)
    #
    #         Knee_ANN_test = mlpreg_Knee.forward(X_test)
    #         Knee_ANN_test = Knee_ANN_test.detach().numpy()
    #
    #         if vaf2 > t1:
    #             t1 = vaf2
    #             t3 = Knee_ANN_test
    #
    #         # BSANN训练集
    #         train_loss_BSANN = 0
    #         train_num_BSANN = 0
    #         test_loss_BSANN = 0
    #         test_num_BSANN = 0
    #
    #         for step, (b_x, b_y) in enumerate(train_loader):
    #             output_BSANN = mlpreg_BSANN_Knee(b_x)
    #             loss_BSANN = loss_func(output_BSANN, b_y)
    #             optimizer_BSANN_Knee.zero_grad()
    #             loss_BSANN.backward()
    #             optimizer_BSANN_Knee.step()
    #             train_loss_BSANN += loss_BSANN.item() * b_x.size(0)
    #             train_num_BSANN += b_x.size(0)
    #         train_loss_all_BSANN.append(train_loss_BSANN / train_num_BSANN)
    #
    #         output_test_BSANN = mlpreg_BSANN_Knee(X_test)
    #         loss_test_BSANN = loss_func(output_test_BSANN, Y_test)
    #
    #         # 精度
    #         lp_BSANN, p_BSANN, vaf_BSANN = evaluate_BSANN(mlpreg_BSANN_Knee, X_train, Y_train)
    #         lp2_BSANN, p2_BSANN, vaf2_BSANN = evaluate_BSANN(mlpreg_BSANN_Knee, X_test, Y_test)
    #
    #         Knee_BSANN_test = mlpreg_BSANN_Knee.forward(X_test)
    #         Knee_BSANN_test = Knee_BSANN_test.detach().numpy()
    #
    #         if vaf2_BSANN > t2:
    #             t2 = vaf2_BSANN
    #             t4 = Knee_BSANN_test
    #
    #
    #     if name_i == 3:
    #         y_test = y_test[0:101]
    #         Knee_ANN_test = Knee_ANN_test[0:101]
    #         Knee_BSANN_test = Knee_BSANN_test[0:101]
    #
    #         print("Knee_ANN_test =", vaf2)
    #         print("Knee_BSANN_test =", vaf2_BSANN)
    #
    #         plt.plot(y_test, "+-", label="Inverse Dynamics")
    #         plt.plot(Knee_ANN_test, "x-", label="Knee_ANN_Prediction")
    #         plt.plot(Knee_BSANN_test, "r*-", label="Knee_BSANN_Prediction")
    #         #         plt.plot(Ankle_ANN_test, "x-", label="Ankle_ANN_Prediction")
    #         #         plt.plot(Ankle_BSANN_test, "r*-", label="Ankle_BSANN_Prediction")
    #
    #         plt.legend(loc=4)
    #         plt.grid()
    #         plt.title("UPS_L", fontsize=15)
    #         plt.show()




    # else:
    #     t1 = 80
    #     t2 = 80
    #     t3 = []
    #     t4 = []
    #     # start = datetime.datetime.now()
    #
    #     for i in range(2):
    #         # 切割数据,这里的数据是numpy
    #         line = x_name[name_i].shape[0]
    #         split = math.floor(line * 0.75 / 101) * 101
    #
    #         # print(x_name[name_i][:split, :].shape)
    #
    #         x_train = x_name[name_i][:split, :]
    #         y_train = y_name[name_i][:split, :]
    #         x_test = x_name[name_i][split:, :]
    #         y_test = y_name[name_i][split:, :]
    #
    #         X_train = torch.from_numpy(x_train.astype(np.float32))
    #         Y_train = torch.from_numpy(y_train.astype(np.float32))
    #         X_test = torch.from_numpy(x_test.astype(np.float32))
    #         Y_test = torch.from_numpy(y_test.astype(np.float32))
    #
    #         # 将数据处理为数据加载器
    #         train_data = Data.TensorDataset(X_train, Y_train)
    #         test_data = Data.TensorDataset(X_test, Y_test)
    #         train_loader = Data.DataLoader(dataset=train_data, batch_size=3, shuffle=True)
    #
    #         # 调用网络
    #         mlpreg_Hip = Net()
    #         mlpreg_Ankle = ANN_Ankle()
    #         # mlpreg = ANN_4()
    #         # mlpreg = ANN_3()
    #
    #         mlpreg_BSANN_Hip = BSANN_Hip()
    #         # mlpreg_BSANN_Ankle = BSANN_Ankle()
    #
    #         # 处理
    #         optimizer = torch.optim.Adam(params=mlpreg_Hip.parameters(), lr=0.01)
    #         # optimizer_ANN_Ankle = torch.optim.Adam(params=mlpreg_Ankle.parameters(), lr=0.01)
    #
    #         optimizer_BSANN_Hip = torch.optim.Adam(params=mlpreg_BSANN_Hip.parameters(), lr=0.01)
    #         # optimizer_BSANN_Ankle = torch.optim.Adam(params=mlpreg_BSANN_Ankle.parameters(), lr=0.01)
    #
    #         # 均方误差损失函数
    #         loss_func = nn.MSELoss()
    #         train_loss_all = []
    #         test_loss_all = []
    #         train_loss_all_BSANN = []
    #         test_loss_all_BSANN = []
    #
    #         for epoch in range(20):
    #             # ANN训练集
    #             train_loss = 0
    #             train_num = 0
    #             test_loss = 0
    #             test_num = 0
    #             for step, (b_x, b_y) in enumerate(train_loader):
    #                 output = mlpreg_Hip(b_x)
    #                 loss = loss_func(output, b_y)  # loss是均方误差，可认为是一个样本的loss
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
    #                 train_loss += loss.item() * b_x.size(0)  # b_x.size(0) = batch_size
    #                 train_num += b_x.size(0)
    #             train_loss_all.append(train_loss / train_num)  # 一个batch_size的loss加和 / batch_size
    #
    #             # ANN测试集
    #             # time1 = time.clock()
    #             output_test = mlpreg_Hip(X_test)
    #             loss_test = loss_func(output_test, Y_test)
    #             # time2 = time.clock()
    #             # time3 = time2 - time1
    #             # print("time= ",time3 * 300)
    #
    #             # 精度
    #             lp, p, vaf = evaluate(mlpreg_Hip, X_train, Y_train)
    #             lp2, p2, vaf2 = evaluate(mlpreg_Hip, X_test, Y_test)
    #
    #             Hip_ANN_test = mlpreg_Hip.forward(X_test)
    #             Hip_ANN_test = Hip_ANN_test.detach().numpy()
    #
    #             if vaf2 > t1:
    #                 t1 = vaf2
    #                 t3 = Hip_ANN_test
    #
    #             # BSANN训练集
    #             train_loss_BSANN = 0
    #             train_num_BSANN = 0
    #             test_loss_BSANN = 0
    #             test_num_BSANN = 0
    #
    #             # start_train = datetime.datetime.now()
    #
    #             for step, (b_x, b_y) in enumerate(train_loader):
    #                 output_BSANN = mlpreg_BSANN_Hip(b_x)
    #                 loss_BSANN = loss_func(output_BSANN, b_y)
    #                 mlpreg_BSANN_Hip.zero_grad()
    #                 loss_BSANN.backward()
    #                 optimizer_BSANN_Hip.step()
    #                 train_loss_BSANN += loss_BSANN.item() * b_x.size(0)
    #                 train_num_BSANN += b_x.size(0)
    #             train_loss_all_BSANN.append(train_loss_BSANN / train_num_BSANN)
    #
    #             # Time_train = datetime.datetime.now()
    #             # print("Time_train = ", (Time_train - start_train) * 300)
    #
    #             # BSANN测试集
    #             # start_test = datetime.datetime.now()
    #             # time1 = time.clock()
    #
    #             output_test_BSANN = mlpreg_BSANN_Hip(X_test)
    #             loss_test_BSANN = loss_func(output_test_BSANN, Y_test)
    #
    #             # time2 = time.clock()
    #             # time3 = time2 - time1
    #             # print("time= ",time3 * 300)
    #
    #             # Time_test = datetime.datetime.now()
    #             # print("Time_test = ", (Time_test - start_test) * 300)
    #
    #             # 精度
    #             lp_BSANN, p_BSANN, vaf_BSANN = evaluate_BSANN(mlpreg_BSANN_Hip, X_train, Y_train)
    #             lp2_BSANN, p2_BSANN, vaf2_BSANN = evaluate_BSANN(mlpreg_BSANN_Hip, X_test, Y_test)
    #
    #             Hip_BSANN_test = mlpreg_BSANN_Hip.forward(X_test)
    #             Hip_BSANN_test = Hip_BSANN_test.detach().numpy()
    #
    #             if vaf2_BSANN > t2:
    #                 t2 = vaf2_BSANN
    #                 t4 = Hip_BSANN_test
    #     # 图
    #     if name_i == 0:
    #         y_test = y_test[0:101]
    #         Hip_ANN_test = Hip_ANN_test[0:101]
    #         Hip_BSANN_test = Hip_BSANN_test[0:101]
    #
    #         # print("speed 1：")
    #         print("HipFE_ANN_test =", vaf2)
    #         print("HipFE_BSANN_test =", vaf2_BSANN)
    #
    #         plt.subplot(141)
    #         plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
    #         plt.plot(Hip_BSANN_test, "x-")
    #         plt.plot(Hip_ANN_test, "r*-")
    #
    #
    #
    #     if name_i == 1:
    #         y_test = y_test[0:101]
    #         Hip_ANN_test = Hip_ANN_test[0:101]
    #         Hip_BSANN_test = Hip_BSANN_test[0:101]
    #
    #         # print("speed 1：")
    #         print("HipAA_ANN_test =", vaf2)
    #         print("HipAA_BSANN_test =", vaf2_BSANN)
    #
    #         plt.subplot(142)
    #         plt.plot(y_test, "+-")  # , label="Inverse Dynamics")
    #         plt.plot(Hip_BSANN_test, "x-")
    #         plt.plot(Hip_ANN_test, "r*-")




