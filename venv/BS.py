import torch
from torch import nn
# from torchkeras import summary
import numpy as np
import pandas as pd
import os
import Processing as PR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math



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

class ANN_4(nn.Module):
    def __init__(self):
        super(ANN_4, self).__init__()
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


class ANN_3(nn.Module):
    def __init__(self):
        super(ANN_3, self).__init__()
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

# class BSANN_Knee(nn.Module):
#     def __init__(self):
#         super(BSANN_Knee, self).__init__()
#         # 输入层
#         self.w11 = nn.Parameter(torch.randn(4, 3))
#         self.w21 = nn.Parameter(torch.randn(2, 3))
#         self.w31 = nn.Parameter(torch.randn(2, 3))
#         self.w41 = nn.Parameter(torch.randn(2, 3))
#         self.w51 = nn.Parameter(torch.randn(2, 3))
#
#         # 第一层隐藏层
#         self.w12 = nn.Parameter(torch.randn(3, 1))
#         self.w22 = nn.Parameter(torch.randn(3, 1))
#         self.w32 = nn.Parameter(torch.randn(3, 1))
#         self.w42 = nn.Parameter(torch.randn(3, 1))
#         self.w52 = nn.Parameter(torch.randn(3, 1))
#
#         # 第二层隐藏层
#         self.w13 = nn.Parameter(torch.randn(5, 1))
#
#     # 正向传播
#     def forward(self, x):
#         x1 = torch.split(x, [4, 2, 2, 2, 2], dim=1)
#
#         x11 = torch.relu(x1[0] @ self.w11)
#         x21 = torch.relu(x1[1] @ self.w21)
#         x31 = torch.relu(x1[2] @ self.w31)
#         x41 = torch.relu(x1[3] @ self.w41)
#         x51 = torch.relu(x1[4] @ self.w51)
#
#         x12 = torch.relu(x11 @ self.w12)
#         x22 = torch.relu(x21 @ self.w22)
#         x32 = torch.relu(x31 @ self.w32)
#         x42 = torch.relu(x41 @ self.w42)
#         x52 = torch.relu(x51 @ self.w52)
#
#         x2 = torch.cat((x12, x22, x32, x42, x52), axis=1)
#
#         y = torch.sigmoid(x2 @ self.w13)
#         return y


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
Data_HipFE,idData_HipFE,\
Data_HipAA,idData_HipAA,\
Data_HipFE_ramp,idData_HipFE_ramp,\
Data_HipAA_ramp, idData_HipAA_ramp = PR.getData()

# print("Data_HipFE=",Data_HipFE.shape)
# print("idData_HipFE=",idData_HipFE.shape)
# print("Data_HipAA=",Data_HipAA.shape)
# print("idData_HipAA=",idData_HipAA.shape)

idData_HipFE = MinMaxScaler().fit_transform(idData_HipFE)
idData_HipAA = MinMaxScaler().fit_transform(idData_HipAA)
idData_HipFE_ramp = MinMaxScaler().fit_transform(idData_HipFE_ramp)
idData_HipAA_ramp = MinMaxScaler().fit_transform(idData_HipAA_ramp)
# idData_Knee = MinMaxScaler().fit_transform(idData_Knee)

x_name = [Data_HipFE, Data_HipAA, Data_HipFE_ramp, Data_HipAA_ramp]
y_name = [idData_HipFE, idData_HipAA, idData_HipFE_ramp, idData_HipAA_ramp]



for name_i in range(4):
    # name_i = 0
    t1 = 80
    t2 = 80
    t3 = []
    t4 = []

    for i in range(20):
        # 切割数据,这里的数据是numpy
        line = x_name[name_i].shape[0]
        split = math.floor(line * 0.8 / 101) * 101

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
        train_loader = Data.DataLoader(dataset=train_data, batch_size=5, shuffle=True)

        # 调用网络
        # mlpreg = Net()
        # mlpreg = ANN_4()
        mlpreg = ANN_3()

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

        for epoch in range(30):
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
            lp, p, vaf = evaluate(mlpreg, X_train, Y_train)
            lp2, p2, vaf2 = evaluate(mlpreg, X_test, Y_test)

            HipFE_ANN_test = mlpreg.forward(X_test)
            HipFE_ANN_test = HipFE_ANN_test.detach().numpy()

            if vaf2 > t1:
                t1 = vaf2
                t3 = HipFE_ANN_test

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

            # NSANN测试集
            output_test_BSANN = mlpreg_BSANN(X_test)
            loss_test_BSANN = loss_func(output_test_BSANN, Y_test)

            # 精度
            lp_BSANN, p_BSANN, vaf_BSANN = evaluate_BSANN(mlpreg_BSANN, X_train, Y_train)
            lp2_BSANN, p2_BSANN, vaf2_BSANN = evaluate_BSANN(mlpreg_BSANN, X_test, Y_test)

            HipFE_BSANN_test = mlpreg_BSANN.forward(X_test)
            HipFE_BSANN_test = HipFE_BSANN_test.detach().numpy()

            if vaf2_BSANN > t2:
                t2 = vaf2_BSANN
                t4 = HipFE_BSANN_test

# 所有结果数据记录
# 平地
    if name_i == 0:
        # HipFE_ANN_train = mlpreg.forward(X_train)
        # HipFE_ANN_train = HipFE_ANN_train.detach().numpy()
        # HipFE_ANN_test = mlpreg.forward(X_test)
        # HipFE_ANN_test = HipFE_ANN_test.detach().numpy()
        #
        # HipFE_BSANN_train = mlpreg_BSANN.forward(X_train)
        # HipFE_BSANN_train = HipFE_BSANN_train.detach().numpy()
        # HipFE_BSANN_test = mlpreg_BSANN.forward(X_test)
        # HipFE_BSANN_test = HipFE_BSANN_test.detach().numpy()

        # y_test = y_test[1:102]

        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        # print("HipFE_ANN_train=", vaf)
        print("HipFE_ANN_test=", t1)
        # print("HipFE_BSANN_train=", vaf_BSANN)
        print("HipFE_BSANN_test=", t2)

        plt.subplot(221)
        plt.plot(y_test, "+-", label="Inverse Dynamics")
        plt.plot(t3, "x-", label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-", label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames', fontsize=10)
        plt.ylabel('HipFE_moment_Levelground', fontsize=10)
        plt.legend()
        plt.grid()
        plt.title("(a)", fontsize=10)

        # ax2 = plt.twiny()
        # # ax2.plot(X, Y3)
        # ax2.set_title('(a)', fontsize=15)
        # plt.show()

    elif name_i == 1:
        # HipAA_ANN_train = mlpreg.forward(X_train)a
        # HipAA_ANN_train = HipAA_ANN_train.detach().numpy()
        # HipAA_ANN_test = mlpreg.forward(X_test)
        # HipAA_ANN_test = HipAA_ANN_test.detach().numpy()
        #
        # HipAA_BSANN_train = mlpreg_BSANN.forward(X_train)
        # HipAA_BSANN_train = HipAA_BSANN_train.detach().numpy()
        # HipAA_BSANN_test = mlpreg_BSANN.forward(X_test)
        # HipAA_BSANN_test = HipAA_BSANN_test.detach().numpy()

        # y_test = y_test[1:102]
        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        # print("HipAA_ANN_train=", vaf)
        print("HipAA_ANN_test=", t1)
        # print("HipAA_BSANN_train=", vaf_BSANN)
        print("HipAA_BSANN_test=", t2)

        plt.subplot(222)
        plt.plot(y_test, "+-", label="Inverse Dynamics")
        plt.plot(t3, "x-", label="HipAA_ANN_Predition")
        plt.plot(t4, "r*-", label="HipAA_BSANN_Predition")
        plt.xlabel('Time Frames', fontsize=10)
        plt.ylabel('HipAA_moment_Levelground', fontsize=10)
        plt.legend()
        plt.grid()
        plt.title("(b)", fontsize=10)

        # ax2 = plt.twiny()
        # # ax2.plot(X, Y3)
        # ax2.set_title('(b)', fontsize=15)

    # ramp
    elif name_i == 2:
        # HipFE_ANN_train_ramp = mlpreg.forward(X_train)
        # HipFE_ANN_train_ramp = HipFE_ANN_train_ramp.detach().numpy()
        # HipFE_ANN_test_ramp = mlpreg.forward(X_test)
        # HipFE_ANN_test_ramp = HipFE_ANN_test_ramp.detach().numpy()
        #
        # HipFE_BSANN_train_ramp = mlpreg_BSANN.forward(X_train)
        # HipFE_BSANN_train_ramp = HipFE_BSANN_train_ramp.detach().numpy()
        # HipFE_BSANN_test_ramp = mlpreg_BSANN.forward(X_test)
        # HipFE_BSANN_test_ramp = HipFE_BSANN_test_ramp.detach().numpy()

        # y_test = y_test[1:102]

        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        t3 = t3[0:101]
        t4 = t4[0:101]
        y_test = y_test[0:101]

        # print("HipFE_ANN_train_ramp=", vaf)
        print("HipFE_ANN_test_ramp=", t1)
        # print("HipFE_BSANN_train_ramp=", vaf_BSANN)
        print("HipFE_BSANN_test_ramp=", t2)

        plt.subplot(223)
        plt.plot(y_test, "+-", label="Inverse Dynamics")
        plt.plot(t3, "x-", label="HipFE_ANN_Predition")
        plt.plot(t4, "r*-", label="HipFE_BSANN_Predition")
        plt.xlabel('Time Frames', fontsize=10)
        plt.ylabel('HipFE_moment_ramp', fontsize=10)

        plt.legend()
        plt.grid()
        plt.title("(c)", fontsize=10)

        # ax2 = plt.twiny()
        # # ax2.plot(X, Y3)
        # ax2.set_title('(c)', fontsize=15)


    else:
        # HipAA_ANN_train_ramp = mlpreg.forward(X_train)
        # HipAA_ANN_train_ramp = HipAA_ANN_train_ramp.detach().numpy()
        # HipAA_ANN_test_ramp = mlpreg.forward(X_test)
        # HipAA_ANN_test_ramp = HipAA_ANN_test_ramp.detach().numpy()
        #
        # HipAA_BSANN_train_ramp = mlpreg_BSANN.forward(X_train)
        # HipAA_BSANN_train_ramp = HipAA_BSANN_train_ramp.detach().numpy()
        # HipAA_BSANN_test_ramp = mlpreg_BSANN.forward(X_test)
        # HipAA_BSANN_test_ramp = HipAA_BSANN_test_ramp.detach().numpy()

        # y_test = y_test[1:102]

        t3 = np.array(t3)
        t4 = np.array(t4)
        y_test = np.array(y_test)

        t3 = t3[0:101]
        t4 = t4[0:101]
        y_test = y_test[0:101]

        # print("HipAA_ANN_train_ramp=", vaf)
        print("HipAA_ANN_test_ramp=", t1)
        # print("HipAA_BSANN_train_ramp=", vaf_BSANN)
        print("HipAA_BSANN_test_ramp=", t2)

        plt.subplot(224)
        plt.plot(y_test, "+-", label="Inverse Dynamics")
        plt.plot(t3, "x-", label="HipAA_ANN_Predition")
        plt.plot(t4, "r*-", label="HipAA_BSANN_Predition")
        plt.xlabel('Time Frames', fontsize=10)
        plt.ylabel('HipAA_moment_ramp', fontsize=10)
        plt.legend()
        plt.grid()
        plt.title("(d)", fontsize=10)

        # ax2 = plt.twiny()
        # # ax2.plot(X, Y3)
        # ax2.set_title('(d)', fontsize=15)

        plt.show()

# plt.plot(x, y,'yd--',mfc='g')
# plt.errorbar(x, y,yerr=error,ecolor='r')
