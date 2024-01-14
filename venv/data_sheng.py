import pandas as pd
import numpy as np
from scipy import interpolate ,signal
import os





# 读取csv文件目录路径
path = r"E:\\2\\SIAT_LLMD20230404\\SIAT_LLMD20230404\\Sub08\\Data_1"
# listdir()--返回path指定 的 文件夹中包含的文件或者文件夹名字的列表
File = os.listdir(path) # 是一个列表


# 每次改这里换文件
i = 1

data = os.path.join(path, File[i])
a = pd.read_csv(data, header=0)
a = a.iloc[:, :]
# a = a.iloc[0:1095, :]
# a = a.iloc[1095:2281, :]

time_1 = a.iloc[:, 0]
L_moment = a.iloc[:, 10]
# hip_flexion_r_moment = a.iloc[:, 4]
peaks = signal.find_peaks(-L_moment, height=(23))
# peaks = signal.find_peaks(-hip_flexion_r_moment, height=(11))

peaks=list(peaks)
# 改区间
# peaks[0] = peaks[0] + 1095


time_2 = time_1[peaks[0]]
# print("time_2 = ",time_2)
b = len(time_2)
print("b = ",b)


# ik_r
hip_flexion_r = a.iloc[:, 6]
hip_adduction_r = a.iloc[:, 5]
knee_angle_r = a.iloc[:, 7]
ankle_flexion_r = a.iloc[:, 8]
# ik_l
hip_flexion_l = a.iloc[:, 2]
hip_adduction_l = a.iloc[:, 1]
knee_angle_l = a.iloc[:, 3]
ankle_flexion_l = a.iloc[:, 4]



# EMG
Tensorfascialata = a.iloc[:, 17]
Rectusfemoris = a.iloc[:, 18]
Vastusmedialis = a.iloc[:, 19]
Semitendinosus = a.iloc[:, 20]
Uppertibialisanterior = a.iloc[:, 21]
Lowertibialisanterior = a.iloc[:, 22]
Lateralgastrocnemius= a.iloc[:, 23]
Medialgastrocnemius = a.iloc[:, 24]
Soleus = a.iloc[:, 25]



# id_r
hip_adduction_r_moment = a.iloc[:, 13]
hip_flexion_r_moment = a.iloc[:, 14]
ankle_angle_r_moment = a.iloc[:, 16]
knee_angle_r_moment = a.iloc[:, 15]
# id_l
hip_flexion_l_moment = a.iloc[:, 10]
hip_adduction_l_moment = a.iloc[:, 9]
ankle_angle_l_moment = a.iloc[:, 12]
knee_angle_l_moment = a.iloc[:, 11]






# 插值
time = np.linspace(min(time_2), max(time_2), 101 * (b - 1))



# ik
tck_A_1 = interpolate.splrep(time_1, hip_adduction_r)
hip_adduction_r = interpolate.splev(time, tck_A_1, der=0)

tck_A_2 = interpolate.splrep(time_1, knee_angle_r)
knee_angle_r = interpolate.splev(time, tck_A_2, der=0)

tck_A_3 = interpolate.splrep(time_1, ankle_flexion_r)
ankle_flexion_r = interpolate.splev(time, tck_A_3, der=0)

tck_A_4 = interpolate.splrep(time_1, hip_flexion_r)
hip_flexion_r = interpolate.splev(time, tck_A_4, der=0)

tck_A_5 = interpolate.splrep(time_1, hip_adduction_l)
hip_adduction_l = interpolate.splev(time, tck_A_5, der=0)

tck_A_6 = interpolate.splrep(time_1, knee_angle_l)  #
knee_angle_l = interpolate.splev(time, tck_A_6, der=0)  #

tck_A_7 = interpolate.splrep(time_1, ankle_flexion_l)
ankle_flexion_l = interpolate.splev(time, tck_A_7, der=0)

tck_A_8 = interpolate.splrep(time_1, hip_flexion_l)
hip_flexion_l = interpolate.splev(time, tck_A_8, der=0)


# emg
tck_E_1 = interpolate.splrep(time_1, Tensorfascialata)
Tensorfascialata = interpolate.splev(time, tck_E_1, der=0)

tck_E_2 = interpolate.splrep(time_1, Rectusfemoris)
Rectusfemoris = interpolate.splev(time, tck_E_2, der=0)

tck_E_3 = interpolate.splrep(time_1, Vastusmedialis)
Vastusmedialis = interpolate.splev(time, tck_E_3, der=0)

tck_E_4 = interpolate.splrep(time_1, Semitendinosus)
Semitendinosus = interpolate.splev(time, tck_E_4, der=0)

tck_E_5 = interpolate.splrep(time_1, Uppertibialisanterior)
Uppertibialisanterior = interpolate.splev(time, tck_E_5, der=0)

tck_E_6 = interpolate.splrep(time_1, Lowertibialisanterior)
Lowertibialisanterior = interpolate.splev(time, tck_E_6, der=0)

tck_E_7 = interpolate.splrep(time_1, Lateralgastrocnemius)
Lateralgastrocnemius = interpolate.splev(time, tck_E_7, der=0)

tck_E_8 = interpolate.splrep(time_1, Medialgastrocnemius)
Medialgastrocnemius = interpolate.splev(time, tck_E_8, der=0)\

tck_E_9 = interpolate.splrep(time_1, Soleus)
Soleus = interpolate.splev(time, tck_E_8, der=0)


# id
tck_1 = interpolate.splrep(time_1, hip_flexion_r_moment) #
hip_flexion_r_moment = interpolate.splev(time, tck_1, der=0) #

tck_2 = interpolate.splrep(time_1, hip_adduction_r_moment)
hip_adduction_r_moment = interpolate.splev(time, tck_2, der=0)

tck_3 = interpolate.splrep(time_1, knee_angle_r_moment)
knee_angle_r_moment = interpolate.splev(time, tck_3, der=0)

tck_4 = interpolate.splrep(time_1, ankle_angle_r_moment)
ankle_angle_r_moment = interpolate.splev(time, tck_4, der=0)

tck_5 = interpolate.splrep(time_1, hip_flexion_l_moment) #
hip_flexion_l_moment = interpolate.splev(time, tck_5, der=0) #

tck_6 = interpolate.splrep(time_1, hip_adduction_l_moment)
hip_adduction_l_moment = interpolate.splev(time, tck_6, der=0)

tck_7 = interpolate.splrep(time_1, knee_angle_l_moment)
knee_angle_l_moment = interpolate.splev(time, tck_7, der=0)

tck_8 = interpolate.splrep(time_1, ankle_angle_l_moment)
ankle_angle_l_moment = interpolate.splev(time, tck_8, der=0)




c = pd.DataFrame({'time': time,
                  # id_r
                  'hip_flexion_r_moment': hip_flexion_r_moment,
                  'hip_adduction_r_moment': hip_adduction_r_moment,
                  'knee_angle_r_moment': knee_angle_r_moment,
                  'ankle_angle_r_moment': ankle_angle_r_moment,
                  # id_l
                  'hip_flexion_l_moment': hip_flexion_l_moment,
                  'hip_adduction_l_moment': hip_adduction_l_moment,
                  'ankle_angle_l_moment': ankle_angle_l_moment,
                  'knee_angle_l_moment': knee_angle_l_moment,
                  # ik_r
                  'hip_flexion_r': hip_flexion_r, 'hip_adduction_r': hip_adduction_r,
                  'ankle_flexion_r': ankle_flexion_r, 'knee_angle_r': knee_angle_r,
                  # ik_l
                  'hip_flexion_l': hip_flexion_l, 'hip_adduction_l': hip_adduction_l,
                  'ankle_flexion_l': ankle_flexion_l, 'knee_angle_l': knee_angle_l,
                  # emg
                  'Tensorfascialata': Tensorfascialata,
                  'Rectusfemoris': Rectusfemoris,
                  'Vastusmedialis': Vastusmedialis,
                  'Semitendinosus': Semitendinosus,
                  'Uppertibialisanterior': Uppertibialisanterior,
                  'Lowertibialisanterior': Lowertibialisanterior,
                  'Lateralgastrocnemius': Lateralgastrocnemius,
                  'Medialgastrocnemius': Medialgastrocnemius,
                  'Soleus': Soleus})


# if (~np.isnan(hip_flexion_r_moment[0])):

sum111 = r"E:\\2\\data_sheng\\Sub08\\"
sum = sum111 + File[i]
c.to_csv(sum, index=False)
