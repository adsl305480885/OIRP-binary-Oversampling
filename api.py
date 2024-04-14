'''
Author: ZhouHao
Email: 2294776770@qq.com
Date: 2023-09-13 22:11:53
LastEditors: Zhou Hao
LastEditTime: 2024-01-19 16:55:39
FilePath: /SMOTE_optimise/api.py
Description: api functions to be called
'''


import numpy as np
import pandas as pd 
from sklearn.datasets import make_blobs,make_circles,make_moons,make_classification
from sklearn.cluster import KMeans
from scipy.stats import poisson
from random import choice
import random
import matplotlib.pyplot as plt


def binary_data(data_name:str, random_state:int=0) ->np.array:
    """Lable is in first col"""
    np.random.seed(random_state)
    if data_name == 'make_moons':
        x, y = make_moons(n_samples=1200, noise=0.35)
        data = np.hstack((y.reshape((len(y), 1)), x))
        np.random.shuffle(data)
        
        data_0 = data[data[:, 0] == 0]
        data_1 = data[data[:, 0] == 1]
        data_0 = data_0[:100]
        data = np.vstack((data_0, data_1))
        
    elif data_name == 'make_circles':
        x, y = make_circles(n_samples=1200, noise=0.3, factor=0.6)
        data = np.hstack((y.reshape((len(y), 1)), x))
        np.random.shuffle(data)

        data_0 = data[data[:, 0] == 0]
        data_1 = data[data[:, 0] == 1]
        data_0 = data_0[:100]
        data = np.vstack((data_0, data_1))

    data = pd.DataFrame(data)
    y = data[0]
    X = data[[1, 2]]

    return X.values, y.values


def mult_calss_data(data_name,noise_rate,n_samples=1500,n_clusters=3):
    if data_name == 'moons':
        X1, Y1 = make_moons(n_samples=300,random_state=0,noise=noise_rate)
        X2, Y2 = make_moons(n_samples=525, random_state=0, noise=noise_rate)
        X3, Y3 = make_moons(n_samples=675, random_state=0, noise=noise_rate)
        X = np.vstack((X1, X2, X3))
        y1 = np.array([0] * 300)
        y2 = np.array([1] * 525)
        y3 = np.array([2] * 675)        #生成标签数组
        Y = np.hstack((y1, y2, y3))

        # 用kmeans聚成三类，将原始数据集划分成三类。
        y_pred = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(X)

        x = pd.DataFrame(X)
        y_pred = pd.DataFrame(y_pred)
        moons = pd.concat([y_pred,x],axis=1)    #按列拼接datafram/series
        l0,l1,l2 = [],[],[]
        moons.columns = ['0','1','2']       #修改df的列名字
        for index,row in moons.iterrows():              #采样
            if row[0] ==0:
                if len(l0) < 200:
                    l0.append(row)
                    continue
            elif row[0] ==1:
                if len(l1) < 350:
                    l1.append(row)
                    continue
            elif row[0] ==2:
                if len(l2) < 450:
                    l2.append(row)
                    continue

        data = l0+l1+l2
        data = pd.DataFrame(data)       # list to dataframe
        data = data.values         # dataframe to ndarra

        moons = data           #dataframe to ndarrar
        X = moons[:,1:]
        Y = moons[:,0]    
        return X,Y
    elif data_name == 'blobs':
        pass
    elif data_name == 'circles':
        pass
    elif data_name == 'toy':
        data = pd.read_csv(r'datasets/toy.csv')
        X = data.iloc[:, 1:]
        y = data.iloc[:,0]
    
        return X.values,y.values


def add_flip_noise(
                    # X:np.ndarray,y:np.array,
                   dataset:np.ndarray, noise_rate:float)->(np.ndarray,np.ndarray):
    '''
    description: 添加翻转标签噪声
    label is in the fist col
    '''
    
    label_cat = list(set(dataset[:, 0]))
    new_data = np.array([])
    flag = 0
    for i in range(len(label_cat)):
        label = label_cat[i]
        other_label = list(filter(lambda x: x != label, label_cat))
        data = dataset[dataset[:, 0] == label]
        n = data.shape[0]
        noise_num = int(n * noise_rate)
        noise_index_list = []  # 记录所有噪声的下标
        n_index = 0
        while True:
            # 每次选择下标
            rand_index = int(random.uniform(0, n))
            # 如果下标已有，执行下一次while
            if rand_index in noise_index_list:
                continue
            # 满足两个条件翻转: 正类且噪声噪声不够
            if n_index < noise_num:
                data[rand_index, 0] = choice(other_label)  # todo
                n_index += 1
                noise_index_list.append(rand_index)
            # 跳出
            if n_index >= noise_num:
                break
        if flag == 0:
            new_data = data
            flag = 1
        else:
            new_data = np.vstack([new_data, data])

    return new_data



def add_noise(X:np.ndarray, y:np.ndarray ,
                    label_or_feature:str,
                    noise_type:str, noise_rate:float,)->np.ndarray:
        
    if label_or_feature == 'label':
        if noise_type == 'gaussian':
            mean = 0    # 均值
            stddev = 0.1    # 标准差
            noise = np.random.normal(mean, stddev, size=y.shape) # 生成高斯噪声
            y_noise = y + noise  # 将高斯噪声添加到标签
        elif noise_type == 'poisson':
            lambda_ = 0.1  # 泊松噪声的参数lambda
            noise = np.random.poisson(lambda_, size=y.shape) # 生成泊松噪声
            y_noise = y + noise # 将泊松噪声添加到标签
        elif noise_type == 'salt_pepper':
            # 椒盐噪声的比例
            salt_pepper_ratio = 0.05

            # 生成椒盐噪声
            num_samples = len(y)
            num_salt = int(num_samples * salt_pepper_ratio / 2)
            num_pepper = int(num_samples * salt_pepper_ratio / 2)

            salt_indices = np.random.choice(num_samples, num_salt, replace=False)
            pepper_indices = np.random.choice(num_samples, num_pepper, replace=False)

            y_noise = np.copy(y)
            y_noise[salt_indices] = np.max(y)
            y_noise[pepper_indices] = np.min(y)
        return X,y_noise
        

    elif label_or_feature == 'feature':
        if noise_rate == 0:
            return X,y
        
        if noise_type == 'gaussian':
            mean = 0  # 噪声均值
            std = 0.1  # 噪声标准差
            noise = np.random.normal(mean, std, size=X.shape)  # 生成与X形状相同的高斯噪声
            X_noise = X + noise  # 将高斯噪声添加到数据集中
        elif noise_type == 'poisson':
            lam = 0.1  # 泊松噪声的参数（均值和方差）
            rng = np.random.default_rng()  # 创建随机数生成器
            noise = rng.poisson(lam, size=X.shape)  # 生成与X形状相同的泊松噪声
            X_noise = X + noise  # 将泊松噪声添加到数据集中
        elif noise_type == 'salt_pepper':
            salt_pepper_ratio = 0.05  # 椒盐噪声的比例
            amount = int(salt_pepper_ratio * X.size)  # 噪声点的数量
            coords = [np.random.randint(0, i - 1, amount) for i in X.shape]  # 随机选择噪声点的坐标
            X_noisy = np.copy(X)
            X_noisy[coords] = np.random.choice([np.min(X), np.max(X)], size=amount)  # 将噪声点设置为最小值或最大值
        else:
            raise TypeError("noise_type is not error")
        return X_noise,y