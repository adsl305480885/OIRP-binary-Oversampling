'''
Author: Zhou Hao
Date: 2024-01-16 21:45:46
LastEditors: Zhou Hao
LastEditTime: 2024-04-14 16:27:49
Description: file content
E-mail: zhouhaocqupt@163.com
'''
# import system packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from imblearn import over_sampling
import random
from collections import Counter
import logging
import time
import warnings 
warnings.filterwarnings("ignore")   # ignore the warings
import matplotlib.pyplot as plt 


# import self-defined packages
from api import binary_data
from OIRP_Bin_Over import PSO_Denoise,opt_n


def main_2d(data:int, noise_rate:float=0.1, is_show:bool=0,is_save:bool=0)->None:  # ndarray

    # get dataset, X:np.ndarray, y:np.ndarray
    if data == 'make_moons':X, y = binary_data(data_name='make_moons')
    elif data =='make_circles':X,y = binary_data(data_name='make_circles')
    assert type(X) == np.ndarray
    assert type(y) == np.ndarray

    X_length = len(X)
    counter = Counter(y)
    n_maj, n_min = max(counter.values()), min(counter.values())

    # settings of fig
    num = 240   # adjust the subfigs: row,col,index
    plt.figure(figsize=(16,10),)    # dpi=800
    font = {'family': 'Times New Roman',
			'size': 14, }
    

    def draw_subfig(title:str)->None:
        """draw subfig, called by main"""

        nonlocal num    # 扩展上层函数中变量的作用域
        num += 1
        ax = plt.subplot(num)
        color = {0: 'darkcyan', 1: 'tan', 2: 'green',-1:'blue'}
        X_colors = [color[label] for label in y]

        # 设置每个坐标轴的标签
        ax.set_xlabel('X',font)     
        ax.set_ylabel('Y',font)

        # 画出样本点
        X_new = X_res[X_length:]
        ax.set_title(title,font)   
        ax.scatter(X[:,0], X[:,1],c=X_colors,s=15,alpha=0.5) # origin samples
        ax.scatter(X_new[:, 0], X_new[:, 1],s=15,c='red',alpha=0.8,marker='+') # new samples
        ax.grid()   # 设置网格线


    '''Original dataset'''
    X_res, y_res = X, y
    print("original rate:\t",Counter(y))
    draw_subfig(title='Original')


    '''SMOTE'''
    model = over_sampling.SMOTE(random_state=42)
    X_res, y_res = model.fit_resample(X, y)
    print('SMOTE Oversampling:\t',Counter(y_res))
    draw_subfig(title='SMOTE')


    '''SMOTE_opt'''
    sampling_strategy = opt_n(X,y)
    model = over_sampling.SMOTE(random_state=42,sampling_strategy=sampling_strategy)
    X_res, y_res = model.fit_resample(X, y)
    print('SMOTE n* Oversampling:\t',Counter(y_res))
    draw_subfig(title='(c) SMOTE_opt')


    '''SMOTE_opt_PSO_denoise'''    
    X_new = X_res[X_length:]
    y_new = y_res[X_length:]
    X_res,y_res = PSO_Denoise(X.copy(),y.copy(),X_new.copy(),y_new.copy())
    print('SMOTE_opt_PSO_denoise:\t',Counter(y_res))
    draw_subfig(title='(d) SMOTE_opt_PSO_denoise')


    # save and show ***********************************************************
    plt.tight_layout()
    if is_save:
        now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
        plt.savefig(fname=now+'.pdf',format='pdf',bbox_inches='tight')
    if is_show:plt.show()    


if __name__ == '__main__':
    data = {
            2:'make_moons',
            3:'make_circles',}
    main_2d(data=data[2],
            noise_rate=0,
            is_save=1,is_show=0,)