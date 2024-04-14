'''
Author: Zhou Hao
Date: 2023-12-08 20:42:55
LastEditors: Zhou Hao
LastEditTime: 2024-04-14 16:23:58
Description: 用于画论文OOF的图
E-mail: zhouhaocqupt@163.com
'''


# import Python library
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_swiss_roll,make_blobs,make_classification
import random
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC


# import Diy library
from OIRP_Bin_Over import PSO_Denoise, opt_n


def Gaussian_3d( is_save:bool,is_show:bool)->None:
    fig = plt.figure(figsize=(10, 8))
    font = {'family': 'Times New Roman',
			'size': 18, }
    plt.rc('font', **font)


    # 定义均值和协方差矩阵
    mean = np.array([0, 0])
    covariance = np.array([[3, 0], [0, 3]])

    # 创建网格矩阵
    x, y = np.meshgrid(np.linspace(-4, 4, 1000), np.linspace(-4, 4, 1000))
    pos = np.dstack((x, y))

    # 计算二维正态分布的概率密度值
    z = multivariate_normal.pdf(pos, mean=mean, cov=covariance)

    # 绘制三维概率密度图像
    ax = fig.add_subplot(111, projection='3d')
    cmap_diy:str = 'cividis' # matplotlib.colormap

    surface = ax.plot_surface(x, y, z,cmap=cmap_diy,alpha=0.4)  # 3D表面图
    ax.plot_wireframe(x, y, z, color='black', alpha=0.1)   # 3D框线图
    ax.contourf(x,y,z,zdir='z',offset=0.07,cmap=cmap_diy,alpha=0.6)  # z轴平面投影
    ax.contourf(x, y, z, cmap=cmap_diy,alpha=0.6) # Z轴立体投影

    # 定义直线的起点和终点坐标，绘制直线
    x_start, y_start, z_start = 0, 0, 0
    x_end, y_end, z_end = 0, 0, 0.08
    ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end],color='black',linewidth=1,alpha=1)

    # 画点
    ax.scatter(0,0,0,color='black') # 画原点
    ax.scatter(0,0,0.07,color='black')  # 画顶点
    ax.text(0,0.2,0,'$\mu(0,0)$')


    # 设置坐标轴 和title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('2D isotropic distribution Gaussian Probability Density')
    ax.set_zlim(0,0.07)
    ax.view_init(elev=30, azim=45)  # 调整可视化角度


    # 绘制投影
    # ax.contourf(x,y,z,zdir='z',offset=-0.04,cmap=cmap_diy,alpha=0.8)  # 绘制z轴投影
    # ax.contourf(x,y,z,zdir = 'x',offset=-3,cmap = cmap_diy,alpha=0.8) # 绘制x轴投影
    # ax.contourf(x,y,z,zdir = 'y',offset= 3,cmap = cmap_diy,alpha=0.8)  # 绘制y轴投影

    fig.colorbar(surface,shrink=0.7)    

    # 在顶部画数据集
    X, y ,centers= make_blobs(n_samples=120, 
                    centers=[[0,0]],    # default == 3  3_ class
                    n_features=2,
                    random_state=5,
                    shuffle=True,
                    cluster_std=1,
                    center_box=[-4,4],
                    return_centers=True,
                    )
    ax.scatter(X[:,0],X[:,1],0.07,c='Black',s=15,alpha=1)


    plt.tight_layout()
    # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        plt.savefig(fname=now+'gs.pdf',format='pdf',bbox_inches='tight',dpi=800)    # 相对路径
    if is_show:
        plt.show()


def ablustion(is_save:bool=False,is_show:bool=False,noise_rate:float=0)->None:
    '''
    description: 消融可视化实验
    return {*}
    '''    
    
    # get dataset ***********************************************************
    # centers = [[-0.01079671,-0.54383379, 0.2205215],
    #             [-0.20734018,-0.2453698 , 0.99314846],
    #             [-0.1836056 , 0.54378799, 0.52107338]]
    X, y ,centers= make_blobs(n_samples=2000, 
                    #   centers=centers,    # default == 3  3_ class
                    n_features=3,
                    random_state=50,
                    shuffle=True,
                    cluster_std=0.35,
                    center_box=[-1,1],
                    return_centers=True,
                    centers=2,
                    )
    print('原始比例：\t', Counter(y))
    # print(centers)

    # make dataset imblance
    index_0 = np.random.choice(np.where(y==0)[0],100)  # 150
    index_1 = np.random.choice(np.where(y==1)[0],500)   # 300   
    # index_2 = np.random.choice(np.where(y==2)[0],500)   # 500  三分类
    index = np.hstack((index_0,index_1))
    X,y = X[index],y[index]
    X_length = len(X)
    print('不平衡比例：\t',Counter(y))

    # make dataset noise
    noise_index = random.sample(list(range(len(y))),int(len(y)*noise_rate))
    for i in noise_index:
        # binary  加标签噪声
        y[i] = 1 if y[i]==0 else 0
        
        # 三分类加噪
        # if y[i] == 0: y[i] = random.choice([1,2])
        # elif y[i] == 1: y[i] = random.choice([0,2])
        # elif y[i] == 2: y[i] = random.choice([0,1])
    print('加噪后比例：\t',Counter(y))
    counter = Counter(y)
    n_maj, n_min = max(counter.values()), min(counter.values())


    # settings of figure ***********************************************************
    num = 140   # adjust the subfigs: row,col,index
    color = {0: 'darkcyan', 1: 'Burlywood', 2: 'green',-1:'blue'}
    X_colors = [color[label] for label in y]
    cmap = {0:'winter',1:'autumn',2:'summer'}
    font = {'family':'Times New Roman','size':12,}
    X_max, y_max = max(X[:,0]),max(X[:,1])
    X_min, y_min = min(X[:,0]),min(X[:,1])
    Z_min, Z_max = min(X[:,2]),max(X[:,2])
    X_min, X_max = X_min-0.01, X_max+0.01
    y_min, y_max = y_min-0.01, y_max+0.01
    Z_min, Z_max = Z_min-0.01, Z_max+0.01

    fig_1 = plt.figure(figsize=(20,6))
    fig_2 = plt.figure(figsize=(8,8))
    ax_2 = fig_2.add_subplot(111,projection='3d')   
    ax_2.scatter(X[:,0], X[:,1],X[:,2],c=X_colors,s=15,alpha=0.4)#origin 
    ax_2.set_xlabel('X',font)
    ax_2.set_ylabel('Y',font)
    ax_2.set_zlabel('Z',font)
    ax_2.set_title('Hyperplane',font,y=-0.005)    # 标题设置在下方
    ax_2.view_init(elev=4,azim=23)       # 调整可视化角度


    def settings_ax(title:str)->None:
        """public settings of each sub ax"""
        nonlocal num    # 扩展上层函数中变量的作用域
        num += 1
        ax = fig_1.add_subplot(num,projection='3d')   

        ax.set_xlabel('X',font)
        ax.set_ylabel('Y',font)
        ax.set_zlabel('Z',font)
        # ax.set_xticks([])   # 隐藏X轴刻度
        # ax.set_xlim(min(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
        # ax.set_ylim(min(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
        # ax.set_zlim(min(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
        ax.set_title(title,font,y=-0.1)    # 标题设置在下方
        ax.view_init(elev=4,azim=23)       # 调整可视化角度

        # X和X_new不用传进来，因为嵌套函数享用上层函数的变量作用域
        ax.scatter(X[:,0], X[:,1],X[:,2],c=X_colors,s=15,alpha=0.5)   # origin samples
        
        # 分类器超平面
        ax.plot_surface(xx,yy,Z1,alpha=0.8,
                        # color='Snow',
                        cmap='summer',
                        )
        # new samples
        if X_res is None:return
        X_new = X_res[X_length:]
        ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2],s=15,c='red',alpha=0.9,marker='+') 


    # origin data ***********************************************************
    X_res = None
    
    clf = SVC(random_state=42, probability=True,kernel='linear')  
    clf.fit(X,y)
    b=clf.intercept_    # 超平面常数
    w=clf.coef_         # 超平面权重系数
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  #计算超平面
    settings_ax(title='(a) Original dataset')
    legend_original=ax_2.plot_surface(xx,yy,Z1,alpha=0.7, color='#63b2ee',)


    # SMOTE
    model = SMOTE(random_state=42)
    X_res, y_res = model.fit_resample(X, y)
    print('SMOTE采样后:\t',Counter(y_res))
    
    clf = SVC(random_state=42, probability=True,kernel='linear')  
    clf.fit(X_res,y_res)
    b=clf.intercept_    
    w=clf.coef_         
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  
    settings_ax(title='(b) SMOTE')
    legend_smote=ax_2.plot_surface(xx,yy,Z1,alpha=0.7,color='#76da91')


    # SMOTE n*
    sampling_strategy = opt_n(X,y)
    print('\t\tsampling_strategy:\t',sampling_strategy)
    model = SMOTE(random_state=42,sampling_strategy=sampling_strategy)
    X_res, y_res = model.fit_resample(X, y)
    print('SMOTE_n*:\t',Counter(y_res))

    clf = SVC(random_state=42, probability=True,kernel='linear')  
    clf.fit(X_res,y_res)
    b=clf.intercept_    # 超平面常数
    w=clf.coef_         # 超平面权重系数
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  #计算超平面
    settings_ax(title='(c) SMOTE $\;\; n^{*}$')
    legend_smote_n=ax_2.plot_surface(xx,yy,Z1,alpha=0.7,color='#F3D266',)


    # SMOTE_opt_PSO_denoise    
    X_new = X_res[X_length:]
    y_new = y_res[X_length:]
    X_res,y_res = PSO_Denoise(X.copy(),y.copy(),X_new.copy(),y_new.copy())
    print('SMOTE_n*_denoise:\t',Counter(y_res))

    clf = SVC(random_state=42, probability=True,kernel='linear')  
    clf.fit(X_res,y_res)
    b=clf.intercept_    # 超平面常数
    w=clf.coef_         # 超平面权重系数
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  #计算超平面
    settings_ax(title='(d) SMOTE $\;\; n^{*} \; denoise$')
    legend_smote_oof=ax_2.plot_surface(xx,yy,Z1,alpha=0.7,color='#f89588',)


    # 调整legend参数，不然会报错
    legend_original._facecolors2d=legend_original._facecolors3d
    legend_original._edgecolors2d=legend_original._edgecolors3d
    legend_smote._facecolors2d=legend_smote._facecolors3d
    legend_smote._edgecolors2d=legend_smote._edgecolors3d
    legend_smote_n._facecolors2d=legend_smote_n._facecolors3d
    legend_smote_n._edgecolors2d=legend_smote_n._edgecolors3d
    legend_smote_oof._facecolors2d=legend_smote_oof._facecolors3d
    legend_smote_oof._edgecolors2d=legend_smote_oof._edgecolors3d
    ax_2.legend(
        (legend_original,legend_smote,legend_smote_n,legend_smote_oof),
        [
            'original',
            'SMOTE',
            'SMOTE $ \; n^{*}$',
            'SMOTE $\;\; n^{*} \; denoise$',
        ],
        loc='upper center',
        ncol=4,
    )


    # save and show ***********************************************************
    fig_1.tight_layout()
    fig_1.subplots_adjust(wspace =0, hspace =0)#调整子图间距
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        fig_1.savefig(fname=now+'Ablation'+'.pdf',format='pdf',bbox_inches='tight',dpi=800)
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        fig_2.savefig(fname=now+'Hyperplane'+'.pdf',format='pdf',bbox_inches='tight',dpi=800)
    if is_show:plt.show()


if __name__ == "__main__":
    is_save,is_show = 1,1   # if show=True, plt will not save ahead!
    noise_rate = 0.1

    #========================   test Gaussian_3d() ===================================
    Gaussian_3d( is_save=is_save,is_show=is_show)

    #========================   test ablustion() ===================================
    for i in range(10):
        ablustion(is_save=is_save,is_show=is_show,noise_rate=noise_rate)
