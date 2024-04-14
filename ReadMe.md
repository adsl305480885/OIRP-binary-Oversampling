<!--
 * @Author: Zhou Hao
 * @Date: 2024-04-07 18:04:04
 * @LastEditors: Zhou Hao
 * @LastEditTime: 2024-04-14 17:43:04
 * @Description: file content
 * @E-mail: 2294776770@qq.com
-->

# OIRP: An Oversampling Optimization Binary Framework for The Optimal Imbalance Ratio and Position


* **Abstract**：Label imbalance and label noise are two critical challenges in the classification. The current solution for this problem is to modify the loss function and classifiers or data resampling. Most of these existing methods lack theoretical guidance and are limited by the dataset simutaneously. To solve these two problems from a deeper perspective and overcome the shortcomings of the existing algorithms, an optimization frame-work OIRP is proposed in this work. ORIP means the optimal Imbalance ratio and position. It is divided into two steps. Step 1): Solve for the optimal oversampling ratio based on the feature distribution and dimensionality of the dataset. Step 2): Optimize the location of the generated samples. The first step tackles label imbalance and the second step targets label noise. We optimally model the imbalance problem and give a convex analysis. OIRP is adaptive and can be used to improve the effectiveness of most oversampling algorithms. Theoretical proofs, visualization
experiments, and numerical experiments demonstrate that OIRP outperforms traditional optimization frameworks and improves most oversampling algorithms. Code and datasets are available at https://github.com/adsl305480885/OIRP-binary-Oversampling.
* **Keyword**: Imbalanced classification, lable noise, optimiza-
tion framework, convex optimization.

# Folders and Filers

* ***.pdf**: Visualisation of experimental results from ablation and other experiments.
* **draw_BO.py**: Some visualisation functions.
* **apis.py**: Some functions for synthesizing artificial datasets.
* **main_BO.py**: Code entry. Call the oversampling algorithms and visualize.
* **OIRP_Bin_Over.pyc**：Encrypted core code, binary files. Decrypted after the paper is received.
* **requirements.txt**: Environment required for code.

# Requirements

### Minimal installation requirements (>=Python 3.7):

* Anaconda 3.
* Linux operating system or Windows operating system.
* Sklearn, numpy, pandas, imbalanced_learn.

### Installation requirements (Python 3):

* conda create -n yourname python=3.7
* conda activate yourname
* pip install -r requirements.txt

# Usage

* pip install -r requirements.txt.
* python ./main_BO.py
* python ./draw_BO.py
* python ./OIRP_Bin_Over.pyc
  
##### Output:
* Optimal sampling ratio and new samples after denoising

# Doesn't work?
* Please contact Hao Zhou at zhouhaocqupt@163.com
