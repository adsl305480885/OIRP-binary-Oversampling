<!--
 * @Author: Zhou Hao
 * @Date: 2024-04-07 18:04:04
 * @LastEditors: Zhou Hao
 * @LastEditTime: 2024-04-14 17:43:04
 * @Description: file content
 * @E-mail: 2294776770@qq.com
-->

# OIRP: An Oversampling Optimization Binary Framework for The Optimal Imbalance Ratio and Position



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
