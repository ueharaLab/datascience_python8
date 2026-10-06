import numpy as np

import math
from scipy.stats import multivariate_normal
from sklearn import datasets
from tqdm import tqdm
import scipy.stats as stats


class coin_trials:
    def __init__(self, flips):
        self.flips= flips
        self.freq=np.zeros(flips)
        
    def bern_trial(self, ):# 
        d = stats.bernoulli(0.5) #確率0.5で1が出るベルヌーイ分布を定義
        x = d.rvs(self.flips) #ベルヌーイ分布に従うサンプルを10個生成
        return np.sum(x)
        
    
    def fit(self, trials):
        c=1
        
        while c <=trials:
            
            pos_num = self.bern_trial()
            self.freq[pos_num]+=1
            c+=1
    
        #return self.freq
        
    
        
        