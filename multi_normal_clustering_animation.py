import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from sklearn import datasets
from tqdm import tqdm
from matplotlib.animation import ArtistAnimation

# 軌跡を描くアニメーション プロットを10000というように大きな数にするとメモリーオーバーフローが起きて
# 白い画面しかでないので注意
# https://qiita.com/sci_Haru/items/278b6a50c4e9f4c07dcf
class Bayes_inference:
    def __init__(self, myu):
        self.myu1 = myu[0]
        self.myu2 = myu[1]
        self.met_samples_myu1=[]
        self.met_samples_myu2=[]
        
    def target_distribution(self, myu1,myu2):# こちらは、単にポワソンの密度関数にデータとパラメータを放り込んで、データ数分の積(.prod()) をとったもの。つまり尤度関数をpythonの密度関数メソッドでうまく（式を実装せず）作っている
        #posterior=1
        #for d in dataset:
            #print(d)
        #    posterior*=multivariate_normal.pdf(d, mean=np.array([myu1,myu2]), cov=np.eye(2))
            
        # 参考　https://pythonguides.com/python-scipy-stats-multivariate_normal/
        #return posterior
        
        return multivariate_normal.pdf(self.dataset, mean=np.array([myu1,myu2]), cov=np.eye(2)).prod()
    
    def fit(self, dataset):
        
        fig = plt.figure()
        plt.scatter(dataset[:,0],dataset[:,1])
        mean = np.mean(dataset,axis=0)
        plt.scatter(mean[0],mean[1],s=600, c="blue",alpha=0.5, marker="o")
        myu1=3
        myu2=2
        myu=[myu1,myu2]
        plt.scatter(myu1,myu2, s=600, c="red",alpha=0.5, marker="*")
                
        
        self.anim = []
        self.dataset = dataset
        met_samples_myu1=[]
        met_samples_myu2=[]
        self.myu_est=np.array([0.0,0.0])
        
        for i in tqdm(range(1000)):
            #print(i+1)
            delta = np.random.normal(loc=0.0, scale=0.1, size=2)
            
            myu1_next = self.myu1 + delta[0]
            myu2_next = self.myu2 + delta[1]
            
            p_next = self.target_distribution(myu1_next,myu2_next)# クラス内の別のメソッドを呼び出すときの構文self. で始まるが、引数にはselfはない
            p_now = self.target_distribution(self.myu1,self.myu2)

            r=p_next/p_now

            R =  np.random.rand()
            #R = 0.5
            if R<r:
                    self.myu1 = myu1_next
                    self.myu2 = myu2_next
            met_samples_myu1.append(self.myu1)
            met_samples_myu2.append(self.myu2)
            
            im=plt.plot(met_samples_myu1,met_samples_myu2, '--', color='red',alpha=0.3, markersize=10, linewidth = 2, aa=True)
            self.anim.append(im)
            
        

        anim = ArtistAnimation(fig, self.anim) # アニメーション作成
        plt.xlim(-2, 4)
        plt.ylim(1,7)
        plt.hlines([0], 0, 2000, linestyles="-")  # y=0に線を描く。

        plt.show() 
        fig.clear()
        plt.close(fig)   
        
                    
        
        
        self.met_samples_myu1 += met_samples_myu1[500:]
        self.met_samples_myu2 += met_samples_myu2[500:]
        self.myu_est[0] = np.mean(np.array(self.met_samples_myu1))
        #std1 = np.std(np.array(met_samples_myu1))
        self.myu_est[1] = np.mean(np.array(self.met_samples_myu2))
        #std2 = np.std(np.array(met_samples_myu2))
    

    def predict(self, data):
        
        self.pred = multivariate_normal.pdf(data, mean=np.array([self.myu_est[0],self.myu_est[1]]), cov=np.eye(2))
        return self.pred
        
        
        