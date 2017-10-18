import os
import numpy as np
import pandas as pd
import glob
import re
os.getcwd() #dir

path =r'C:\Users\huang\Desktop\train'
allFiles1 = glob.glob(path + "/9*.csv")
allFiles2= glob.glob(path + "/1*.csv")
allFiles=allFiles1+allFiles2 #sort by 民國 9~10
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None,header=None,encoding='big5')
    list_.append(df)

allcsv=pd.concat(list_)
allcsv.columns=['日期','測站','測項','01','02','03','04','05','06','07','08','09','10','11','12','13','14'
               ,'15','16','17','18','19','20','21','22','23','24']

#a.to_csv("C:\\Users\\huang\\Desktop\\a.csv",encoding='big5')
def air_choose(air):
    mean=[]
    data=allcsv.loc[allcsv['測項']==str(air)]# find air
    for i in range(data.shape[0]):  #mean & clear data
        mean.append(data.iloc[i,3:27].str.replace('.+x{1}','nan').str.replace('.+#{1}','nan').str.replace('.+\*{1}','nan').astype(float).mean())
    data.iloc[:,3]=mean
    data=data.iloc[:,0:4]
    data.columns=['日期','測站','測項','平均']
    return(data)
    
a=air_choose("PM2.5")
a
