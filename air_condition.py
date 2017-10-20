import os
import numpy as np
import pandas as pd
import glob
import re
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import normalize
os.getcwd() #dir
#preprocess
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
# to time series
b=pd.DataFrame.copy(a)
b['前三天']=np.zeros(len(a))
b['前兩天']=np.zeros(len(a))
b['前一天']=np.zeros(len(a))
b['今天']=np.zeros(len(a))
for i in range(3,len(a)):
    b.iloc[i,4]=b.iloc[i-3,3]
    b.iloc[i,5]=b.iloc[i-2,3]
    b.iloc[i,6]=b.iloc[i-1,3]
    b.iloc[i,7]=b.iloc[i,3]
    
b=b.drop('平均',axis=1)
b=b.drop(b.index[0:3])
b

#model lstm
end=np.around(b.values[:,3:].astype("float32"))
end=end.astype('int').copy()
#scaler = MinMaxScaler(feature_range=(0, 1))
#end = scaler.fit_transform(end)
end=normalize(end)
end
train_X, train_y = end[:3000, :-1], end[:3000,-1]
test_X, test_y = end[3000:, :-1], end[3000:,-1]
train_X,test_X
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = Sequential()
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


history=model.fit(train_X, train_y, epochs=50, batch_size=30,validation_data=(test_X, test_y),verbose=2, shuffle=False)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
