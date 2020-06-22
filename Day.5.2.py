import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler

SGTR_1=pd.read_csv('DB/13_10004_60.csv')

#라벨링할 공간확보
SGTR_1['La'] = 0
print(SGTR_1['La'])

SGTR_1['La'].iloc[0:12] = 1 #사고가 일어난 시점이 12니깐 0~11까지는 1. 12부터는 0

plt.plot(SGTR_1['ZINST70'])
plt.plot(SGTR_1['La'])
plt.show()

train_x= SGTR_1.loc[:, ['ZINST70', 'QPRZP']]
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
train_y = SGTR_1['La'].to_numpy()
import numpy as np
#print(np.shape(train_x),type(train_x))
#print(np.shape(train_y),type(train_y))

#plt.plot(train_x)
#plt.plot(train_y)
#plt.show()

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='softmax')
])
out = model.predict(train_x[:])

#학습

model.compile(optimizer='adam', #temsprflpw
              loss='sparse_categorical_crossentropy',#손실함수로 숫자값을 뿌려서 원하는 결과값과 매칭
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200)
out_trained = model.predict(train_x[:])

#plt.plot((train_x))
#plt.plot(train_y)
plt.plot(out)
#plt.plot(out_trained)
plt.show()