#워나?인코딩을 쓰는 이유
#1.값을 예측 (커브피팅하고 유사)
#2.분류 (softmax는 분류에 속한다)
#출력된데이터의 편차를 줄여서 최종적으로 원하는값을 100%로 출력되기 위한 인코딩을 말한다.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#
#LOCA_1 = pd.read_csv('DB/(12, 100010, 25).csv')
#plt.plot(LOCA_1['ZINST70'])
#plt.show()
import glob
import numpy as np

#
PARA = ['UHOLEG1','UHOLEG2','UHOLEG3','ZINST58']
train_x = []
train_y = []
for one_file in glob.glob('DB/*.csv'):
    LOCA = pd.read_csv(one_file)

    if len(train_x)==0:
        train_x = LOCA.loc[:, PARA].to_numpy()
        train_y = LOCA.loc[:, ['Normal_0']]. to_numpy()
    else:
        get_x =LOCA.loc[:,PARA].to_numpy()
        get_y =LOCA.loc[:,['Normal_0']].to_numpy()
        train_x = np.vstack((train_x,get_x))
        train_y = np.vstack((train_y,get_y))
    print(f'X_SHAPE : {np.shape(train_x)} | '
          f'Y_SHAPE : {np.shape(train_y)}')
print('DONE')

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(np.shape(train_x)[1]),
    tf.keras.layers.Dense(200),
    tf.keras.layers.Dense(np.shape(train_y)[1]+1,
                          activation='softmax'),
])
model.compile(optimizer='adam', #temsprflpw
              loss='sparse_categorical_crossentropy',#손실함수로 숫자값을 뿌려서 원하는 결과값과 매칭 softmax 필요
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs = 200)
out_trained = model.predict(train_x[0:60])
plt.plot(train_y[0:60])
plt.plot(out_trained)
plt.show()