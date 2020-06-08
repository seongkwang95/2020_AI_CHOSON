import tensorflow as tf
import numpy as np
#데이터 받아오기
mnist = tf.keras.datasets.mnist

#데이터 로드하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 레이어 설게
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #2차원 행렬을 1차원으로 변환
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#
model.compile(optimizer='adam', #temsprf;pw
              loss='sparse_categorical_crossentropy',#손실함수로 숫자값을 뿌려서 원하는 결과값과 매칭??
              metrics=['accuracy'])

#
print(np.shape(x_train), np.shape(y_train))
print(type(x_train),type(y_train))
model.fit(x_train, y_train, epochs=5) #5번 반복해서 학습을 한다. 학습할수록 오차를 줄여간다.

#검증
print(model.predict(x_test[0:3]))
print(y_test[0:3])

#뉴럴네트워크
model.save_weights('save_model')
