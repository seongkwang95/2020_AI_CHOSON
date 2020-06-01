import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#데이터 받아오기
mnist = tf.keras.datasets.mnist
mnist_data = mnist.load_data()

#데이터 모양 보기
print(np.shape(mnist_data))
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train))
print(x_train[0])

# 이미지 봐보기
plt.imshow(x_train[1])
plt.title(y_train[1])
plt.show()

# 레이어 설게
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #2차원 행렬을 1차원으로 변환
    tf.keras.layers.Dense(5, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='tanh')
])

out = model.predict([x_train[0:2]])

print(out)
print(np.shape(out))
#-----------------------
print(y_train[0], type(y_train[0]))
print(x_train[0], type(y_train[0]))
#softmax는 출력값의 확률중 가장 큰 것을 선택하게 해주는 언어

temp_y =[]
for one_y_cal in y_train: # 6만개의 y train 값중 하나를 가져온다
    zero_array =np.zeros(10) #zero array라는 공간이 [0,0,0,0,0,0,0,0,0,0,0,0]로 이루어져있다
    zero_array[one_y_val] =1 #0~9의 수중에 해당하는 수에 1료 표시
    temp_y.append(zero_array) #생성된 값은 array에 저장
temp_y = np.array(temp_y)
print(type(temp_y))

