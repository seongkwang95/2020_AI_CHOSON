import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 레이어 설게
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #2차원 행렬을 1차원으로 변환
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#검증
out_net = model.predict(x_test[0:3])
for x_out, y_out in zip(out_net, y_test[0:3]):
    print(np.argmax(x_out), y_out)

#저장된 학습내용 가져오기
model.load_weights('save_model')

out_net = model.predict(x_test[0:3])
for x_out, y_out in zip(out_net, y_test[0:3]):
    print(np.argmax(x_out), y_out)
