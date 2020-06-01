import pandas as pd


temp = pd.read_csv('Data_1.csv')
print(temp)

print(temp['Mon'])

temp['T'] = 0
print(temp)

import matplotlib.pyplot as plt
plt.plot(temp['A'], dashes=[6,2])
plt.xlim(0,5)
plt.ylim(0,10)
plt.show()
#make one hot
#print(max(y_train))

temp_y =[]
for one_y_cal in  y_train:
    zero_array =np.zeros(10)
    zero_array[one_y_val] =1
    temp_y.append(zero_array)
temp_y = np.array(temp_y)
print(type(temp_y))

