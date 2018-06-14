import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 数据集参数
data_num = 60000 #The number of figures
test_data_num = 10000 #The number of figures
fig_w = 45       #width of each figure

# KNN模型参数
n_neighbors = 3

# 数据导入、处理
data = np.fromfile("mnist_train\\mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist_train\\mnist_train_label",dtype=np.uint8)
data = data.reshape(data_num, fig_w * fig_w)

test_data = np.fromfile("mnist_test\\mnist_test_data",dtype=np.uint8)
test_label = np.fromfile("mnist_test\\mnist_test_label",dtype=np.uint8)
test_data = test_data.reshape(test_data_num, fig_w * fig_w)

# 建立、训练模型
neigh = KNeighborsClassifier(n_neighbors = n_neighbors)
neigh.fit(data, label)

# 预测及分析
predict = neigh.predict(test_data)
print (classification_report(test_label, predict))

