import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data_num = 60000 #The number of figures
test_data_num = 10000 #The number of figures
fig_w = 45       #width of each figure

"""
#读取并处理、储存数据
data = np.fromfile("mnist_train\\mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist_train\\mnist_train_label",dtype=np.uint8)
print(data.shape)
print(label.shape)

trainx = np.zeros(121500000)
for i in range(121500000):
    if i%1000000==0 :
        print(i)
    trainx[i] = data[i] * 1.0 / 255
trainy = np.zeros((60000,10))
for i in range(60000):
    trainy[i][label[i]] = 1

trainx.tofile("processed_train_data")
trainy.tofile("processed_train_label")

print ("done1")

test_data = np.fromfile("mnist_test\\mnist_test_data",dtype=np.uint8)
test_label = np.fromfile("mnist_test\\mnist_test_label",dtype=np.uint8)
testx = np.zeros(20250000)
for i in range(20250000):
    testx[i] = test_data[i] * 1.0 / 255
testy = np.zeros((10000,10))
for i in range(10000):
    testy[i][test_label[i]] = 1

testx.tofile("processed_test_data")
testy.tofile("processed_test_label")
"""

# 导入数据
trainx = np.fromfile("processed_train_data",dtype=np.float64)
trainy = np.fromfile("processed_train_label",dtype=np.float64)

testx = np.fromfile("processed_test_data",dtype=np.float64)
testy = np.fromfile("processed_test_label",dtype=np.float64)

#reshape the matrix
trainx = trainx.reshape(data_num, fig_w * fig_w)
testx = testx.reshape(test_data_num, fig_w * fig_w)
trainy = trainy.reshape(data_num, 10)
testy = testy.reshape(test_data_num, 10)

# 神经网络结构参数
INPUT_NODE = 2025  # 输入层节点数。等于MNIST图片的像素
LAYER_NODE = 4096  # 隐藏层节点数。只用一个隐藏层
OUTPUT_NODE = 10  # 输出层节点数。等于0~9对应的10个数字

# 优化方法参数
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.001  # 正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 训练参数
BATCH_SIZE = 100  # 一个训练batch中的图片数
TRAINING_STEPS = 3000  # 训练轮数


# 利用给定神经网络的输入和参数，返回前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 如果没有提供滑动平均类，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果
        layer = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 返回输出层的前向传播结果
        return tf.matmul(layer, weights2) + biases2
    else:
        # 计算变量的滑动平均值，再计算前向传播结果
        layer = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) +
            avg_class.average(biases1))
        return tf.matmul(
            layer, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程
def train():
    # 实现模型
    x = tf.placeholder(tf.float32, [None, INPUT_NODE])  # 输入层
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])  # 标签
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))  # 隐藏层权重
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))  # 隐藏层偏置
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))  # 输出层权重
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))  # 输出层偏置
    y = inference(x, None, weights1, biases1, weights2, biases2)  # 输出层

    # 存储训练轮数，设置为不可训练
    global_step = tf.Variable(0, trainable=False)

    # 设置滑动平均方法
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)  # 定义滑动平均类
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())  # 在所有可训练的变量上使用滑动平均
    average_y = inference(x, variable_averages, weights1, biases1, weights2,
                          biases2)  # 计算使用了滑动平均的前向传播结果

    # 设置正则化方法
    regularizer = tf.contrib.layers.l2_regularizer(
        REGULARIZATION_RATE)  # 定义L2正则化损失函数
    regularization = regularizer(weights1) + regularizer(
        weights2)  # 计算模型的正则化损失

    # 设置指数衰减法
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, data_num / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    # 最小化损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))  # 计算每张图片的交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 计算当前batch中所有图片的交叉熵平均值
    loss = cross_entropy_mean + regularization  # 总损失等于交叉熵损失和正则化损失的和
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)  # 优化损失函数

    # 同时反向传播和滑动平均
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 评估模型
    correct_prediction = tf.equal(tf.argmax(average_y, 1),
                                  tf.argmax(y_, 1))  # 检验使用滑动平均模型的前向传播的是否正确
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算正确率

    train_acc_curve = np.zeros((30,2)) #训练集accuracy曲线
    acc_curve = np.zeros((30,2)) #测试集accuracy曲线
    # 开始训练
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()

        # 测试数据
        train_feed = {x: trainx, y_: trainy}
        test_feed = {x: testx, y_: testy}

        # 迭代训练
        for i in range(TRAINING_STEPS):
            # 每100轮输出在训练/测试数据集上的正确率
            if i % 100 == 0:
                train_acc = sess.run(accuracy, feed_dict=train_feed)
                print('After %d training steps, train accuracy is %g ' %
                      (i, train_acc))
                train_acc_curve[int(i/100)][0] = i
                train_acc_curve[int(i/100)][1] = train_acc

                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print('After %d training steps, test accuracy is %g ' %
                      (i, test_acc))
                acc_curve[int(i/100)][0] = i
                acc_curve[int(i/100)][1] = test_acc

            # 产生新一轮batch
            xs = trainx[(i*128)%60000:((i+1)*128)%60000]
            ys = trainy[(i*128)%60000:((i+1)*128)%60000]
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束在测试集上计算正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training steps, test accuracy is %g ' %
              (TRAINING_STEPS, test_acc))
        # 画图
        plt.plot(train_acc_curve[:,0], train_acc_curve[:,1], label = '(train)MOVING_AVERAGE_DECAY = '+str(MOVING_AVERAGE_DECAY))
        plt.plot(acc_curve[:,0], acc_curve[:,1], label = 'MOVING_AVERAGE_DECAY = '+str(MOVING_AVERAGE_DECAY))

train()
MOVING_AVERAGE_DECAY = 0.99     # 调整参数
train()                         # 再次训练并画图
MOVING_AVERAGE_DECAY = 0.999
train()
plt.legend() # 显示图例
plt.xlabel('iteration times')
plt.ylabel('accuracy')
plt.show()
