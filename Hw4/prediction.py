import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow as tf

train_dir = "./hw4_train"
model_path = "./model/"
test_dir = "./hw4_test/hw4_test"

def read_data(test_dir):
    tests = []
    for fname in os.listdir(test_dir):
        if (fname == '.DS_Store'): continue
        fpath = os.path.join(test_dir, fname)  # 返回文件绝对路径

        image = Image.open(fpath)
        test = np.array(image) / 255  # 化成numpy矩阵
        tests.append(test)

    len_test = len(tests)
    tests = np.array(tests)

    return tests, len_test
num_classes = 10

tests, len_test = read_data(test_dir)

datas_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
dropout_placeholder = tf.placeholder(tf.float32)

# 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool1)

# 全连接层，转换为长度为100的特征向量
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# 加上DropOut，防止过拟合
dropout_fc = tf.layers.dropout(fc, dropout_placeholder)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.arg_max(logits, 1)

saver = tf.train.Saver()

tests = tests.reshape(len_test, 28, 28, 1)

with tf.Session() as sess:

    fo = open("prediction.txt", "w")
        # 如果是测试，载入参数
    saver.restore(sess, model_path)
    print("从{}载入模型".format(model_path))

    predicted_labels_val = sess.run(predicted_labels, feed_dict={datas_placeholder: tests, dropout_placeholder: 0})
    for predicted_label in predicted_labels_val:
        fo.write("{}\n".format(predicted_label))

    fo.close()