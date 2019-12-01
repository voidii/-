# Starter code for CS 165B HW4

"""
Implement the testing procedure here. 

Inputs:
    Given the flder named "ohw4_test" that is put in the same directory of your "predictio.py" file, like:
    - Main folder
        - "prediction.py"
        - folder named "hw4_test" (the exactly same as the uncompressed hw4_test folder in Piazza)
    Your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        elsewise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instaed of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 socre for your hw4.
"""
import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow as tf

data = np.load('train_data.npy')
label = np.load('label_data.npy')  

model_path = "./model/"

num_classes = len(set(label))

# 定义Placeholder，存放输入和标签
datas_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
labels_placeholder = tf.placeholder(tf.int32, [None,])

# 存放DropOut参数的容器，训练时为0.25，测试时为0
dropout_placeholdr = tf.placeholder(tf.float32)

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
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.arg_max(logits, 1)

# 利用交叉熵定义损失
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)
# 平均损失
mean_loss = tf.reduce_mean(losses)

# 定义优化器，指定要优化的损失函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

# 用于保存和载入模型
saver = tf.train.Saver()

data = data.reshape(len(label), 28, 28, 1)
step = 0
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        print("训练模式")
        # 如果是训练，初始化参数
        sess.run(tf.global_variables_initializer())
        # 定义输入和Label以填充容器，训练时dropout为0.25
   
        while step < 1260:
            datas = data[step::1260]
            labels = label[step::1260]
            train_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0.25
        }
            _, mean_loss_val = sess.run([optimizer, mean_loss],
                                        feed_dict=train_feed_dict)
            step = step + 1
            if step % 200 == 0:
                print("step = {}/n".format(step))
                print("%.4f" % mean_loss_val)
        saver.save(sess, model_path)
        print("训练结束，保存模型到{}".format(model_path))













