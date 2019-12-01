import numpy as np
from PIL import Image
import os

train_dir = "./hw4_train"
model_path = "./model/"
test_dir = "./hw4_test/hw4_test"

def read_data(tarin_dir):
    datas = []
    labels = []
    fpaths = []
    for name in os.listdir(tarin_dir):
        if (name == '.DS_Store'): continue
        ffpath = os.path.join(tarin_dir, name)
        for fname in os.listdir(ffpath):
            fpath = os.path.join(ffpath, fname) #返回文件绝对路径

            fpaths.append(fpath)

            image = Image.open(fpath)
            data = np.array(image)/255 #化成numpy矩阵
            label = int(fname.split("_")[0])  #label在文件名字中
            datas.append(data)
            labels.append(label)
    datas = np.array(datas)
    labels = np.array(labels)

    return datas, labels
	
def read_data2(test_dir):
    tests = []
    for fname in os.listdir(test_dir):
        if (fname == '.DS_Store'): continue
        fpath = os.path.join(test_dir, fname)  # 返回文件绝对路径

        image = Image.open(fpath)
        test = np.array(image) / 255  # 化成numpy矩阵
        tests.append(test)

    len_test = len(tests)
    tests = np.array(tests)

    return tests
	
a, b = read_data(train_dir)
np.save('train_data.npy', a)
np.save('label_data.npy', b)
c = read_data2(test_dir)
np.save('test_data.npy', c)