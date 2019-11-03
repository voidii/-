# Starter code for CS 165B HW2 Spring 2019
import numpy as np


def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """

    dimen = training_input[0][0]
    number_A = training_input[0][1]
    number_B = training_input[0][2]
    number_C = training_input[0][3]

    centroid_A = []  # x,y,z为中心的坐标
    centroid_A_x = 0.0

    for x in range(0, dimen):
        for y in range(1, number_A + 1):
            centroid_A_x = centroid_A_x + training_input[y][x]
        centroid_A_x = centroid_A_x / number_A
        centroid_A.append(centroid_A_x)
        centroid_A_x = 0

    centroid_B = []
    centroid_B_x = 0.0

    for z in range(0, dimen):  # x是维度
        for m in range(number_A + 1, number_A + number_B + 1):  # y是数量
            centroid_B_x = centroid_B_x + training_input[m][z]
            m = m + 1
        centroid_B_x = centroid_B_x / number_B
        centroid_B.append(centroid_B_x)
        centroid_B_x = 0

    centroid_C = []
    centroid_C_x = 0.0

    for n in range(0, dimen):  # x是维度
        for l in range(number_A + number_B + 1, number_A + number_B + number_C + 1):  # y是数量
            centroid_C_x = centroid_C_x + training_input[l][n]
            l = l + 1
        centroid_C_x = centroid_C_x / number_C
        centroid_C.append(centroid_C_x)
        centroid_C_x = 0

    print(centroid_A)
    print(centroid_B)
    print(centroid_C)

    # 得到法向量
    A_to_B = [centroid_B[i] - centroid_A[i] for i in range(len(centroid_A))]
    C_to_A = [centroid_A[i] - centroid_C[i] for i in range(len(centroid_C))]
    B_to_C = [centroid_C[i] - centroid_B[i] for i in range(len(centroid_B))]
    # 得到中点坐标
    AB_mid = [centroid_A[i] + (A_to_B[i] / 2) for i in range(len(centroid_B))]
    CA_mid = [centroid_C[i] + (C_to_A[i] / 2) for i in range(len(centroid_A))]
    BC_mid = [centroid_B[i] + (B_to_C[i] / 2) for i in range(len(centroid_C))]
    # 计算平面方程中的常数项
    AB_eps = 0 - sum(np.multiply(np.array(A_to_B), np.array(AB_mid)))
    CA_eps = 0 - sum(np.multiply(np.array(C_to_A), np.array(CA_mid)))
    BC_eps = 0 - sum(np.multiply(np.array(B_to_C), np.array(BC_mid)))
    # 参数设置完毕

    # 导入测试
    test_number_A = testing_input[0][1]
    test_number_B = testing_input[0][2]
    test_number_C = testing_input[0][3]

    # A类的计算
    A_TP = 0
    A_FN = 0
    A_FP = 0
    A_TN = 0
    B_TP = 0
    B_FN = 0
    B_FP = 0
    B_TN = 0
    C_TP = 0
    C_FN = 0
    C_FP = 0
    C_TN = 0
    for y in range(1, test_number_A + 1):  # y是数量
        result_AorB = sum(np.multiply(np.array(testing_input[y]), np.array(A_to_B))) + AB_eps
        # 数据坐标和法向量相乘并减去常数项，如果点在平面上那么该result应该等于0
        result_CorA = sum(np.multiply(np.array(testing_input[y]), np.array(C_to_A))) + CA_eps
        result_BorC = sum(np.multiply(np.array(testing_input[y]), np.array(B_to_C))) + BC_eps
        if (result_AorB < 0) & (result_CorA > 0):  # 归到了A类
            A_TP = A_TP + 1
            B_TN = B_TN + 1
            C_TN = C_TN + 1
        elif (result_AorB > 0) & (result_BorC < 0):  # 归到了B类
            A_FN = A_FN + 1
            B_FP = B_FP + 1
            C_TN = C_TN + 1
        elif (result_BorC > 0) & (result_CorA < 0):  # 归到了C类
            A_FN = A_FN + 1
            C_FP = C_FP + 1
            B_TN = B_TN + 1

    # B类的计算

    for y in range(test_number_A + 1, test_number_B + test_number_A + 1):  # y是数量
        result_AorB = sum(np.multiply(np.array(testing_input[y]), np.array(A_to_B))) + AB_eps
        # 数据坐标和法向量相乘并减去常数项，如果点在平面上那么该result应该等于0
        result_CorA = sum(np.multiply(np.array(testing_input[y]), np.array(C_to_A))) + CA_eps
        result_BorC = sum(np.multiply(np.array(testing_input[y]), np.array(B_to_C))) + BC_eps
        if (result_AorB < 0) & (result_CorA > 0):  # 归到了A类
            A_FP = A_FP + 1
            B_FN = B_FN + 1
            C_TN = C_TN + 1
        elif (result_AorB > 0) & (result_BorC < 0):  # 归到了B类
            B_TP = B_TP + 1
            A_TN = A_TN + 1
            C_TN = C_TN + 1
        elif (result_BorC > 0) & (result_CorA < 0):  # 归到了C类
            B_FN = B_FN + 1
            C_FP = C_FP + 1
            A_TN = A_TN + 1

    # C类的计算

    for y in range(test_number_B + test_number_A + 1, test_number_B + test_number_A + test_number_C + 1):  # y是数量
        result_AorB = sum(np.multiply(np.array(testing_input[y]), np.array(A_to_B))) + AB_eps
        # 数据坐标和法向量相乘并减去常数项，如果点在平面上那么该result应该等于0
        result_CorA = sum(np.multiply(np.array(testing_input[y]), np.array(C_to_A))) + CA_eps
        result_BorC = sum(np.multiply(np.array(testing_input[y]), np.array(B_to_C))) + BC_eps
        if (result_AorB < 0) & (result_CorA > 0):  # 归到了A类
            A_FP = A_FP + 1
            C_FN = C_FN + 1
            B_TN = B_TN + 1
        elif (result_AorB > 0) & (result_BorC < 0):  # 归到了B类
            B_FP = B_FP + 1
            C_FN = C_FN + 1
            A_TN = A_TN + 1
        elif (result_BorC > 0) & (result_CorA < 0):  # 归到了C类
            C_TP = C_TP + 1
            B_TN = B_TN + 1
            A_TN = A_TN + 1

    TPR = (A_TP + B_TP + C_TP) / (test_number_B + test_number_A + test_number_C)
    FPR = (A_FP + B_FP + C_FP) / ((test_number_B + test_number_A + test_number_C)*2)
    err = (A_FP + B_FP + C_FP + A_FN + B_FN + C_FN) / ((test_number_C + test_number_B + test_number_A) * 3)
    acc = (A_TP + B_TP + C_TP + A_TN + B_TN + C_TN) / ((test_number_C + test_number_B + test_number_A) * 3)
    pre = TPR
    print('error_rate:', err, 'accuracy:', acc, 'tpr:', TPR, 'precision:', pre, 'fpr:', FPR)

    # 测试数据导入完成

    # TODO: IMPLEMENT
    pass


#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data


if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw2.py [training file path] [testing file path]
    """
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])

    run_train_test(training_input, testing_input)
