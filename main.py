#!/usr/bin/python
#-*_coding:utf-8-*-

import kNN

from numpy import *
import operator
import matplotlib.pyplot as plt



def main1():
    '''
    生成数据
    '''
    group,labels = kNN.createDataSet()
    
def main2():
    '''
    测试分类器
    '''
    group,labels = kNN.createDataSet()    
    result = kNN.classify0([0,0], group, labels, 3)
    print result
    
def main3():
    '''
    将文本数据读入并绘图
    '''
    datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0*array(datingLabels), 15.0*array(datingLabels)) #根据类别绘制不同的散点
    plt.show()
    
def main4():
    datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
    # 获得归一化参数
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat) 
    
def main5():
    '''
    使用分类器对约会网站数据测试
    '''
    kNN.datingClassTest()
    
def main6():
    '''
    约会网站测试函数
    '''
    kNN.classifyPerson()
    
def main7():
    '''
    手写数字识别系统
    '''
    testVector = kNN.img2vector('testDigits/0_0.txt')
    kNN.handwritingClassTest()
    
if __name__ == "__main__":
    main7()
