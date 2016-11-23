from numpy import * #导入numpy库
import operator #导入操作符函数
'''
代码功能：KNN分类算法的HelloWord版Demo。主要是目的是练习KNN算法的代码实现过程，运行结果是测试sample=[0,0]属于group中的哪一类
代码作者：生命奇迹泉
代码时间：2016.11.02

'''
'''1.收集数据'''
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    print(group)
    return group,labels
'''2.实施KNN算法'''
def classify(inX,dataSet,labels,k):#四个参数，inX是输入向量即待预测向量，dataSet是训练样本集，labels是标签向量即所包含的类，它与dataSet的行数相同，k是临近的采样点个数
    dataSetSize=dataSet.shape[0] #得到数组的行数。即知道有几个训练数据
    '''2.1距离计算【欧式距离】'''
    diffMat=tile(inX,(dataSetSize,1))-dataSet #tile:numpy中的函数。他的功能是重复某个数组。比如tile(A,n)，功能是将数组A重复n次，构成一个新的数组。tile将原来的一个数组，扩充成了4个一样的数组。diffMat得到了目标与训练数值之间的差值。
    sqDiffMat=diffMat**2 #差值矩阵平方，各个元素分别平方
    sqDistances=sqDiffMat.sum(axis=1) #对应列相乘，即得到了每一个距离的平方；计算每一行上元素的和，求和，返回的是一维数组。axis=0, 表示列。axis=1, 表示行。
    distances=sqDistances**0.5 #开平方，得到距离即测试点到其余各个点的距离
    sortedDistIndicies=distances.argsort() #排序，argsort()返回值是原数组从小到大排序的下标值
    '''2.2选择距离最小的K个点'''
    classCount={}#定义一个空的字典
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]] #返回距离最近的k个点所对应的标签值
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #存放到字典中
        print(classCount)
    '''2.3排序[逆序从大到小]'''
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #排序 classCount.iteritems() 输出键值对 key代表排序的关键字 True代表降序
    return sortedClassCount[0][0] #返回距离最小的点对应的标签

def main():
    sample=[0,0]#待预测点
    k=3#选择最近邻的几个点
    group,labels=createDataSet()#调用createDataSet()方法，返回值为所有样本数值和对应下标
    label=classify(sample,group,labels,k)
    print("Classified Label:"+label)

if __name__=='__main__':
    main()
