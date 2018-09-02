# user-churn-model-python
#基于随机逻辑回归模型进行数据降维
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

inputfile1 = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/data/model_data_jiangwei_test.csv'
data = pd.read_csv(inputfile1)

x=data.iloc[:,:42].as_matrix()
y=data.iloc[:,42].as_matrix()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

rlr=RLR(selection_threshold=0.5)
rlr.fit(x,y)
rlr.get_support()
print(u'通过随机逻辑回归模型筛选特征结束。')
print(u'有效特征为：%s'%','.join(data.drop('IS_CJ',axis=1).columns[rlr.get_support()]))
x=data[data.drop('IS_CJ',axis=1).columns[rlr.get_support()]].as_matrix()
lr=LR()
lr.fit(x,y)
print(u'逻辑回归模型训练结束。')
print(u'模型的平均正确率为：%s'%lr.score(x,y))

#LM神经网络训练模型
from __future__ import print_function
import pandas as pd
import pandas as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

inputfile1 = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/data/model_data_train_2.xlsx'
inputfile2 = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/data/model_data_test_2.xlsx'
outputfile = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/tmp/test_result.csv'

data_train = pd.read_excel(inputfile1)
data_train[u'FEE_Trend'] = (data_train[u'FEE_Trend'] - data_train[u'FEE_Trend'].mean())/(data_train[u'FEE_Trend'].std())
data_train[u'FEE_Sum'] = (data_train[u'FEE_Sum'] - data_train[u'FEE_Sum'].mean())/(data_train[u'FEE_Sum'].std())
data_train[u'STM_DATA_Sum'] = (data_train[u'STM_DATA_Sum'] - data_train[u'STM_DATA_Sum'].mean())/(data_train[u'STM_DATA_Sum'].std())
data_train[u'STM_DATA_Fluctuation'] = (data_train[u'STM_DATA_Fluctuation'] - data_train[u'STM_DATA_Fluctuation'].mean())/(data_train[u'STM_DATA_Fluctuation'].std())

y_train = data_train.iloc[:, 2].as_matrix()
x_train = data_train.iloc[:, 3:].as_matrix()
x_train = round(pd.DataFrame(x_train), 2).values.tolist()

data_test = pd.read_excel(inputfile2)
data_test[u'FEE_Trend'] = (data_test[u'FEE_Trend'] - data_test[u'FEE_Trend'].mean())/(data_test[u'FEE_Trend'].std())
data_test[u'FEE_Sum'] = (data_test[u'FEE_Sum'] - data_test[u'FEE_Sum'].mean())/(data_test[u'FEE_Sum'].std())
data_test[u'STM_DATA_Sum'] = (data_test[u'STM_DATA_Sum'] - data_test[u'STM_DATA_Sum'].mean())/(data_test[u'STM_DATA_Sum'].std())
data_test[u'STM_DATA_Fluctuation'] = (data_test[u'STM_DATA_Fluctuation'] - data_test[u'STM_DATA_Fluctuation'].mean())/(data_test[u'STM_DATA_Fluctuation'].std())

y_test = data_test.iloc[:, 2].as_matrix()
x_test = data_test.iloc[:, 3:].as_matrix()
x_test = round(pd.DataFrame(x_test), 2).values.tolist()


model = Sequential() #建立模型

model.add(Dense(input_dim = 10, units = 40))
model.add(Activation('tanh'))
model.add(Dense(input_dim = 40, units = 40))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(input_dim = 40, units = 30))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(input_dim = 30, units = 30))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(input_dim = 30, units = 20))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(input_dim = 20, units = 20))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(input_dim = 20, units = 10))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(input_dim = 10, units = 1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs = 500, batch_size = 200)
model.save('D:/ProgramData/Anaconda2/envs/python36/project/project02/tmp/neural_net0.model')

r = pd.DataFrame(model.predict_classes(x_test), columns=[u'预测结果'])
# pd.concat([data_test.iloc[:, :3], r], axis=1).to_csv(outputfile)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, r))

def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
    cm = confusion_matrix(y, yp)  # 混淆矩阵
    import matplotlib.pyplot as plt  # 导入作图库
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图
    plt.colorbar()  # 颜色标签
    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    return plt

yp = model.predict_classes(x_test).reshape(len(y_test))
cm_plot(y_test,yp).show()

#CART决策树训练模型
import pandas as pd #导入数据分析库
from random import shuffle #导入随机函数shuffle，用来打算数据
from sklearn.tree import DecisionTreeClassifier #导入决策树模型
from sklearn.metrics import confusion_matrix #导入混淆矩阵函数
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO


inputfile1 = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/data/model_data_train_2.xlsx'
inputfile2 = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/data/model_data_test_2.xlsx'
outputfile = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/tmp/dt_test_result1_1.csv'
treefile = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/tmp/tree_net0.pkl' #模型输出名字
dotfilr = 'D:/ProgramData/Anaconda2/envs/python36/project/project02/tmp/tree_net0.dot'

data_train = pd.read_excel(inputfile1)
data_train[u'FEE_Trend'] = (data_train[u'FEE_Trend'] - data_train[u'FEE_Trend'].mean())/(data_train[u'FEE_Trend'].std())
data_train[u'FEE_Sum'] = (data_train[u'FEE_Sum'] - data_train[u'FEE_Sum'].mean())/(data_train[u'FEE_Sum'].std())
data_train[u'STM_DATA_Sum'] = (data_train[u'STM_DATA_Sum'] - data_train[u'STM_DATA_Sum'].mean())/(data_train[u'STM_DATA_Sum'].std())
data_train[u'STM_DATA_Fluctuation'] = (data_train[u'STM_DATA_Fluctuation'] - data_train[u'STM_DATA_Fluctuation'].mean())/(data_train[u'STM_DATA_Fluctuation'].std())

y_train = data_train.iloc[:, 2].as_matrix()
x_train = data_train.iloc[:, 3:].as_matrix()
x_train = round(pd.DataFrame(x_train), 2).values.tolist()

data_test = pd.read_excel(inputfile2)
data_test[u'FEE_Trend'] = (data_test[u'FEE_Trend'] - data_test[u'FEE_Trend'].mean())/(data_test[u'FEE_Trend'].std())
data_test[u'FEE_Sum'] = (data_test[u'FEE_Sum'] - data_test[u'FEE_Sum'].mean())/(data_test[u'FEE_Sum'].std())
data_test[u'STM_DATA_Sum'] = (data_test[u'STM_DATA_Sum'] - data_test[u'STM_DATA_Sum'].mean())/(data_test[u'STM_DATA_Sum'].std())
data_test[u'STM_DATA_Fluctuation'] = (data_test[u'STM_DATA_Fluctuation'] - data_test[u'STM_DATA_Fluctuation'].mean())/(data_test[u'STM_DATA_Fluctuation'].std())

y_test = data_test.iloc[:, 2].as_matrix()
x_test = data_test.iloc[:, 3:].as_matrix()
x_test = round(pd.DataFrame(x_test), 2).values.tolist()

tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(x_train,y_train) #训练

from sklearn.externals import joblib
joblib.dump(tree, treefile)
result = tree.predict(x_test)
# df = pd.DataFrame(result)
# df.to_csv(outputfile)
r = pd.DataFrame(tree.predict(x_test), columns=[u'cj_predict'])
# pd.concat([data_test.iloc[:, :14], r], axis=1).to_csv(outputfile)

x=data_train.iloc[:,3:].astype(int)
with open(dotfilr,'w') as f:
    f = export_graphviz(tree,feature_names=x.columns,out_file=f)

def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
    cm = confusion_matrix(y, yp)  # 混淆矩阵
    import matplotlib.pyplot as plt  # 导入作图库
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图
    plt.colorbar()  # 颜色标签
    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    return plt

yp = tree.predict(x_test).reshape(len(y_test))
cm_plot(y_test,yp).show()


