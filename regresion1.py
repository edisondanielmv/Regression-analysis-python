
a=2.3
print(a)

b="edison"
print(b)

x="curso {}".format(b)

print(x)
print(len(x))

b=b.replace("edison","Katy")
x="curso {}".format(b)
print(x)
print(x[5:12:2])
print(type(a))
list=[1,2,3,4,5]
print(list)
list=[1,"a",1.1]
print(type(list[0]),type(list[1]),type(list[2]))
list1 = range(1,20)
# # print(help(range))
# list=list1.append
print(len(list1))
print(list1[18])
list.append(123)
print(list)
list[1]=99
print(list)
list.remove(99)
print(list)
del list[1]
print(list)
t1={1,2,3,4}
print(t1)
print(type(t1))
d1={"k1":1,
    "k2":2,
    "k3":3}
print(d1)
print(type(d1))
print(d1["k1"])

import numpy as np

np1 = np.array([1,2,3,4])
print(np1)
print(type(np1))

mat1= np.array([[1,2],[3,4]])
print(mat1)
print(np1.shape)
print(mat1.shape)
print(mat1.dtype)
mat1[0,0]=5
print(mat1)
mat2=np.arange(0,10,1)
print(mat2)
mat3=np.linspace(0,10,9)
print(mat3)
mat4=np.random.rand(5,5)
print(mat4)
mat5=(np.random.randn(5,5))
print(mat5)
# mat6=(np.diag(0,10))
# print(mat6)
# mat7=(np.zeros(5,5))
# print(mat7)
# mat7=(np.ones(5,5))
# print(mat7)
print(mat5[0,0])
print(mat5[0:3,:])

# PANDAS
import pandas as pd
data1 = pd.read_csv('C:/Users/ediso/Desktop/Regression Analysis in Python/Data Files/Customer.csv' , header=0)
print(data1.head(10))
data2 = pd.read_csv('C:/Users/ediso/Desktop/Regression Analysis in Python/Data Files/Customer.csv' , header=0, index_col = 0)
print(data2.head(10))
print(data1.describe())
print(data1.iloc[0])
print(data2.loc["CG-12520"])
print(data2.iloc[0])
print(data2.iloc[0:5:2])




#SEABORN

import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_theme(style="ticks")


sns.histplot(data2.Age, kde=True, color="red")
# plt.show() 


sns.pairplot(data2)
# plt.show() 



iris=sns.load_dataset("iris")
print(iris.head())
print(iris.shape)
print(iris.describe())
sns.jointplot(x="sepal_length", y="sepal_width",data=iris)

sns.pairplot(iris)

plt.show()

