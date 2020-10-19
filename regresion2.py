import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_csv('C:/Users/ediso/Desktop/Regression Analysis in Python/Data Files/House_Price.csv', header=0)
# print(df.head())
# print(df.shape)
# print(df.describe())
# sns.jointplot(x='n_hot_rooms',y='price',data=df)
# sns.jointplot(x='rainfall',y='price',data=df)
# sns.countplot(x='airport',data=df)
# print(df.head())
# sns.countplot(x='waterbody',data=df)
# sns.countplot(x='bus_ter', data=df)
# plt.show()
# # pd.set_option('display.max_rows', None)
# # pd.set_option('display.max_columns', None)
# # pd.set_option('display.width', None)
# # pd.set_option('display.max_colwidth', -1)




# print(df.iloc[0])
# sns.jointplot(x='crime_rate',y='price',data=df)
# sns.jointplot(x='resid_area',y='price',data=df)
# sns.jointplot(x='air_qual',y='price',data=df)
# sns.jointplot(x='room_num',y='price',data=df)
# sns.jointplot(x='age',y='price',data=df)
# sns.jointplot(x='dist1',y='price',data=df)
# sns.jointplot(x='dist2',y='price',data=df)
# sns.jointplot(x='dist3',y='price',data=df)
# sns.jointplot(x='dist4',y='price',data=df)
# sns.jointplot(x='teachers',y='price',data=df)
# sns.jointplot(x='poor_prop',y='price',data=df)
# sns.jointplot(x='airport',y='price',data=df)
# sns.jointplot(x='n_hos_beds',y='price',data=df)
# sns.jointplot(x='n_hot_rooms',y='price',data=df)
# sns.jointplot(x='waterbody',y='price',data=df)
# sns.jointplot(x='rainfall',y='price',data=df)
# sns.jointplot(x='bus_ter',y='price',data=df)
# sns.jointplot(x='parks',y='price',data=df)
# plt.show()

# ELIMINACION DE OUTLIERS
print(df.info())
print(np.percentile(df.n_hot_rooms,[99]))
print(np.percentile(df.n_hot_rooms,[99])[0])
uv=np.percentile(df.n_hot_rooms,[99])[0]
print(df[(df.n_hot_rooms>uv)])
sns.jointplot(x='n_hot_rooms',y='price',data=df)
# plt.show()

print(df.n_hot_rooms[(df.n_hot_rooms>3*uv)])
df.n_hot_rooms[(df.n_hot_rooms>3*uv)]=3*uv
print(df[(df.n_hot_rooms>uv)])
sns.jointplot(x='n_hot_rooms',y='price',data=df)
# plt.show()
sns.jointplot(x='rainfall',y='price',data=df)
# plt.show()
pc1=(np.percentile(df.rainfall,[1]))[0]
print(pc1)
print(df[(df.rainfall<pc1)])
df.rainfall[(df.rainfall<0.3*pc1)]=0.3*pc1
sns.jointplot(x='rainfall',y='price',data=df)
# plt.show()

print(df.info())
sns.jointplot(x='crime_rate',y='price',data=df)
# plt.show()
print(df.describe)






#  MISSING VALUE

print(df.info())
sns.jointplot(x='n_hos_beds',y='price',data=df)
df.n_hos_beds=df.n_hos_beds.fillna(df.n_hos_beds.mean())

print(df.info())
sns.jointplot(x='n_hos_beds',y='price',data=df)

# plt.show()

#  En caso se requiera rellenar datos para todas las columnas >>>> df= df.fillna(df.mean())





# TRANSFORMING VARIABLES
sns.jointplot(x='crime_rate',y='price',data=df)
# plt.show()
df.crime_rate=np.log(1+df.crime_rate)
sns.jointplot(x='crime_rate',y='price',data=df)
# plt.show()
df['avg_dist'] = (df.dist1+df.dist2+df.dist3+df.dist4)/4
print(df.info())
del df['dist1']
del df['dist2']
del df['dist3']
del df['dist4']
del df['bus_ter']
print(df.info())


# CREATING DUMMIES 

df=pd.get_dummies(df)
print(df.head())
print(df.info())

del df['airport_NO']
del df['waterbody_None']
print(df.info())

# CORRELATION

print(df.corr())
del df['parks']
print(df.info())

# REGRESION LINEAL

import statsmodels.api as sn
from sklearn.linear_model import LinearRegression


# STATMODELS LIBRARY  -- LINEAR REGRESSION 1 VARIABLE

# x=sn.add_constant(df['room_num'])
# lm=sn.OLS(df['price'],x).fit()
# print(lm.summary())


# SKLEARN LIBRARY -- LINEAR REGRESSION 1 VARIABLE

# y= df['price']
# x= df[['room_num']]
# lm2 = LinearRegression()
# lm2.fit(x,y)
# print(lm2.intercept_,lm2.coef_)
# print(lm2.predict(x))
# # help(sns.jointplot)
# sns.jointplot ( x=df['room_num'], y = df['price'], data=df, kind='reg')
# plt.show()


# STATMODELS LIBRARY  -- LINEAR REGRESSION MULTI VARIABLE

# x_multi = df.drop("price",axis=1)
# print(x_multi.head())
# y_multi=df["price"]
# print(y_multi.head())
# x_multi_cons=sn.add_constant(x_multi)
# print(x_multi_cons.head())
# lm_multi = sn.OLS(y_multi, x_multi_cons).fit()
# print(lm_multi.summary())

# SKLEARN LIBRARY -- LINEAR REGRESSION MULTI VARIABLE
lm3 = LinearRegression()
x_multi = df.drop("price",axis=1)
y_multi=df["price"]
lm3.fit(x_multi, y_multi)
print(lm3.intercept_,lm3.coef_)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_multi, y_multi, test_size=0.2,random_state=0)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
lm_a=LinearRegression()
lm_a.fit(x_train,y_train)
y_test_a=lm_a.predict(x_test)
y_train_a=lm_a.predict(x_train)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_test_a))
print(r2_score(y_train,y_train_a))

print(y_test_a)
print(y_train_a)


# ESTANDARIZACION DE VARIABLES
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_s=scaler.transform(x_train)
x_test_s=scaler.transform(x_test)

# BUSQUEDA DEL MEJOR MODELO
from sklearn.linear_model import Ridge
lm_r=Ridge(alpha=0.5)
lm_r.fit(x_train_s,y_train)
print(r2_score(y_test, lm_r.predict(x_test_s)))

from sklearn.model_selection import validation_curve
param_range = np.logspace(-2,8,100)
print(param_range)
train_scores, test_scores = validation_curve(Ridge(),x_train_s,y_train,"alpha",param_range,scoring='r2')
print(train_scores)
print(test_scores)
train_mean=np.mean(train_scores, axis=1)
test_mean=np.mean(test_scores, axis=1)
print(train_mean)
print(max(test_mean))
sns.jointplot(x=np.log(param_range),y=test_mean)
# plt.show()
print(np.where(test_mean==max(test_mean)))
print(param_range[31])
lm_r_best=Ridge(alpha=param_range[31])
lm_r_best.fit(x_train_s,y_train)
print(r2_score(y_test,lm_r_best.predict(x_test_s)))
print(r2_score(y_train,lm_r_best.predict(x_train_s)))

from sklearn.linear_model import Lasso
lm_1=Lasso(alpha=0.4)