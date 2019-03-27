import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestClassifier

traffic=pd.DataFrame(pd.read_csv('USvideos.csv',encoding='ANSI', low_memory=False))

#标称属性，每个可能取值的频数
print(traffic['category_id'].value_counts())

#######################
#views
print("nan:")
print(traffic.isna().sum())
print(traffic['likes'].dropna().astype(int).describe())



#删除
#直方图
plt.hist(traffic['likes'].dropna().astype(int),bins=100)
plt.savefig('videoResult\likes_delete_hist.png')
plt.show()

#QQ图
sorted_ = np.sort(traffic['likes'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(traffic['likes'].dropna().astype(int), dist="norm", plot=plt)
plt.savefig('videoResult\likes_delete_qq.png')
plt.show()

#盒图
plt.boxplot(traffic['likes'].dropna().astype(int))
plt.ylabel('likes')
plt.legend()
plt.savefig('videoResult\likes_delete_box.png')
plt.show()


#最高频率
#直方图
plt.hist(traffic['likes'].fillna(traffic['likes'].interpolate(missing_values = 'NaN', strategy = 'mode', axis = 0, verbose = 0, copy = True)),bins=100)
plt.savefig('videoResult\likes_mode_hist.png')
plt.show()

#QQ图

sorted_ = np.sort(traffic['likes'].fillna(traffic['likes'].interpolate(missing_values = 'NaN', strategy = 'mode', axis = 0, verbose = 0, copy = True)))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(traffic['likes'], dist="norm", plot=plt)
plt.savefig('videoResult\likes_mode_qq.png')
plt.show()

#盒图
plt.boxplot(traffic['likes'].fillna(traffic['likes'].interpolate(missing_values = 'NaN', strategy = 'mode', axis = 0, verbose = 0, copy = True)))
plt.ylabel('likes')
plt.legend()
plt.savefig('videoResult\likes_mode_box.png')
plt.show()

#通过属性的相关关系来填补缺失值
#直方图
plt.hist(traffic['likes'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True),bins=100)
plt.savefig('videoResult\likes_means_hist.png')
plt.show()

#QQ图
sorted_ = np.sort(traffic['likes'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(traffic['likes'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True), dist="norm", plot=plt)
plt.savefig('videoResult\likes_means_qq.png')
plt.show()

#盒图
plt.boxplot(traffic['likes'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True))
plt.ylabel('likes')
plt.legend()
plt.savefig('videoResult\likes_means_box.png')
plt.show()

#通过数据对象之间的相似性来填补缺失值
traffic = traffic[traffic['views'].notnull()]
known_price = traffic[traffic['likes'].notnull()].sample(frac=0.1)
unknown_price = traffic[traffic['likes'].isnull()]
x = known_price[['views']]
y = known_price[['likes']]
t_x = unknown_price[['views']]
fc=RandomForestClassifier()
fc.fit(x,y)
pr=fc.predict(t_x)
traffic.loc[traffic.likes.isnull(),'likes'] = pr

#直方图
plt.hist(traffic['likes'].astype(int),bins=100)
plt.savefig('videoResult\likes_relative_hist.png')
plt.show()

#QQ图
sorted_ = np.sort(traffic['likes'])
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(traffic['likes'])
plt.savefig('videoResult\likes_relative_qq.png')
plt.show()

#盒图
plt.boxplot(traffic['likes'])
plt.ylabel('likes')
plt.legend()
plt.savefig('videoResult\likes_relative_box.png')
plt.show()
