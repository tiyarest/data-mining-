import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestClassifier
traffic=pd.DataFrame(pd.read_csv('winemag-data_first150k.csv'))

#标称属性，每个可能取值的频数
print(traffic['country'].value_counts())


#######################
#price
print(traffic.isna().sum())
print(traffic['price'].describe())

#删除
#直方图
plt.hist(traffic['price'].dropna(),bins=100)
plt.savefig('wineResult\price_delete_hist.png')
plt.show()

#QQ图
sorted_ = np.sort(traffic['price'].dropna())
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(traffic['price'].dropna(), dist="norm", plot=plt)
plt.savefig('wineResult\price_delete_qq.png')
plt.show()

#盒图
plt.boxplot(traffic['price'].dropna())
plt.ylabel('price')
plt.legend()
plt.savefig('wineResult\price_delete_box.png')
plt.show()


#最高频率
#直方图
traffic=pd.DataFrame(pd.read_csv('winemag-data_first150k.csv'))


plt.hist(traffic['price'].fillna(traffic['price'].interpolate(missing_values = 'NaN', strategy = 'mode', axis = 0, verbose = 0, copy = True)),bins=100)
plt.savefig('wineResult\price_mode_hist.png')
plt.show()

#QQ图

sorted_ = np.sort(traffic['price'].fillna(traffic['price'].interpolate(missing_values = 'NaN', strategy = 'mode', axis = 0, verbose = 0, copy = True)))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(traffic['price'], dist="norm", plot=plt)
plt.savefig('wineResult\price_mode_qq.png')
plt.show()

#盒图
plt.boxplot(traffic['price'].fillna(traffic['price'].interpolate(missing_values = 'NaN', strategy = 'mode', axis = 0, verbose = 0, copy = True)))
plt.ylabel('price')
plt.legend()
plt.savefig('wineResult\price_mode_box.png')
plt.show()

#通过属性的相关关系来填补缺失值
traffic=pd.DataFrame(pd.read_csv('winemag-data_first150k.csv'))
#直方图
plt.hist(traffic['price'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True),bins=100)
plt.savefig('wineResult\price_means_hist.png')
plt.show()

#QQ图
sorted_ = np.sort(traffic['price'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(traffic['price'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True), dist="norm", plot=plt)
plt.savefig('wineResult\price_means_qq.png')
plt.show()

#盒图
plt.boxplot(traffic['price'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True))
plt.ylabel('price')
plt.legend()
plt.savefig('wineResult\price_means_box.png')
plt.show()

#通过数据对象之间的相似性来填补缺失值
traffic=pd.DataFrame(pd.read_csv('winemag-data_first150k.csv'))
known_price = traffic[traffic['price'].notnull()]
unknown_price = traffic[traffic['price'].isnull()]
x = known_price[['points']]
y = known_price[['price']]
t_x = unknown_price[['points']]
fc=RandomForestClassifier()
fc.fit(x,y)
pr=fc.predict(t_x)
traffic.loc[traffic.price.isnull(),'price'] = pr
#直方图
plt.hist(traffic['price'],bins=100)
plt.savefig('wineResult\price_relative_hist.png')
plt.show()

#QQ图
sorted_ = np.sort(traffic['price'])
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(traffic['price'])
plt.savefig('wineResult\price_relative_qq.png')
plt.show()

#盒图
plt.boxplot(traffic['price'])
plt.ylabel('price')
plt.legend()
plt.savefig('wineResult\price_relative_box.png')
plt.show()

