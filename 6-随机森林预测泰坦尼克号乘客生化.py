import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# 1、pd读取数据
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
# 2、选择有影响的特征
# 区分特征值和目标值
x = data[["pclass", "age", "sex"]]
y = data[["survived"]]
# 3、填补缺失值age
x["age"].fillna(x["age"].mean(), inplace=True)
# 4、数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1,random_state=1)
print("x_train:",x_train)
print("y_train:",y_train)
print("x_test:",x_test)
print("y_test:",y_test)
# 5、转换成字典，对数据集特征抽取将有类别的特征（票类，性别）变成One-Hot编码形式
x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
print("xtrain_dict",x_train)
print("xtest_dict",x_test)
dv = DictVectorizer(sparse=False)
# 特征名
x_train = dv.fit_transform(x_train)
x_test = dv.transform(x_test)
print("fit_train_x",x_train)
print("fit_test_x",x_test)
# print(dv.get_feature_names())

# 6、随机森林估计器流程
rfc = RandomForestClassifier(n_estimators=5, max_depth=4, criterion="entropy")
rfc.fit(x_train, y_train)
score = rfc.score(x_test, y_test)
print(score)
y_predict = rfc.predict(x_test)
print("y_predict:",y_predict)
report = classification_report(y_true=y_test, y_pred=y_predict)
print(report)