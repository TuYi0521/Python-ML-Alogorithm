import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer

# 单个字或者字母不当作特征
data = ["生活 很短，我 喜欢 python", "生活 太久了，我 不喜欢 python"]
# cv = CountVectorizer()
# # 特征抽取
# result = cv.fit_transform(data)
# # 特征名
# print(cv.get_feature_names())
# # 特征值
# print(result.toarray())
#
# transformer = TfidfTransformer()
# tfidf = transformer.fit_transform(result)
# weight = tfidf.toarray()
# print(weight,len(weight))
#
# for i in range(len(weight)):
#     for j in range(len(cv.get_feature_names())):
#         print (weight[i][j])


# tfidf合并方法
# for i in data:
#
# res = " ".join(data)
# print(res)
tfidf = TfidfVectorizer()
result = tfidf.fit_transform(data)
# 特征名
print("TFIDF特征名",tfidf.get_feature_names())
# 特征值
print("TFIDF词频",result.toarray())
#
# result = jieba.cut("我是一个好程序员")

# 遍历分词结果，加入列表
content = []
for word in result:
    content.append(word)

print(' '.join(content))