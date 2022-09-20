from LAC import LAC

# 装载LAC模型
lac = LAC(mode='lac')

# 单个样本输入，输入为Unicode编码的字符串
text = u"市政府派遣专家团队小组前往当地进行事故调查"
lac_result = lac.run(text)

print(lac_result)
print(lac_result[0])
print(lac_result[1])
