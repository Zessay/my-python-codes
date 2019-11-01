=# 输入的形式为：`{1:2, 4:3, 5:1}`，其中键表示要采样的值，值表示对应出现的次数。要求按照频率每次随机输出一个采样值。
def generate(input_dict):
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    
    sum_values = sum(values)
    # 计算概率
    probas = []
    for v in values:
        probas.append(v / sum_values)
    
    # 产生随机数
    rand = random.uniform(0, 1)
    # 累积概率
    cum_proba = 0
    for value, proba in zip(values, probas):
        cum_proba += proba
        if cum_proba >= rand:
            return value