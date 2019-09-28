import pandas as pd 
from collections import Counter

with open("风格标签标注-总.xls", 'rb') as f:
    data = pd.read_excel(f)

data = data.drop(columns=[s for s in data.columns if s.startswith("Unnamed")], axis=1)
data = data.astype(str)

f_use = open('have_style.txt', 'w', encoding="utf8")
f_notuse = open("no_style.txt", 'w', encoding="utf8")

for row in data.iterrows():
    row = row[1]
    styles = row.values[-3:]
    #print(styles)
    count = Counter(styles)
    keys = list(count.keys())  # 保存的是风格
    values = list(count.values()) # 保存的是次数
    # 出现次数的最大值
    max_value = max(values)
    # 如果相同项次数大于2，并且该相同项不是“？？？”
    if max_value >= 2 and count.get("？？？", 0) < 2:
        f_use.write(row['query']+'\t'+row['response']+'\t'+keys[values.index(max_value)]+'\n')
    else:
        if str(row['response']) == 'nan':
            f_notuse.write(row['query']+'\t'+''+'\t'+'\t'.join(styles)+'\n')
        else:
            f_notuse.write(row['query']+'\t'+row['response']+'\t'+'\t'.join(styles)+'\n')