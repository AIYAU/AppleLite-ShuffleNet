import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

# 从Excel文件中读取数据
df = pd.read_excel('D:\Awesome-Backbones/utils/a.xlsx')

# 进行One-way ANOVA
f_statistic, p_value = f_oneway(df['tar'], df['water'], df['coke'])

# 输出ANOVA结果
print("ANOVA结果:")
print("F统计量:", f_statistic)
print("P值:", p_value)

# 创建箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['tar', 'water', 'coke']], width=0.5)
plt.title('不同产率下的箱线图')
plt.xlabel('产率类型')
plt.ylabel('产率值')
plt.show()
