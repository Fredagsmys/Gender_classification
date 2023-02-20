import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# sns.pairplot(train_data[['Year','']])
# test_data = test_data.replace('Male',1).replace('female',0)
# pysqldf = lambda q: sqldf(q, globals())
# response = sqldf(text("SELECT 'Year' FROM train_data LIMIT 10;"),locals())
sns.set_style("dark")
b = sns.boxplot(x = "Lead",y = "Gross",data=train_data,width=0.95)
b.set_xlabel("Gender of lead character",fontsize=15)
b.set_ylabel("Gross income",fontsize=15)
plt.grid(True)
plt.savefig('GrossByGender')
plt.show()


