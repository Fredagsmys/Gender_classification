import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from pandasql import sqldf
from sqlalchemy import text

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# sns.pairplot(train_data[['Year','']])
# test_data = test_data.replace('Male',1).replace('female',0)
# print(test_data)
# query = 
# pysqldf = lambda q: sqldf(q, globals())
# response = sqldf(text("SELECT 'Year' FROM train_data LIMIT 10;"),locals())
# print(type((train_data[train_data['Lead'] == 'Female']['Gross']).tolist()))
sns.set_style("dark")
b = sns.boxplot(x = "Lead",y = "Gross",data=train_data,width=0.95)
b.set_xlabel("Gender of lead character",fontsize=15)
b.set_ylabel("Gross income",fontsize=15)
plt.grid(True)


# plt.xlabel('Gender of lead character')
# plt.ylabel('Gross income')
plt.savefig('GrossByGender')
plt.show()
# plt.boxplot([train_data[train_data['Lead'] == 'Female']['Gross'], train_data[train_data['Lead'] == 'Male']['Gross']], labels = ['Female', 'Male'], autorange=True, meanline=True)
# plt.title('Boxplot of gross income of movie per gender')


