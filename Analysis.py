import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sns.set_style("dark")

b = sns.boxplot(x = "Lead",y = "Gross",data=train_data,width=0.95)
b.set_xlabel("Gender of lead character",fontsize=15)
b.set_ylabel("Age Lead",fontsize=15)
plt.grid(True)
plt.savefig('GrossByGender')
plt.show()

