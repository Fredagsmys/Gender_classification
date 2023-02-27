import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def choose_data(df):
    newdf = pd.DataFrame({
                        "percent words female": df['Number words female']/df['Total words'],
                        "percent words male": df['Number words male']/df['Total words'],
                        # "Number words female": df['Number words female'],
                        # "Number words male": df['Number words male'],
                        "Total words": df['Total words'],
                        "Number of words lead": df['Number of words lead']*1.1,
                        "Difference in words lead and co-lead":df['Difference in words lead and co-lead']*1.5,
                        "Number of male actors":df['Number of male actors']*1,
                        # "Year":df['Year'],
                        "Number of female actors":df['Number of female actors'],                       
                        "Gross":df['Gross']*1.5,
                        "Mean Age Male":df['Mean Age Male']*0.8,
                        "Mean Age Female":df['Mean Age Female'],
                        "Age Lead":df['Age Lead'],
                        "Age Co-Lead":df['Age Co-Lead']*0.01,
                        # "Lead":df['Lead'],
                        })
    return newdf


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data = choose_data(train_data)
train_data["Gross"] = train_data["Gross"].rolling(51).mean().shift(-25)
newdf = pd.DataFrame({"percent_fem" : train_data['percent words female'], "Gross": train_data['Gross']})
newdf.hist()
# grossfemale = train_data["Gross"].rolling(7).mean().shift(-3)
# sns.pairplot(train_data[['Year','']])
# test_data = test_data.replace('Male',1).replace('female',0)
# pysqldf = lambda q: sqldf(q, globals())
# response = sqldf(text("SELECT 'Year' FROM train_data LIMIT 10;"),locals())
sns.set_style("dark")

# b = sns.boxplot(x = "Lead",y = "Gross",data=train_data,width=0.95)

# b.set_xlabel("Gender of lead character",fontsize=15)
# b.set_ylabel("Gross income",fontsize=15)
# g = sns.plot(data = train_data, x="percent words female", y= "Gross",)
plt.grid(True)
plt.savefig('GrossByGender')
plt.show()

b = sns.boxplot(x = "Lead",y = "Age Lead",data=train_data,width=0.95)
b.set_xlabel("Gender of lead character",fontsize=15)
b.set_ylabel("Age Lead",fontsize=15)
plt.grid(True)
# plt.savefig('GrossByGender')
plt.show()

