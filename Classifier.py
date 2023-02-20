import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns

def choose_data(df):
    newdf = pd.DataFrame({
                        # "percent words female": df['Number words female']/df['Total words'],
                        # "percent words male": df['Number words male']/df['Total words'],
                        "Number words female": df['Number words female'],
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
# sns.pairplot(train_data[['Year', 'Gross']])
# plt.show()

# plt.plot(train_data['Lead'])

X_test = test_data
#==============DATA ANALYSIS=============


X_train = train_data.loc[:, train_data.columns != 'Lead']
y_train = train_data['Lead']


X_test=(X_test-X_test.mean())/X_test.std()
X_train=(X_train-X_train.mean())/X_train.std()

X_train = choose_data(X_train)
print(X_train['Total words'])




# non-noralized data
# test_data = X_test
# train_data = X_train
# X_val = X_val

np.random.seed(200)




LDA = skl_da.LinearDiscriminantAnalysis()
print("\n=================LDA=================")
scores = cross_val_score(LDA,X_train,y_train,cv=5,scoring="accuracy")
print(scores.mean())

print("\n=================QDA=================")
QDA = skl_da.QuadraticDiscriminantAnalysis()
scoresQDA = cross_val_score(QDA,X_train,y_train,cv=5,scoring="accuracy")
# QDA.fit(X_train,y_train)
# print(QDA.predict(X_test))
print(scoresQDA.mean())