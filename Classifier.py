import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler as MMScaler

def choose_data(df):
    newdf = pd.DataFrame({
                        # "percent words female": df['Number words female']/df['Total words'],
                        # "percent words male": df['Number words male']/df['Total words'],
                        "Number words female": df['Number words female'],
                        "Number words male": df['Number words male'],
                        "Total words": df['Total words'],
                        "Number of words lead": df['Number of words lead'],
                        "Difference in words lead and co-lead":df['Difference in words lead and co-lead'],
                        "Number of male actors":df['Number of male actors'],
                        "Year":df['Year'],
                        "Number of female actors":df['Number of female actors'],                        
                        "Gross":df['Gross'],
                        "Mean Age Male":df['Mean Age Male'],
                        "Mean Age Female":df['Mean Age Female'],
                        "Age Lead":df['Age Lead'],
                        "Age Co-Lead":df['Age Co-Lead'],
                        "Lead":df['Lead'],
                        })
    return newdf


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# plt.plot(train_data['Lead'])

X_test = test_data
#==============DATA ANALYSIS=============







scaler = MMScaler()


train_data = choose_data(train_data)
print(train_data['Number words female'])
print(train_data['Number words male'])
print(train_data['Total words'])
# print(train_data['Lead'].describe())
X_train = np.array(train_data.loc[:, train_data.columns != 'Lead'])
y_train = np.array(train_data['Lead'])


# normalized data
# X_test = scaler.fit_transform(X_test)
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.fit_transform(X_val)

# non-noralized data
# test_data = X_test
# train_data = X_train
# X_val = X_val






LDA = skl_da.LinearDiscriminantAnalysis()
print("\n=================LDA=================")
scores = cross_val_score(LDA,X_train,y_train,cv=5,scoring="accuracy")
print(scores.mean())

print("\n=================QDA=================")
QDA = skl_da.QuadraticDiscriminantAnalysis()
scoresQDA = cross_val_score(QDA,X_train,y_train,cv=5,scoring="accuracy")
QDA.fit(X_train,y_train)
# print(QDA.predict(X_test))
print(scoresQDA.mean())
scoresQDA