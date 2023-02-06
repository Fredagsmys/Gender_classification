import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.preprocessing import MinMaxScaler as MMScaler

def choose_data(df):
    newdf = pd.DataFrame({"percent words female": df['Number words female']/df['Total words'],
                        "percent words male": df['Number words male']/df['Total words'],
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


scaler = MMScaler()


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
X_test = test_data
print(train_data.info())
print(choose_data(train_data).info())
train_data = choose_data(train_data)

# Splitting dataset




np.random.seed(1)
trainI = np.random.choice(train_data.shape[0], size=train_data.shape[0]-200, replace=False)
trainIndex = train_data.index.isin(trainI)
train = train_data.iloc[trainIndex]
validation = train_data.iloc[~trainIndex]
X_train = np.array(train.loc[:, train.columns != 'Lead'])
y_train = np.array(train['Lead'])
X_val = np.array(validation.loc[:, train.columns != 'Lead'])
y_val = np.array(validation['Lead'])
#end


test_data = scaler.fit_transform(X_test)
train_data = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)








LDA = skl_da.LinearDiscriminantAnalysis()
LDA.fit(X_train, y_train)

prediction = LDA.predict(X_val)
err = np.mean(prediction == y_val)
confusion = pd.crosstab(prediction,y_val)
print("\n=================LDA=================")
print(f"Confusion matrix:\n{confusion}")
print(f"\nError on validation set with LDA: {1-err}")

print("\n=================QDA=================")
QDA = skl_da.QuadraticDiscriminantAnalysis()
QDA.fit(X_train, y_train)

prediction = QDA.predict(X_val)
err = np.mean(prediction == y_val)
confusion = pd.crosstab(prediction,y_val)
print(f"Confusion matrix:\n{confusion}")
print(f"\nError on validation set with QDA: {1-err}")
