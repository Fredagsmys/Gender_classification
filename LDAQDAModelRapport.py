import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import StratifiedKFold

def choose_data(df):
    newdf = pd.DataFrame({
                        # "percent words female": df['Number words female']/df['Total words'],
                        # "percent words male": df['Number words male']/df['Total words'],
                        "Number words female": df['Number words female'],
                        # "Number words male": df['Number words male'],
                        "Total words": df['Total words'],
                        "Number of words lead": df['Number of words lead'],
                        "Difference in words lead and co-lead":df['Difference in words lead and co-lead'],
                        "Number of male actors":df['Number of male actors'],
                        # "Year":df['Year'],
                        "Number of female actors":df['Number of female actors'],                       
                        "Gross":df['Gross'],
                        "Mean Age Male":df['Mean Age Male'],
                        "Mean Age Female":df['Mean Age Female'],
                        "Age Lead":df['Age Lead'],
                        "Age Co-Lead":df['Age Co-Lead'],
                        # "Lead":df['Lead'],
                        })
    return newdf

np.random.seed(200)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_test = test_data

X_train = train_data.loc[:, train_data.columns != 'Lead']
y_train = train_data['Lead']

X_test=(X_test-X_test.mean())/X_test.std()
X_train=(X_train-X_train.mean())/X_train.std()

X_train = choose_data(X_train)

def do_kfold(X,Y,model,splits):
    result = []
    for i, (trainI, testI) in enumerate(splits):
        model.fit(X.iloc[trainI], Y.iloc[trainI])
        prediction = model.predict(X.iloc[testI])
        answer = Y.iloc[testI]
        result.append(np.mean(answer == prediction))
    return np.mean(result)

k = 5
kfold = StratifiedKFold(n_splits=k, shuffle=True)
splits = kfold.split(X_train, y_train)
LDA = skl_da.LinearDiscriminantAnalysis()
print("\n=================LDA=================")
accuracyLDA = do_kfold(X_train,y_train, LDA, splits)
print(f'Avg accuracy k-fold: {accuracyLDA}')
print("\n=================QDA=================")

splits = kfold.split(X_train, y_train)
QDA = skl_da.QuadraticDiscriminantAnalysis()
accuracyQDA = do_kfold(X_train,y_train, QDA, splits)
print(f'Avg accuracy k-fold: {accuracyQDA}')