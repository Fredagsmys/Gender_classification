import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import StratifiedKFold,GridSearchCV

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

X = train_data.loc[:, train_data.columns != 'Lead']
y = train_data['Lead']


X = choose_data(X)

def do_kfold(Xk,Yk,model,splits):
    result = []
    for i, (trainI, testI) in enumerate(splits):
        model.fit(Xk.iloc[trainI], Yk.iloc[trainI])
        prediction = model.predict(Xk.iloc[testI])
        answer = Yk.iloc[testI]
        result.append(np.mean(answer == prediction))
    return np.mean(result)

k = 5
kfold = StratifiedKFold(n_splits=k, shuffle=True)
splits = kfold.split(X, y)
LDA = skl_da.LinearDiscriminantAnalysis()
print("\n=================LDA=================")
accuracyLDA = do_kfold(X,y, LDA, splits)
print(f'Avg accuracy k-fold: {accuracyLDA}')


print("\n=================QDA=================")
params = [{'reg_param': np.linspace(0,1,100)}]
clf = GridSearchCV(skl_da.QuadraticDiscriminantAnalysis(), params, cv=k)
clf.fit(X,y)

best_param = clf.best_params_['reg_param']
print(f"best regularization parameter is {best_param} and gives an accuracy of {clf.best_score_*100:.2f}%")