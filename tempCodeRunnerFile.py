t
np.random.seed(1)
trainI = np.random.choice(train_data.shape[0], size=train_data.shape[0]-200, replace=False)

trainIndex = train_data.index.isin(trainI)
train = train_data.iloc[trainIndex]
validation = train_data.iloc[~trainIndex]

X_train = np.array(train.loc[:, train.columns != 'Lead'])
y_train = np.array(train['Lead'])

X_val = np.array(validation.loc[:, train.columns != 'Lead'])
y_val = np.array(validation['Lead'])

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
