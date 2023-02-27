import pandas as pd
import numpy as np
import sklearn.neural_network as skl_nn
import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import StratifiedKFold
from time import time
import VisualizeNN as VisNN


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

X_test = test_data

X_train = train_data.loc[:, train_data.columns != 'Lead']
y_train = train_data['Lead']


X_test=(X_test-X_test.mean())/X_test.std()
X_train=(X_train-X_train.mean())/X_train.std()

X_train = choose_data(X_train)
# print(X_train['Total words'])

np.random.seed(200)

def dokfold(X,Y,model,splits):
    result = []
    for i, (trainI, testI) in enumerate(splits):
        model.fit(X.iloc[trainI], Y.iloc[trainI])
        prediction = model.predict(X.iloc[testI])
        # print(f'prediction: {prediction}')
        answer = Y.iloc[testI]
        
        result.append(np.mean(answer == prediction))
        print(f'split {i+1} done')
    return np.mean(result)

k = 5
kfold = StratifiedKFold(n_splits=k, shuffle=True)
splits = kfold.split(X_train, y_train)
network_struct = (500,2)
model = skl_nn.MLPClassifier(solver='lbfgs', hidden_layer_sizes=network_struct, max_iter= 1000,alpha=3e-3)
ts = time()
accuracyNN = dokfold(X_train,y_train, model, splits)
print(f'run time: {time()-ts} seconds')
print(f'Avg accuracy k-fold: {accuracyNN}')
struct = [11,2]
for _ in range(network_struct[1]):
    struct.insert(1,network_struct[0])

network=VisNN.DrawNN(struct)
network.draw()
# model.fit(X_train,y_train)
# pred = model.predict(X_train)
# print(np.mean(pred==y_train))