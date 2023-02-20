
kfold = StratifiedKFold(n_splits=k, shuffle=True)
splits = kfold.split(X_t