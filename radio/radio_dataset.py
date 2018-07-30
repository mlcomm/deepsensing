import numpy as np

def to_onehot(x):
    onehot = np.zeros([len(x), max(x)+1])
    onehot[np.arange(len(x)),x] = 1
    return onehot

def datatype(Xd):
    t = map(lambda i:i, range(len(Xd.keys()[0])))
    labels = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), t)
    return labels
    
def dataset(Xd, label_index = 0):
    labels = datatype(Xd)
    label = labels[label_index]
    
    X = []
    Y = []
    for key in Xd.keys():
        X.append(Xd[key])
        for i in range(Xd[key].shape[0]):  Y.append(key)
    X = np.vstack(X)
   
    np.random.seed(2018)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    
    X_train = X[train_idx]
    X_test =  X[test_idx]
    Y_train = to_onehot(map(lambda x: label.index(Y[x][0]), train_idx))
    Y_test = to_onehot(map(lambda x: label.index(Y[x][0]), test_idx))

    return X_train, Y_train, X_test, Y_test
    
