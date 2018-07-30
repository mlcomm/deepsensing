import pickle

def dataset_load(filepath):
    [X_train, Y_train, X_test, Y_test] = pickle.load(open(filepath,'rb'), encoding='latin1')
    return [X_train, Y_train, X_test, Y_test]

def dataset_save(X_train, Y_train, X_test, Y_test, filepath):
    pickle.dump([X_train, Y_train, X_test, Y_test], file(filepath, "wb" ) )
