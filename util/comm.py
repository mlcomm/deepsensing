def cal_roc(test_Y_hat):
    pds = []
    pfs = []
    for j in range(0,100):
        delta = 1/100*j
        Y_predict = np.zeros([len(Y_test),2])
        for i in range(0,X_test.shape[0]):
           # if test_Y_hat[i,1] -test_Y_hat[i,1] < delta:
            if test_Y_hat[i,1] < delta:
                Y_predict[i, 0]=1
            else: 
                Y_predict[i, 1]=0
        pd = 0
        pd_count = 0
        pf = 0
        px = 0
        for i in range(0, X_test.shape[0]):
            if Y_test[i,0] == 1:
                pd_count += 1

            if Y_test[i,0] == 1 and Y_predict[i,0] == 1:
                pd += 1
            if Y_test[i,0] == 0 and Y_predict[i,0] == 1:
                pf += 1
            if Y_test[i,0] == 0 and Y_predict[i,0] == 0:
                px += 1   
            
        pds.append(pd/(pd_count + 0.0))
        pfs.append(pf/(Y_test.shape[0]-pd_count + 0.0))
        
    return pds, pfs

  def plot_roc(pds,pfs):
    try:
      import matplotlib.pyplot as plt
      plt.plot( pfs, pds)
      plt.xlabel('pfs')
      plt.ylabel('pds')
      plt.show()
    except ImportError as error:
      print(error.__class__.__name__ + ": " + error.message)
      
      
 def cal_pd_pf(X_test, Y_test, test_Y_hat):
    Y_predict = np.zeros([len(Y_test),2])
    for i in range(0,X_test.shape[0]):
        k = int(np.argmax(test_Y_hat[i,:]))
        Y_predict[i, k] = 1
    pd = 0
    pd_count = 0
    pf = 0
    px = 0
    for i in range(0, X_test.shape[0]):
        if Y_test[i,0] == 1:
            pd_count += 1

        if Y_test[i,0] == 1 and Y_predict[i,0] == 1:
            pd += 1
        if Y_test[i,0] == 0 and Y_predict[i,0] == 1:
            pf += 1
        if Y_test[i,0] == 0 and Y_predict[i,0] == 0:
            px += 1
            
    pd = pd/(pd_count + 0.0)
    pf = pf/(Y_test.shape[0]-pd_count + 0.0)
    return pd, pf
