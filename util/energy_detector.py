def ed_power(dataset, EbN0):
  Xd = dataset
def cal_pd_pf(dataset, SNR, threshold, vec_length):
    Xd = dataset
    X_busy = Xd[('busy', SNR)]
    X_idle = Xd[('idle', SNR)]
    
    pbusy = np.zeros([X_busy.shape[0], 1])
    for i in range(0, X_busy.shape[0]):
        x = X_busy[i]
        pbusy[i] = np.sum(np.multiply(x, x))/vec_length
    
    pidle = np.zeros([X_idle.shape[0], 1])
    for i in range(0, X_idle.shape[0]):
        x = X_idle[i]
        pidle[i] = np.sum(np.multiply(x, x))/vec_length
        
    print "busy power range ", [np.amin(pbusy), np.amax(pbusy)]
    print "idle power range ", [np.amin(pidle), np.amax(pidle)]    
    
    pd = 0
    pf = 0
    for i in range(0, len(pbusy)):
        if pbusy[i] >= threshold:
            pd += 1  

    for i in range(0, len(pidle)):
        if pidle[i] >= threshold:
            pf += 1  

    pd = pd/(len(pbusy) + 0.0)
    pf = pf/(len(pbusy) + 0.0)
    print("pd ", pd)
    print("pf ", pf)
    
    return [threshold, pd, pf]
