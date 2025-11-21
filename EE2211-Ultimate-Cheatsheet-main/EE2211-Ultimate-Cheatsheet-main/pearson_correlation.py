import numpy as np




def pearson_correlation(X,Y):
    import numpy as np
    meanX = np.mean(X, axis=1)
    standard_devX = np.std(X, axis=1)
    Var_X = np.var(X, axis=1)
    print('meanX',meanX)
    print('stdX',standard_devX)
    print('varX',Var_X)

    print(Y)
    meanY = np.mean(Y, axis=1)
    standard_devY = np.std(Y, axis=1)
    Var_Y = np.var(Y, axis=1)
    print('meanY',meanY)
    print('stdY',standard_devY)
    print('varY',Var_Y)

    pearson = []
    covs = []
    for i in range(len(meanX)):
        cov_sum = 0
        for j in range(len(Y[0])):
            cov_sum += (X[i][j] - meanX[i]) * (Y[0][j] - meanY[0])
        pearson.append(cov_sum / len(Y[0]) / standard_devY[0] / standard_devX[i])
    print(pearson)



X = np.array([[0.3510, 2.1812, 0.2415, -0.1096, 0.1544],
              [1.1769, 2.1068, 1.7753, 1.2747, 2.0851],
              [0.2758, 1.4392, -0.4611, 0.6154, 1.0006]])

Y = np.array([0.2758, 1.4392, -0.4611, 0.6154, 1.0006])

pearson_correlation(X,Y.reshape(1,5))