import numpy as np



def compute_pderivatives(listX, listY, listW, b):
    m = listX.shape[0]
    n = listX.shape[1]
    listPDerivatives = np.zeros(n)
    for i in range(n):
        pDerivative = 0
        for j in range(m):
            pDerivative += (np.dot(listW,listX[j]) + b - listY[j]) * (listX[j][i])
        listPDerivatives[i] = pDerivative

    pDerivativeb = 0 
    for i in range(m):
        pDerivativeb += (np.dot(listW,listX[i]) + b - listY[i])
    pDerivativeb = pDerivativeb/m
    listPDerivatives = listPDerivatives/m

    return listPDerivatives, pDerivativeb


def costFunc(listX, listY, listW, b):
    cost = 0.0
    for i in range(listX.shape[0]):
        cost += ((np.dot(listX[i],listW) + b) - listY[i])**2
    cost = cost/(2 *listX.shape[0])
    return cost




def gradient_Descent(listX, listY, listW, b, alpha):
    costHistory = np.zeros(100000000000)
    for i in range(100000000000):
        cost = costFunc(listX, listY, listW, b)
        costHistory[i] = cost
        listPDerivatives, pDerivativeB = compute_pderivatives(listX, listY, listW, b)
        listW = listW - alpha * listPDerivatives
        b = b - alpha * pDerivativeB

    return listW,b,costHistory
