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


def costFunc(listX, listY, listW, b, lamb):
    cost1 = 0.0
    cost2 = 0.0
    m = listX.shape[0]
    for i in range(m):
        cost1 += ((np.dot(listX[i],listW) + b) - listY[i])**2
    cost1 = cost1/(2 *listX.shape[0])
    for i in range(listW.shape[0]):
        cost2 += (listW[i] ** 2)
    cost2 = (cost2 * lamb)/2*m
    return cost1+cost2




def gradient_Descent(listX, listY, listW, b, alpha, lamb):
    costHistory = np.zeros(100000)
    m = listX.shape[0]
    for i in range(100000):
        cost = costFunc(listX, listY, listW, b, lamb)
        costHistory[i] = cost
        listPDerivatives, pDerivativeB = compute_pderivatives(listX, listY, listW, b)
        listW = listW - alpha * (lamb/m) * listW - alpha * listPDerivatives
        b = b - alpha * pDerivativeB
    return listW,b,costHistory

