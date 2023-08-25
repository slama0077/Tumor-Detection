import numpy as np
import matplotlib.pyplot as plt

def sigmoidFunction(x, w, b):
    funcValue = 1/(1 + np.exp(-(np.dot(x, w) + b)))
    return funcValue


def costFunc(listX, listY, listW, b, lamb):
    cost1 = 0.0
    cost2 = 0.0
    m = listX.shape[0]
    for i in range(m):
        funcValue = sigmoidFunction(listX[i], listW, b)
        cost1 = cost1 - (listY[i] * np.log(funcValue)) - (1-listY[i]) * np.log(1- funcValue)
    cost1 = cost1 / m
    for i in range(listW.shape[0]):
        cost2 += (listW[i]**2)
    cost2 = (cost2 * lamb)/(2*m)

    return (cost1 + cost2)

def compute_pderivatives(listX, listY, listW, b, lamb):
    m = listX.shape[0]
    n = listX.shape[1]
    listPDerivatives = np.zeros(n)

    for i in range(n):
        pDerivative = 0
        for j in range(m):
            funcValue = sigmoidFunction(listX[j], listW, b)
            pDerivative += (funcValue - listY[j]) * (listX[j][i])
        listPDerivatives[i] = pDerivative

    pDerivativeb = 0 
    for i in range(m):
        funcValue = sigmoidFunction(listX[i], listW, b)
        pDerivativeb += (funcValue - listY[i])
    pDerivativeb = pDerivativeb/m
    listPDerivatives = listPDerivatives/m
    listPDerivatives = listPDerivatives + (lamb/m) * listW

    return listPDerivatives, pDerivativeb



def gradient_Descent(listX, listY, listW, b, alpha, lamb):
    costHistory = np.zeros(100000)
    m = listX.shape[0]
    for i in range(100000):
        cost = costFunc(listX, listY, listW, b, lamb)
        costHistory[i] = cost
        listPDerivatives, pDerivativeB = compute_pderivatives(listX, listY, listW, b, lamb)
        listW = listW  -  (alpha * listPDerivatives)
        b = b - alpha * pDerivativeB
        
    return listW,b,costHistory

def scatterPlot(x_train, y_train):
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            plt.scatter(x_train[i][0], x_train[i][1], marker = '+', c = 'r', label = 'malignant')
        else:
            plt.scatter(x_train[i][0], x_train[i][1],  c = 'b', marker = 'o', label = 'non-malignant')

def decisionBoundary(w, b):
    x_graph = np.arange(20)
    w1 = w[0]
    w2 = w[1]
    y_graph = (-1) * x_graph * w1/w2 - b
    plt.plot(x_graph, y_graph)

def modelEquation(features, w):
    equation = ''
    for i in range(w.shape[0]):
        if i == w.shape[0] - 1:
            equation = equation + str(w[i]) + '*' + str(features[i]) + " " 
        else: 
            equation = equation + str(w[i]) + '*' + str(features[i]) + " " + '+' + " "
    return equation

def apply(listX, listW, b):
    resultData = np.zeros(listX.shape[0])
    for i in range(listX.shape[0]):
        result = sigmoidFunction(listX[i], listW, b)
        if result < 0.5:
            resultData[i] = 0
        else:
            resultData[i] = 1
    return resultData