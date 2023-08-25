import numpy as np

def featureScale(x_train):
    meanList = np.mean(x_train, axis= 0)   #axis can interpreted kind of like x axis (columns) and y axis (rows)
    print(meanList)
    standardDeviationList = np.std(x_train, axis=0)
    x_norm = (x_train - meanList)/standardDeviationList    #this 2d vector operation with 1d vector can be interpreted as 1d vector operation with a number
    return x_norm, meanList, standardDeviationList   