import numpy as np
import matplotlib.pyplot as plt
from Packages import LogisticGradientDescent as gd
from Packages import LoadData as ld
plt.style.use(plt.style.available[0])

#This program uses the regularized gradien descent algorithm to predict whether the given patient has a malignant tumor or not.
#The features (specifically for the training vector in the folder) are tumor size and amount of years patient have had tumor
#You can include more features such as patient's age, patient's contraction to other illness, etc. and the code should work fine
#However, due to restriction of plt the data visualization will be compromised if the number of features smaller or bigger than 2
#The training data and data to be assessed both should be a CSV file 


def modelEquation(features, w):
    equation = ''
    for i in range(w.shape[0]):
        if i == w.shape[0] - 1:
            equation = equation + str(w[i]) + '*' + str(features[i]) + " " 
        else: 
            equation = equation + str(w[i]) + '*' + str(features[i]) + " " + '+' + " "
    return equation

x_train, y_train, features = ld.loadData()
x_train = np.array(x_train)
y_train = np.array(y_train)
#gd.scatterPlot(x_train, y_train)  #use this if you have exactly two features to visualize the data

w_train = np.zeros(x_train.shape[1])
b = 0
alpha = 0.12
lamb = 2

w_train, b, costHistory = gd.gradient_Descent(x_train, y_train, w_train, b, alpha, lamb)
#gd.decisionBoundary(w_train, b)  #use this if youi have exactly two features to visualize the data
print(f"The data has been trained. Your data is modeled by the equation:\n1/(1 + e^{modelEquation(features, w_train)})")
x_check = ld.loadInputData()
x_check = np.array(x_check)
print("Applying this model to the data you have provided and generating the result. 0 represents non-malignant and 1 represents malignant")
resultData = gd.apply(x_check, w_train, b)
print(features)
for i in range(x_check.shape[0]):
    print(f"{x_check[i]} {resultData[i]}")
    