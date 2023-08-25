import csv


def loadData():
    while True: 
        try:    
            filename = input("Provide the name of your file. The file must be a CSV file: ")
            istream = open(filename, mode = 'r', encoding = 'utf-8')
        except: 
            print(f"There is no such file called {filename}. Or maybe you have not provided the extension. Try again!")
        else:
            print("The file name provided is correct. Thank you!")
            break
    
    csv_input = csv.reader(istream)
    data = list(csv_input)
    feature_length = len(data[0])
    features = data[0][0:feature_length-1]
    i = 1
    x_train = [[]]
    y_train = []
    while i < len(data):
        x = []
        j = 0
        while j < feature_length:
            if j < (feature_length-1):
                x.append(float(data[i][j]))
            else:
                y_train.append(float(data[i][j]))
            j += 1
        x_train.append(x)
        i += 1
    x_train = x_train[1:]
    istream.close()
    return (x_train, y_train, features)

def loadInputData():
    while True: 
        try:    
            filename = input("Provide the name of your file that you want to apply this model to. The file must be a CSV file: ")
            istream = open(filename, mode = 'r', encoding = 'utf-8')
        except: 
            print(f"There is no such file called {filename}. Or maybe you have not provided the extension. Try again!")
        else:
            print("The file name provided is correct. Thank you!")
            break
    csv_input = csv.reader(istream)
    data = list(csv_input)
    feature_length = len(data[0])
    i = 1
    x_train = [[]]
    while i < len(data):
        x = []
        j = 0
        while j < feature_length:
            x.append(float(data[i][j]))
            j += 1
        x_train.append(x)
        i += 1
    x_train = x_train[1:]
    istream.close()
    return x_train