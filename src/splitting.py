from sklearn.model_selection import train_test_split
import numpy as np


def splitting_fun(data):

    X = data.drop(['Price'], axis = 1)
    Y = np.log(data['Price'])
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
    
    return  x_train,x_test,y_train,y_test