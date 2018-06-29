import numpy as np
import pandas as pd
from operator import itemgetter
from numpy import linalg as Edist
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
 
def nearest_25(X_test,x_train,y_train): #Calculating 25 Nearest Neighbour
    dist={}
    X_25=np.zeros(shape=(25,5))
    y_25=np.zeros(shape=(25,))
    for i in range(x_train.shape[0]):
        dist[i]=Edist.norm(X_test-x_train[i])
    sorted_dist=sorted(dist.items(), key = itemgetter(1))
    for i,(a, b) in enumerate(sorted_dist[:25]):
        X_25[i]=x_train[a]
        y_25[i]=y_train[a]
    return X_25,y_25

def get_weight(x, x0, c=0.2): #Calculating weight for test instance with other instance in training set.
    diff = x - x0
    dot_product = np.dot(diff,diff.T)
    return np.exp(dot_product / (-2.0 * c**2))

def feature_derivative(error, feature, weight): #Calculating Partial derivative
        derivative = 2*weight*error*feature
        return(derivative)
    
def Lwr_predict(X_test,H,Y,learning_rate=0.1,epsilon=6e-15,max_iterations=1000):

    '''
Obtaining  the test instance parameter and predicting the value by minimizing the error function,
Applying Gradient descent on error function which is formed by considering the distance of training instance from test instance and prediction error.(As explained in the lecture) Applying gradient descent by considering training instance one by one and updating the feature weight value.
Parameter:
    X_test: Test instance
    H: Training instance
    learning rate: 0.1
    epsilon: Mininum error after which it stop or converged.
    max_iteration: Maximium iteration after which it stop or converged.
    The hyper-parameter value used are:
    c=0.2, learning rate = 0.1
    
    '''
    x_25,y_25= nearest_25(X_test,H,Y)
    x_25=np.c_[np.ones(x_25.shape[0]),x_25]
    X_test.shape=(1,5)
    X_test=np.c_[np.ones(X_test.shape[0]),X_test] 
    feat_weight = np.array(np.zeros(x_25.shape[1],)) # Initially all feature weight are initialized to zero.
    converged=False
    iteration = 0
    while not converged:
        if iteration > max_iterations:
                #print('Exceeded max iterations\nFeature weight:', feat_weight)
                return np.dot(X_test,feat_weight)
        error=0.0
        for i in range(x_25.shape[0]):
            f=np.dot(x_25[i],feat_weight)
            residual=(f-y_25[i])
            weight=get_weight(x_25[i],X_test[0])
            error=error+(residual**2) * weight
            if error < epsilon:
                converged = True
                continue
            for j in range(len(feat_weight)):
                partial = feature_derivative(residual, x_25[i,j],weight)
                feat_weight[j] = feat_weight[j] - learning_rate * partial
        iteration += 1
    #print("Feature weight\t:" ,feat_weight)
    return np.dot(X_test,feat_weight)

'''
Reading dataset
'''
dataset=pd.read_csv('eda-18-ass1-data.txt', delimiter=' ',header=None)
X=dataset.iloc[: , 0:6].values
y=dataset.iloc[: , 6].values

'''
Splitting Dataset into training and test dataset.(Splitting Criteria: 90% for Training and 10% for Testing the Model)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

'''
Predicting the value for the X_test, After obtaining the model parameter for each test_data by considering the 25 nearest neighbour from X_train and y_train.

'''
y_pred=np.zeros(y_test.shape)
for i,test_data in enumerate(X_test[:,1:6]):
    y_pred[i]=Lwr_predict(test_data, X_train[:,1:6], y_train)
    
'''
Generating the result.csv file
'''
result_data=np.c_[X_test,y_test,y_pred]
np.savetxt('result.csv',result_data, delimiter="\t",fmt='%10.2f',header='\tindex\t\tfeat1\t\tfeat2\t\t\tfeat3\t\tfeat4\tfeat5\t Actual_Val\t\tPred_Val')    

'''
Evaluating the locally weighted regression model : y_pred(Predicted value) and y_test(Actual value)
'''

Mean_abs_err=metrics.mean_absolute_error(y_pred , y_test) # Calculating Mean Absolute Error
Mean_sqr_err=metrics.mean_squared_error(y_pred , y_test)  #Calculating Mean Squared Error
Root_mean_sqr_err=math.sqrt(Mean_sqr_err)   #Calculating Root Mean Squared Error

print("\n\nMean absolute error\t:",Mean_abs_err)
print("Mean square error\t:",Mean_sqr_err)
print("Root mean square error\t:",Root_mean_sqr_err)
