#Importing the Dependencies 
import numpy as np 

class Logistic_Regression():

  #Declaring the Learning rate and No. of Iterations(Hyperparameters)
  def __init__(self, Learning_rate, No_of_Iterations):
    self.Learning_rate = Learning_rate
    self.No_of_Iterations = No_of_Iterations

  # Fit function to train model with dataset
  def fit(self, X, Y):

    #No. of datapoints in the dataset(No. of Rows ) ==> m 
    #No. of input features in the dataset(No. of Colomns) ==> n
    self.m, self.n = X.shape

    #Initiating the Weight and Bias Value 
    self.w = np.zeros(self.n)

    self.b = 0

    self.X = X 
    
    self.Y = Y

    # Implementing Gradient Descent for Optimization
    for i in range(self.No_of_Iterations):
      self.Update_Weights()

  def Update_Weights(self):

    #Y_cap Formula (Sigmoid Function)

    Y_cap = 1/ (1 + np.exp( -(self.X.dot(self.w)+ self.b ) )) # wX + b 

    #Derivatives 

    dw = (1/self.m)*np.dot(self.X.T, (Y_cap - self.Y))

    db = (1/self.m)*np.sum(Y_cap - self.Y)

    #Updating the Weight and bias using Gradient Descent
    self.w = self.w - Learning_rate * dw 

    self.b = self.b - Learning_rate * db 
 


  def predict(self, X):
    Y_pred = 1/ (1 + np.exp( -(X.dot(self.w)+ self.b ) ))
    Y_pred = np.where( Y_pred > 0.5, 1, 0)
    return Y_pred
