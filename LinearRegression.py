import numpy as np

# Linear Regression Model Code From Scratch
class LinerRegression:
    # Initaiting the parameters (Learning rate & no. of iterations)
    def __init__(self, learning_rate,no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    # Function to Fit the data to the model
    def fit(self,X,Y):
        # number of trainng examples & number of features
        self.m,self.n = X.shape # it the number of rows & columns m=>rows,n=>columns

        # initiating the wights and bias 
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # implementing the gradient descent algorithm
        for i in range(self.no_of_iterations):
            self.update_wights()
    

    def update_wights(self,):
        Y_prediction = self.predict(self.X)
        
        # calculate gradients
        dw = -(2*(self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = -2 * np.sum(self.Y - Y_prediction)/self.m

        # updating the weights
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate * db
        
    # function for the prediction  
    def predict(self,X):
        return X.dot(self.w) + self.b