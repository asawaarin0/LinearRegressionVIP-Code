#I made this code during week 7-8 but noticed today that it wasn't upload to gogs so I am now uploading it
import numpy as np
class Linear_Regression:
    def __init__(self, m, b, l):
        #initialize m and b model parameters
        self.m = m
        self.b = b
        #initialize learning rate
        self.l = l

    def train(self, X, Y, num_iterations):
        #X is a vector consisting of feature values 
        #Y is a vector consistsing of the corresponding target values 
        #num_iterations specifies how many ierations of gradient descent to conduct
        
        n = float(len(X))
        loss = np.zeros(num_iterations)
        for i in range(num_iterations):
            predicted_values = self.m * X + self.b
            #store calculated loss at every iteration
            loss[i] = sum(np.square((Y - predicted_values)))/n

            #calculate and update gradients
            d_m = sum(X * (Y - predicted_values)) * (-2/n)
            d_b = sum(Y - predicted_values) * (-2/n)
            self.m = self.m - (self.l * d_m)
            self.b = self.b = (self.l * d_b)
        
    def predict(self, X):
        #uses the model to predict target values for input feature values
        return self.m * X + self.b