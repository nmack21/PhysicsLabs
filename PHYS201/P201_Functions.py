import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

reg = linear_model.LinearRegression()

def linear_fit_plot(x_data,y_data,x_low="",x_high=""):
    if x_low=="":
        # Takes the x and y values to make a trendline
        reg.fit(x_data,y_data)
        print ('Linear Fit:')
        print ('Linear Fit: slope = ',reg.coef_)
        print ('Linear Fit: y_intercept = ',reg.intercept_)
        y_fit = x_data*reg.coef_+reg.intercept_
        plt.plot(x_data,y_fit,label='Linear Fit')
        return reg.intercept_,reg.coef_
    else:
        if x_high=="":
            print ('Missing x_high parameter!!')
            return -1000,-1000
        else:
            x_data_cut = []
            y_data_cut = []
            for i in range(len(x_data)):
                if x_data[i]>=float(x_low) and x_data[i]<=float(x_high):
                    x_data_cut.append(x_data[i])
                    y_data_cut.append(y_data[i])
            x_data_cut = np.array(x_data_cut)
            y_data_cut = np.array(y_data_cut)
            reg.fit(x_data_cut,y_data_cut)
            print ('Linear Fit:')
            print ('Linear Fit: slope = ',reg.coef_)
            print ('Linear Fit: y_intercept = ',reg.intercept_)
            y_fit = x_data_cut*reg.coef_+reg.intercept_
            plt.plot(x_data_cut,y_fit,label='Linear Fit')
            
            return reg.intercept_,reg.coef_
