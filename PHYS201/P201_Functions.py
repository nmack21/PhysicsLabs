import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

reg = linear_model.LinearRegression()

def linear_fit_plot(x_data,y_data,x_low="",x_high=""):
    if x_low=="":
        # Takes the x and y values to make a trendline
        reg.fit(x_data,y_data)
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
            print ('Linear Fit: slope = ',reg.coef_)
            print ('Linear Fit: y_intercept = ',reg.intercept_)
            y_fit = x_data_cut*reg.coef_+reg.intercept_
            plt.plot(x_data_cut,y_fit,label='Linear Fit')
            
            return reg.intercept_,reg.coef_

from scipy.optimize import curve_fit

def linearfitfunction(x,*paramlist):
    return paramlist[0]+paramlist[1]*x

def linear_fit_plot_errors_core(xi,yi):
    
    init_vals = [0.0 for x in range(2)]
    popt, pcov = curve_fit(linearfitfunction,xi,yi,p0=init_vals)
    perr = np.sqrt(np.diag(pcov))

    ps = np.random.multivariate_normal(popt,pcov,10000)
    ysample=np.asarray([linearfitfunction(xi,*pi) for pi in ps])

    lower = np.percentile(ysample,2.5,axis=0)
    upper = np.percentile(ysample,97.5,axis=0)
    middle = (lower+upper)/2.0

    print("Coefficients (from curve_fit)")
    print (popt)
    print("Covariance Matrix (from curve_fit)")
    print (pcov)

    print()
    print ("Final Result: y = (%0.2f +/- %0.2f) x + (%0.2f +/- %0.2f)" % (popt[1],perr[1],popt[0],perr[0]))

    #plt.plot(xi,yi,'o')

    plt.plot(xi,middle,label='Linear Fit')
    plt.plot(xi,lower)
    plt.plot(xi,upper)

    return popt[1],perr[1],popt[0],perr[0]

def linear_fit_plot_errors(xi,yi,x_low="",x_high=""):
    if x_low=="":
        # Takes the x and y values to make a trendline
        intercept,slope,dintercept,dslope = linear_fit_plot_errors_core(xi,yi)
        return intercept,slope,dintercept,slope
    else:
        if x_high=="":
            print ('Missing x_high parameter!!')
            return -1000,-1000,-1000,-1000
        else:
            x_data_cut = []
            y_data_cut = []
            for i in range(len(xi)):
                if xi[i]>=float(x_low) and xi[i]<=float(x_high):
                    x_data_cut.append(xi[i])
                    y_data_cut.append(yi[i])
            x_data_cut = np.array(x_data_cut)
            y_data_cut = np.array(y_data_cut)
            # Takes the x and y values to make a trendline
            intercept,slope,dintercept,dslope = linear_fit_plot_errors_core(x_data_cut,y_data_cut)
            return intercept,slope,dintercept,slope
