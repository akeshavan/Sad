import sys
import os
import numpy as np
import nibabel as nib
from glob import glob
import rpy2.robjects as rob
from rpy2.robjects import FloatVector as FV
from rpy2.robjects.packages import importr
from sklearn.decomposition import PCA
from sklearn import linear_model
import rpy2.robjects as rob
from rpy2.robjects import FloatVector as FV
from rpy2.robjects.packages import importr
import sklearn.cross_val as cv
from sklearn.linear_model.least_angle import LassoLarsCV
import matplotlib.pyplot as plt
import sklearn
import sklearn.feature_selection as fs
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

stats = importr('stats')
base = importr('base')

from Outputsclass import Outputsclass
from regression_functions import do_Regression #mult_lm = do_Regression(y,Vals,gr=0)

def do_FS(X,y):
    Reg = linear_model.RidgeCV()
    
    #Reg1 = sklearn.svm.SVR(kernel = 'linear')
    #Reg1.fit(X,y)
    selector = fs.RFECV(Reg, step=1, cv=3)
    #selector = RFE(Reg,1,step = 1)
    #Reg.fit(X,y)
    #print Reg.coef_
    #selector = selector.fit(X, y)
    #print selector.support_ 
    #print selector.ranking_
    return selector, Reg



if __name__ == "__main__":

    pdata = np.recfromcsv('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l2output/social/split_halves/regression/lsasDELTA/6mm/allsubs.csv',names=True)
    
    subject_num = len(pdata.subject)
    # initialize dependent variable
    y = pdata.lsas_pre-pdata.lsas_post
    ind_variables_num = 4 #if change number here, also modify assignments below (and vice versa)
    # initialize design matrix
    X = np.zeros([subject_num,ind_variables_num])
    X[:,0] = pdata.lsas_pre
    X[:,1] = pdata.classtype-2
    X[:,2] = pdata.age
    X[:,3] = pdata.sex- 1
    print "running FS"
    
    from sklearn.datasets import make_friedman1

    X1, y1 = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    selector1 = RFE(estimator, 3, step=1)
    selector1 = selector1.fit(X, y)
    
    
    selector, Reg = do_FS(X,y)
    
    
    
    
    
