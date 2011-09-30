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

stats = importr('stats')
base = importr('base')

from Outputsclass import Outputsclass

def do_Regression(y,Vals,gr=0):
    formulastr = 'y ~ '
    rob.globalenv["y"] = FV(y)
    if type(gr) == type(y):
        rob.globalenv["group"] = FV(gr)
        formulastr += 'group +'
    for i in xrange(Vals.shape[1]):
            addstr = " roi%02d +"%i
            if type(gr) == type(y):
                addstr1 = " roi%02d:group +"%i  
            name = "roi%02d"%i
            rob.globalenv[name] = FV(Vals[:,i]) 
            formulastr += addstr
            if type(gr) == type(y):
                formulastr += addstr1

    formulastr += " 1"
    mult_lm = stats.lm(formulastr)
    return mult_lm

def do_R_Crossval(y,yname,files,X,gr = 0):
    subject_num = y.shape[0]
    output = Outputsclass(subject_num,y,yname)
    data, idx, img  = load_data(files)
    for train, test in cv.LeaveOneOut(subject_num):
        pca, data_red = do_PCA(data[train])
        output.pca.append(pca)
        data_red_test = pca.transform(data[test])
        desmat_cv = np.hstack((data_red,X[train]))
        desmat_cv = np.array(desmat_cv)
        y_cv = y[train]
        if type(gr) == type(y):
            gr_cv = gr[train]
        else:
            gr_cv = 0
        
        test_vec = np.hstack((data_red_test,X[test]))
    
        test_vec = np.array(test_vec)
        mult_lm = do_Regression(y_cv,desmat_cv,gr_cv)
        output.mult_lm.append(mult_lm)
        output.rsq[test] = np.array(base.summary(mult_lm).rx("r.squared")[0])
        output.adjrsq[test] = np.array(base.summary(mult_lm).rx("adj.r.squared")[0])
        
        # Prediction
        predd = dict()
        for i, vec in enumerate(test_vec[0]):  
            name = "roi%02d"%i
            predd[name] = vec
        if type(gr) == type(y):
            predd["group"] = FV([gr[test]])
        
        preddataf = rob.DataFrame(predd)
        output.prediction[test] = stats.predict(mult_lm, preddataf)[0]
        output.pred_errors[test] = y[test] - output.prediction[test]
        #print "did prediction, error = ", output.pred_errors[test]
    return output
    
