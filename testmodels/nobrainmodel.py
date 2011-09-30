import numpy as np
import os
import sys

import sklearn.cross_val as cv

import rpy2.robjects as rob
from rpy2.robjects import FloatVector as FV
from rpy2.robjects.packages import importr

stats = importr('stats')
base = importr('base')

#for debugging: to print big arrays (and hopefully save big arrays too...):
np.set_printoptions(threshold='nan')


## INITIAL SETUP
    
# original input file with test scores etc for every subject + all amygdala activations
pdata = np.recfromcsv('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l2output/social/split_halves/regression/lsasDELTA/6mm/allsubs.csv',names=True)
# put here either lsas_delta or lsas_post
responsevar = pdata.lsas_pre - pdata.lsas_post
datadict = dict()
datadict['group'] = pdata.classtype - 2
datadict['lsas_pre'] = pdata.lsas_pre
datadict['age'] = pdata.age
datadict['sex'] = pdata.sex
subject_num = len(pdata.subject)

def make_model(formulastr2, y, explvars):
    """
    run a regression and return the model
    """
    rob.globalenv["y"] = FV(y)
    formulastr = "y ~ 1"
    if not formulastr2 == "":
        formulastr += " + "+formulastr2
        for var in explvars:
            rob.globalenv[var] = FV(explvars[var])
    # do the regression
    mult_lm = stats.lm(formulastr)
    return mult_lm
    
def predict(model, explvars):
    if not len(explvars) == 0:
        preddataf = rob.DataFrame(explvars)    
        # PREDICT SCORE
        prediction = stats.predict(model, preddataf)[0]
    else:
        prediction = model.rx("coefficients")[0][0] #if model only contains intercept, return this as prediction (mean of lsas_delta)
    return prediction

def crossval(formulastr):
    """
    runs a complete cross-validation with "y ~ 'formulastr' + 1"
    returns a vector with the predicted lsas_delta/_post scores
    """    
    predscores = []
    actualscores = []
    for trainidx, testidx in cv.LeaveOneOut(subject_num):
        # make vardicts
        vardict_train = dict()
        vardict_test = dict()
        for var in datadict:
            if var in formulastr:
                vardict_train[var] = datadict[var][trainidx]
                vardict_test[var] = datadict[var][testidx][0]
        # fit the model
        model = make_model(formulastr, responsevar[trainidx], vardict_train)
        # predict
        prediction = predict(model, vardict_test)
        predscores.append(prediction)
        actualscores.append(responsevar[testidx][0])
    """   
    predscores_beta = []
    actualscores_beta = []
    for y in xrange(len(predscores)):
        [predscores_beta.append(x) for x in predscores[y]]
        [actualscores_beta.append(x) for x in actualscores[y]]
    predscores_alpha = np.array(predscores_beta)
    actualscores_alpha = np.array(actualscores_beta)
    prederrors = predscores_alpha - actualscores_alpha
    """
    predscores = np.array(predscores)
    actualscores = np.array(actualscores)
    prederrors = predscores - actualscores
    meanerr = np.mean(np.abs(prederrors))
    rmsqerr = np.sqrt(np.mean(prederrors**2))     
    return predscores, actualscores, meanerr, rmsqerr   
   
if __name__=="__main__":
    ### CROSS-VALIDATION LOOP ###
    print "##### CROSS VALIDATION #####"
    predscores, actualscores, meanerr, rmsqerr = crossval("lsas_pre*group")
    print "mean error: %s, root mean squared error: %s"%(meanerr, rmsqerr)


