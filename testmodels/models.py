import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from glob import glob
import sklearn.cross_validation as cv
from scipy.ndimage import label
import nibabel as nib

import rpy2.robjects as rob
from rpy2.robjects import FloatVector as FV
from rpy2.robjects.packages import importr

stats = importr('stats')
base = importr('base')

#for debugging: to print big arrays (and hopefully save big arrays too...):
np.set_printoptions(threshold='nan')

## INITIAL SETUP
outdir = '/mindhive/gablab/u/fhorn/testmodels/figures_scatterpreds'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
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

# in this folder, all sym links to con files and all the SPM output will be saved/loaded from if already exist
spmdir = '/mindhive/scratch/fhorn/model_spminp_l2o/con1'
if not os.path.isdir(spmdir):
    sys.exit("please run the clustersmodel_l2ocrossval.py script first to generate the necessary input files") 
    
confiles = sorted(glob(os.path.join(spmdir,'*_con1.nii')))

def get_labels(analdirs, hthr=0.01, fthr=0.05):
    """
    returns the label to the conjunction of clusters based on the spm thr images from all the analysis directories
    """
    thrdir = 'thresh_h%s_f%s'%(str(hthr)[2:], str(fthr)[2:])
    fname = os.path.join(analdirs[0],thrdir,'spmT_0001_thr.img')
    img = nib.load(fname)
    data = img.get_data()
    conjunc = np.ones(data.shape)
    for analdir in analdirs:  
        # GET CLUSTERS + LABEL THEM
        fname = os.path.join(analdir,thrdir,'spmT_0001_thr.img')
        img = nib.load(fname)
        data = img.get_data()
        idx = np.where(data == 0)
        conjunc[idx] = 0
    labels, nlabels = label(conjunc)    
    return labels, nlabels
    
def get_clustermeans(labels, nlabels, confiles):
    """
    finds the cluster means of each of the nlabels clusters for each subject in the confiles
    """
    clustermeans = np.zeros([len(confiles), nlabels])
    for sub, conf in enumerate(confiles):
        data = nib.load(conf).get_data()
        mean_val = np.zeros(nlabels)

        for clusteridx in range(1,nlabels+1):
            #for each cluster, find the average value of voxel intensity from data
            idx = np.where(labels == clusteridx)
            mean_val[clusteridx-1] = np.mean(data[idx])

        clustermeans[sub,:] = mean_val
    return clustermeans

"""
# this uses the clusters determined for the final model, i.e. the conjunction of clusters from a leave1out crossval on all the data
analdirs = [os.path.join(spmdir,'analysis_run%02d'%run) for run in range(1,subject_num+1)]
labels, nlabels = get_labels(analdirs, hthr, fthr)
clustermeans = get_clustermeans(labels, nlabels, confiles)
# append clustermeans to datadict
for r in range(nlabels):
    datadict['roi%02d'%(r+1)] = clustermeans[:,r]
"""
    
#### DONE WITH SETUP #####

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
        prediction = model.rx("coefficients")[0][0] #if model only contains intercept, return that as prediction (= mean of lsas_delta)
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
    predscores = np.array(predscores)
    actualscores = np.array(actualscores)
    prederrors = predscores - actualscores
    meanerr = np.mean(np.abs(prederrors))
    rmsqerr = np.sqrt(np.mean(prederrors**2))     
    return predscores, actualscores, meanerr, rmsqerr
    
def final_model(formulastr):
    vardict = dict()
    for var in datadict:
        if var in formulastr:
            vardict[var] = datadict[var]
    model = make_model(formulastr, responsevar, vardict)
    print base.summary(model)

def actvspred(modelname, predmodel):
    """
    plot the predicted vs. the actual score
    """
    predscores, actualscores, meanerr, rmsqerr = predmodel
    # make regression
    rob.globalenv["pred"] = FV(predscores)
    rob.globalenv["act"] = FV(actualscores)
    mult_lm = stats.lm("pred ~ act + 1")
    coeffs = np.array(mult_lm.rx("coefficients")[0])
    rsqrd = base.summary(mult_lm).rx("r.squared")[0][0]
    x = np.array(range(100))
    y = coeffs[1]*x+coeffs[0]
    plt.figure()
    plt.scatter(actualscores,predscores,s=70)
    plt.plot(x,x,'g',label='optimal model')  
    plt.plot(x,y,'k',label='our model',linewidth=2)
    plt.xlabel("actual lsas delta")
    plt.ylabel("predicted lsas delta")
    plt.title(modelname)
    plt.axis([0,100,0,100])
    axes = plt.axes()
    axes.grid(b=True)
    axes.text(0.05,0.8,"meanerr: %.2f\nrmse: %.2f\nr: %.2f (Rsqrd: %.2f)"%(meanerr,rmsqerr,np.sqrt(rsqrd),rsqrd),transform=axes.transAxes)
    #plt.legend()
    plt.savefig(os.path.join(outdir,"%s_crossval.png"%modelname),dpi=300,format="png")
    return rmsqerr
    
def determine_model(braindict=None):
    num_vars = len(datadict) # how many variables the final model should have
    error = actvspred("1",crossval(""))
    errors = np.zeros(num_vars+1)
    errors[0] = error
    formulastr = ""
    formulastr_final = formulastr
    for i in range(1, num_vars+1):
        # find the model that has the lowest error
        for var in datadict:
            if not var in formulastr_final:
                if formulastr_final == "":
                    formulastr_temp = var
                else:
                    formulastr_temp = formulastr_final + " + " + var
                _, _, _, error_temp = crossval(formulastr_temp)
                if error_temp < error:
                    error = error_temp
                    formulastr = formulastr_temp
        # update the formulastr of the final model with the best variable
        formulastr_final = formulastr
        errors[i] = error
    # save error trajectory figure
    x = np.array(range(num_vars+1))
    plt.figure()
    plt.plot(x,errors)
    plt.xlabel("numbers of variables included")
    plt.ylabel("RMSE")
    plt.title(formulastr_final)
    plt.savefig(os.path.join(outdir,"errortrajectory_crossval.png"),dpi=300,format="png")
   
if __name__=="__main__":
    # plot actual vs. predicted lsas_delta for different models
    # and find the optimal model (?)
    num_vars = len(datadict) # how many variables the final model should have
    error = actvspred("1",crossval(""))
    errors = np.zeros(num_vars+1)
    errors[0] = error
    formulastr = ""
    formulastr_final = formulastr
    for i in range(1, num_vars+1):
        # find the model that has the lowest error
        for var in datadict:
            if not var in formulastr_final:
                if formulastr_final == "":
                    formulastr_temp = var
                else:
                    formulastr_temp = formulastr_final + " + " + var
                error_temp = actvspred(formulastr_temp,crossval(formulastr_temp))
                if error_temp < error:
                    error = error_temp
                    formulastr = formulastr_temp
        # update the formulastr of the final model with the best variable
        formulastr_final = formulastr
        errors[i] = error
    final_model(formulastr_final)
    print "Best Model determined: %s\n with RMSError: %s"%(formulastr_final, error)
    # plot error trajectory
    x = np.array(range(num_vars+1))
    plt.figure()
    plt.plot(x,errors)
    plt.xlabel("numbers of variables included")
    plt.ylabel("RMSE")
    plt.title(formulastr_final)
    plt.savefig(os.path.join(outdir,"errortrajectory_crossval.png"),dpi=300,format="png")
