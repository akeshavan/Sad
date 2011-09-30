from glob import glob
import numpy as np
import os
import re
import sys

import nibabel as nib

from scipy.ndimage import label

import scikits.learn.cross_val as cv

import rpy2.robjects as rob
from rpy2.robjects import FloatVector as FV
from rpy2.robjects.packages import importr

#for debugging: to print big arrays (and hopefully save big arrays too...):
np.set_printoptions(threshold='nan')

stats = importr('stats')
base = importr('base')

## INITIAL SETUP
# in this folder, all sym links to con files and all the SPM output will be saved/loaded from if already exist
spmdir = '/mindhive/scratch/fhorn/model_spminp_l2o/con1'

if not os.path.isdir(spmdir):
    sys.exit("please run the clustersmodel_l2ocrossval.py script first to generate the necessary input files")
# original input file with test scores etc for every
pdata = np.recfromcsv('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l2output/social/split_halves/regression/lsasDELTA/6mm/allsubs.csv',names=True)

# put here either lsas_delta or lsas_post
responsevar = pdata.lsas_pre - pdata.lsas_post

group = pdata.classtype - 2
lsas_pre = pdata.lsas_pre
subject_num = len(pdata.subject)

def get_confiles():
    # list of all the confiles
    confiles = sorted(glob(os.path.join(spmdir,'*_con%d.nii'%(con))))
    return confiles

def get_labels(analdirs, hthr, fthr):
    """
    returns the label to the conjunction of clusters based on the spm thr images from all the analysis directories
    """
    thrdir = 'thresh_h%s_f%s'%(str(hthr)[2:], str(fthr)[2:])
    fname = os.path.join(analdirs[0],thrdir,'spmT_0001_thr.img')
    img = nib.load(fname)
    data, aff, head = img.get_data(), img.get_affine(), img.get_header()
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


def make_model(y, group, lsas_pre, clustermeans):
    """
    makes the model. assumes the first column is group, second lsas_pre, all others cluster means
    returns model
    """
    formulastr = "y ~ lsas_pre + group"
    rob.globalenv["y"] = FV(y)
    rob.globalenv["group"] = FV(group)
    rob.globalenv["lsas_pre"] = FV(lsas_pre)
    for i in xrange(clustermeans.shape[1]):
        name = "roi%02d"%(i+1)
        #formulastr += " + %s + group:%s"%(name, name)        
        formulastr += " + %s"%(name)
        rob.globalenv[name] = FV(clustermeans[:,i])

    formulastr += " + 1"
    # do the regression
    mult_lm = stats.lm(formulastr)
    return mult_lm
    
    
def predict(model, group, lsas_pre, clustermeans):
    """
    predict scores for all left out subjects
    """
    predictions = np.zeros(len(group))
    for sub in xrange(len(group)):
        # GET PREDICTION DICTIONARY & DATAFRAME
        predd = dict()
        predd['group'] = group[sub]
        predd['lsas_pre'] = lsas_pre[sub]
        for i in xrange(clustermeans.shape[1]):
            name = "roi%02d"%(i+1)
            predd[name] = clustermeans[sub,i]
        preddataf = rob.DataFrame(predd)
        
        # PREDICT SCORE
        prediction = stats.predict(model, preddataf)[0]
        predictions[sub] = prediction
    return predictions
    
def crossval(con,hthr,fthr):
    """
    runs a complete nested leave-1-out cross-validation for the given contrast
    returns a vector with the predicted lsas_delta/_post scores
    """

    confiles = get_confiles(con)
    # create the contrast folder in the outdir
    condir = os.path.join(outdir,'con%s'%con)
    if not os.path.isdir(condir):
        os.mkdir(condir)
        
    predscores = []
    actualscores = []
    num_labels = []
    run = 1
    for trainidx, testidx in cv.LeaveOneOut(subject_num):
        analoutdir = os.path.join(condir,"analysis_run%02d"%run)
        if not os.path.isdir(analoutdir):
            os.mkdir(analoutdir)
        # n-p training files
        trainconfiles = [cf for i, cf in enumerate(confiles) if trainidx[i]]
        # left out subjects to test with
        testconfiles = [cf for i, cf in enumerate(confiles) if testidx[i]]
        _, name = os.path.split(testconfiles[0])
        sid = name.split('con')[0][:-1]
        # sidx is the row# of the sid in our pdata variable
        sidx = np.nonzero(pdata.subject == sid)[0][0]
        analysisdirs = []
        for idx in range(subject_num):
            if not idx == sidx:
                left_out = [sidx, idx]
                left_out.sort()
                analysisdirs.append(os.path.join(spmdir,"con%s"%(con),'analysis_lo_%02d_%02d'%(left_out[0],left_out[1])))

        # fit the model
        labels, nlabels = get_labels(analoutdir, analysisdirs, hthr, fthr)
        num_labels.append(nlabels)
        clustermeans = get_clustermeans(labels, nlabels, trainconfiles)
        model = make_model(responsevar[trainidx], group[trainidx], lsas_pre[trainidx], clustermeans)
        # predict
        clustermeans = get_clustermeans(labels, nlabels, testconfiles)
        prediction = predict(model, group[testidx], lsas_pre[testidx], clustermeans)
        predscores.append(list(prediction))
        actualscores.append(list(responsevar[testidx]))
        run += 1
    predscores_beta = []
    actualscores_beta = []
    for y in xrange(len(predscores)):
        [predscores_beta.append(x) for x in predscores[y]]
        [actualscores_beta.append(x) for x in actualscores[y]]
    predscores_alpha = np.array(predscores_beta)
    actualscores_alpha = np.array(actualscores_beta)
    prederrors = predscores_alpha - actualscores_alpha
    meanerr = np.mean(np.abs(prederrors))
    rmsqerr = np.sqrt(np.mean(prederrors**2))
    # save stuff
    predfile = open(os.path.join(condir,"crossval_output_hthr%s_fthr%s_%s_runs.txt"%(hthr,fthr,run-1)),"w")
    predfile.write("ACTUAL LSAS_DELTA SCORE\n"+str(actualscores)+"\nPREDICTED LSAS_DELTA SCORE\n"+str(predscores)+
                    "\nPREDICTION ERRORS:\n"+str(prederrors)+"\nMEAN PREDICTION ERROR:\n"+str(meanerr)
                   +"\nROOT MEAN SQUARED ERROR:\n"+ str(rmsqerr)+"\n\nNUMBER OF LABELS IN EACH RUN:\n"+str(num_labels))
    predfile.close()
    print num_labels       
    return predscores_alpha, actualscores_alpha, meanerr, rmsqerr
    
def final_model(con, hthr, fthr):
    # create the contrast folder in the outdir
    condir = os.path.join(outdir,'con%s'%con)
    if not os.path.isdir(condir):
        os.mkdir(condir)
    analysisdir = os.path.join(condir,'analysis_finalmodel')
    if not os.path.isdir(analysisdir):
        os.mkdir(analysisdir)     
    
    confiles = get_confiles(con)
    # fit the model
    analdirs = [os.path.join(spmdir,"con%s"%(con),'analysis_run%02d'%run) for run in range(1,subject_num+1)]
    labels, nlabels = get_labels(analysisdir, analdirs, hthr, fthr)
    clustermeans = get_clustermeans(labels, nlabels, confiles)
    model = make_model(responsevar, group, lsas_pre, clustermeans)
    return model
    
def final_pred(con, hthr, fthr):
    """
    uses the final model to predict the subjects lsas_delta scores
    """    
    finalmodel = final_model(con, hthr, fthr)
    condir = os.path.join(outdir,'con%s'%con)
    analysisdir = os.path.join(condir,'analysis_finalmodel')
    confiles = get_confiles(con)
    analdirs = [os.path.join(spmdir,"con%s"%(con),'analysis_run%02d'%run) for run in range(1,subject_num+1)]
    labels, nlabels = get_labels(analysisdir, analdirs, hthr, fthr)
    clustermeans = get_clustermeans(labels, nlabels, confiles)
    predscores = predict(finalmodel, group, lsas_pre, clustermeans)
    prederrors = np.array(predscores) - np.array(responsevar)
    meanerr = np.mean(np.abs(prederrors))
    rmsqerr = np.sqrt(np.mean(prederrors**2))
    return predscores, responsevar, meanerr, rmsqerr
   
if __name__=="__main__":
    # when calling the script, you have to specify the contrasts you want to use,
    # e.g. python clustermodel.py 1 2 3 4 
    # to do contrasts 1-4
    if len(sys.argv) == 1:
        sys.exit("please call the script by giving it the contrasts as arguments")
    contrasts = [int(x) for x in sys.argv[1:]]
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
    # RUN FOR ALL CONTRASTS
    for con in contrasts: 
        condir = os.path.join(outdir,'con%s'%con)
        if not os.path.isdir(condir):
            os.mkdir(condir)
        print "############### CONTRAST %s ###############"%con  
        ### CROSS-VALIDATION LOOP ###
        for hthr in [0.001, 0.01, 0.05]:
            for fthr in [0.001, 0.01, 0.05]:
                print "######## FOR THRESHOLDS: height: %s and fdr: %s"%(hthr, fthr)
                predscores, actualscores, meanerr, rmsqerr = crossval(con, hthr, fthr)
                print "mean error: %s, root mean squared error: %s"%(meanerr, rmsqerr)
                finalmodel = final_model(con, hthr, fthr)
                modelsum = base.summary(finalmodel)
                rsq = modelsum.rx("r.squared")[0][0]
                adjrsq = modelsum.rx("adj.r.squared")[0][0]
                print "### FINAL MODEL: Multiple R-Squared: %s  Adjusted R-Squared: %s"%(rsq, adjrsq)
                # save stuff
                outfile = open(os.path.join(condir,"modelsummary_hthr%s_fthr%s.txt"%(hthr, fthr)),"w")
                outfile.write(str(modelsum))
                outfile.close() 






        

