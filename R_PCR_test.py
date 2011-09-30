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
outdir = '/mindhive/gablab/users/keshavan/sad'

from Outputsclass import Outputsclass
import regression_functions
# mult_lm = do_Regression(y,Vals,gr=0):
# output = do_R_Crossval(y,yname,files,X,gr = 0):
from LassoPCR_test import load_data
# data,idx,img = load_data(files)
from LassoPCR_test import do_PCA
# pca, data_red = do_PCA(data)

if __name__ == "__main__":
    # when calling the script, you have to specify the contrasts you want to use,
    # e.g. python doVoxMuReg.py 1 2 3 4 
    # to do contrasts 1-4
    if len(sys.argv) == 1:
        sys.exit("please call the script by giving it the contrasts as arguments")
    contrasts = [int(x) for x in sys.argv[1:]]
    
    pdata = np.recfromcsv('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l2output/social/split_halves/regression/lsasDELTA/6mm/allsubs.csv',names=True)
    
    subject_num = len(pdata.subject)
    # initialize dependent variable
    y = pdata.lsas_pre-pdata.lsas_post
    ind_variables_num = 2 #if change number here, also modify assignments below (and vice versa)
    # initialize design matrix
    X = np.zeros([subject_num,ind_variables_num])
    X[:,1] = pdata.classtype-2
    X[:,0] = pdata.lsas_pre
    # run this for all the contrasts specified when calling the script
    
    for con in contrasts:
        condir = os.path.join(outdir,'con%d'%con)
        if not os.path.exists(condir):
            os.mkdir(condir)
        for i, s in enumerate(pdata.subject):
            # the original nii file for each subject  
            fname = os.path.join('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l1output/social/norm_contrasts/_subject_id_%s/_fwhm_6/wcon_%04d_out_warped.nii'%(s,con))
            # in our conX folder in the output directory we want a symbolic link to the original nii files for each subject
            newname = os.path.join(condir,'%s_con%d.nii'%(s,con))
            if not os.path.islink(newname):
                os.symlink(fname, newname)

        confiles = sorted(glob(os.path.join(condir,'*_con%d.nii'%(con))))

        print("starting crossval")
        output = regression_functions.do_R_Crossval(y,'lsas_delta',confiles,X)
        print np.sqrt(np.mean(output.pred_errors**2))
        output.draw_plots(condir,'1','R_PCR_15comp')

