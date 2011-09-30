from nipy.labs import viz
import nibabel as nib
import os
import sys
import numpy as np
from glob import glob
import pylab
import matplotlib.pyplot as plt

import rpy2.robjects as rob
from rpy2.robjects import FloatVector as FV
from rpy2.robjects.packages import importr

stats = importr('stats')
base = importr('base')

import nobrainmodel

np.set_printoptions(threshold='nan')

outdir = '/mindhive/gablab/u/fhorn/testmodels/figures_scatterpreds'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

def actvspred(modelname, predmodel):
    """
    plot the predicted vs. the actual score
    """
    predscores, actualscores, meanerr, rmsqerr = predmodel
    plt.figure()
    plt.scatter(actualscores,predscores,s=70)
    x = np.array(range(100))
    plt.plot(x,x,'g',label='optimal model')
    # make regression
    rob.globalenv["pred"] = FV(predscores)
    rob.globalenv["act"] = FV(actualscores)
    mult_lm = stats.lm("pred ~ act + 1")
    coeffs = np.array(mult_lm.rx("coefficients")[0])
    rsqrd = base.summary(mult_lm).rx("r.squared")[0][0]
    y = coeffs[1]*x+coeffs[0]
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


if __name__=="__main__":
    # actual vs. predicted lsas_delta
    """
    for hthr in [0.001, 0.01, 0.05]:
        for fthr in [0.001, 0.01, 0.05]:
            actvspred("conjunclusters_h%s_f%s"%(str(hthr)[2:], str(fthr)[2:]),conjunclustersmodel.crossval(1,hthr,fthr))
    """
    actvspred("1",nobrainmodel.crossval(""))
    actvspred("lsas_pre",nobrainmodel.crossval("lsas_pre"))
    actvspred("group",nobrainmodel.crossval("group"))
    actvspred("age",nobrainmodel.crossval("age"))
    actvspred("sex",nobrainmodel.crossval("sex"))

