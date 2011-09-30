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
 
outdir = '/mindhive/gablab/users/keshavan/sad'
#for debugging: to print big arrays:
np.set_printoptions(threshold='nan')

def load_data(files):
    """Load data from a list of files coregistered to each other and return a 2d matrix [subjects x voxels]
    """
    data = None
    for fname in files:
        if data is None:
            data = nib.load(fname).get_data().ravel()
        else:
            data = np.vstack((data, nib.load(fname).get_data().ravel()))
    img = nib.load(fname)
    idx = np.isfinite(np.sum(data,axis=0))
    data = data[:,idx]
    return data,idx,img

def do_PCA(data):
    pca = PCA(n_components = 15)
    pca.fit(data)
    data_red = pca.transform(data)
    return pca, data_red
    
def do_LASSO(y,X):
    lasso = linear_model.LassoLarsCV()
    lasso.fit(X,y)
    return lasso


def do_Lasso_Kfold(y,yname,files,X):
    subject_num = y.shape[0]
    output = Outputsclass(subject_num,y,yname)
    output.prepare_kfold()
    data, idx, img  = load_data(files)
    
    for train, test in cv.StratifiedKFold(np.zeros(subject_num), k = 4):
        # PCA
        pca, data_red = do_PCA(data[train])
        output.pca.append(pca)
        data_red_test = pca.transform(data[test])
        # Build design matrix & test vector
        desmat_cv = np.hstack((data_red,X[train]))
        desmat_cv = np.array(desmat_cv)
        y_cv = y[train]
        test_vec = np.hstack((data_red_test,X[test]))
        test_vec = np.array(test_vec)
        #Lasso
        lasso = do_LASSO(y_cv,desmat_cv)
        output.lasso.append(lasso)
        output.rsq[test] = lasso.score(desmat_cv,y_cv)
        output.adjrsq[test] = 1 - (1 - output.rsq[test])*(subject_num-1-1)/(subject_num-1 - lasso.coef_.shape[0] -1)
        # Prediction
        output.prediction[test] = lasso.predict(test_vec)
        output.pred_errors[test] = y[test] - output.prediction[test]
        print "did prediction, error = ", output.pred_errors[test]
        output.append_kfold(train,test)
        
    return output


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
    ind_variables_num = 3 #if change number here, also modify assignments below (and vice versa)
    # initialize design matrix
    X = np.zeros([subject_num,ind_variables_num])
    X[:,1] = pdata.classtype-2
    X[:,0] = pdata.lsas_pre
    X[:,2] = pdata.age
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
        """
        data, idx, img = load_data(confiles)
        pca, data_red = do_PCA(data)
        desmat = np.hstack((data_red,X))
        lasso = do_LASSO(y,desmat)
        predictions = lasso.predict(desmat)
        print prediction
        coeff = lasso.coef_
        coeff = coeff[0:len(coeff)-2]
        weighted = pca.inverse_transform(coeff)
        brain = np.zeros(img.shape).ravel()
        #brain[:] = np.NAN
        datax = 0
        for idxx, mask in enumerate(idx):
            if mask:
                brain[idxx] = weighted[datax]
                datax += 1
                
        brain = brain.reshape(img.shape)
        imagefiles = []
        imagefiles.append(nib.Nifti1Image(brain,img.get_affine()))
        imagefiles[0].to_filename(os.path.join(condir,'weighted_image.nii' ))
        
        """
        output_lasso = do_Lasso_Kfold(y,'lsas_delta',confiles,X)
        output_lasso.write_outputs(condir, 'lassoPCRkfold')
        output_lasso.draw_plots(condir,str(con),'lassoPCRkfold_age_15')
        
        """
        eigenvalues = pca.explained_variance_ratio_
        plt.figure()
        plt.plot(eigenvalues,'b*')
        plt.xlabel("eigenvector #")
        plt.ylabel("explained variance ratio")
        plt.title("Explained variance ratio")
        plt.savefig(os.path.join(condir,"con%s_eigenvalues.pdf"%con),dpi=300,format="pdf")
        """
    



