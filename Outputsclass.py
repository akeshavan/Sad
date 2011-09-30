"""
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
"""

class Outputsclass:
    def __init__(self,subject_num,y,yname):
        import numpy as np
        self.prediction = np.zeros(subject_num)
        self.actual = y
        self.name = yname
        self.pred_errors = np.zeros(subject_num)
        self.rmse = []
        self.rsq = np.zeros(subject_num)
        self.adjrsq = np.zeros(subject_num)
        self.mult_lm = []
        self.anov_brain = []
        self.anov_lsaspre = []
        self.pca = []
        self.lasso = []
    def prepare_bootstrap(self):
        self.boot_prediction = []
        self.boot_pred_errors = []
        self.boot_rsq = []
        self.boot_adjrsq = []
        self.boot_train = []
        self.boot_test = []
    def append_bootstrap(self, train,test, subject_num):
        import numpy as np
        self.boot_prediction.append(self.prediction)
        self.boot_pred_errors.append(self.pred_errors)
        self.boot_rsq.append(self.rsq)
        self.boot_adjrsq.append(self.adjrsq)
        self.boot_train.append(train)
        self.boot_test.append(test)
        #reset to 0
        self.prediction = np.zeros(subject_num)
        self.pred_errors = np.zeros(subject_num)
        self.adjrsq = np.zeros(subject_num)
        self.rsq = np.zeros(subject_num)
    def prepare_kfold(self):
        self.kfold_train = []
        self.kfold_test = []
    def append_kfold(self, train,test):
        import numpy as np
        self.kfold_train.append(train)
        self.kfold_test.append(test)
    def write_outputs(self,outdir,title):
        import os
        CGfile = open(os.path.join(outdir,(title+".txt")),"w")
        CGfile.write(title+"\n\n")
        CGfile.write("Fold#\t"+self.name+"\tprediction\tError\t\t\tR2\t\t\t\tadj.R2\n")
        self.get_rmse()
        for i in xrange(len(self.prediction)):
            CGfile.write(str(i+1)+"\t\t%2d"%(self.actual[i])+ "\t\t\t%3.4f"%(self.prediction[i])+ "\t\t%02.3f"%(self.pred_errors[i])
                            + "\t\t\t%2.4f"%(self.rsq[i])+ "\t\t\t%2.4f"%(self.adjrsq[i])+"\n")
        CGfile.write("\nRoot Mean Square Error: %2.3f\n\n"%(self.rmse))
        if self.anov_brain:
            CGfile.write("ANOVA with whole-brain model\nNOTE: roi00 = lsas_pre\n\n")
            CGfile.write(str(self.anov_brain))
            f_val = self.anov_brain.rx("F")[0][1]
            ResDF = self.anov_brain.rx("Res.Df")[0]
            p_val = 1-stats.pf(f_val,ResDF[0],ResDF[1])[0]
            CGfile.write("\np value = "+str(p_val)+"\n")
        if self.anov_lsaspre:
            CGfile.write("\n\nANOVA with lsas_pre model\nNOTE: roi00 = lsas_pre\n\n")
            CGfile.write(str(self.anov_lsaspre))
            f_val = self.anov_lsaspre.rx("F")[0][1]
            ResDF = self.anov_lsaspre.rx("Res.Df")[0]
            p_val = 1-stats.pf(f_val,ResDF[0],ResDF[1])[0]
            CGfile.write("\np value = "+str(p_val)+"\n")
        CGfile.close()
    def get_rmse(self):
        import numpy as np
        self.rmse = np.sqrt(np.mean(self.pred_errors**2))
    def draw_plots(self,savedir, con,modelname):
        """
        plot the predicted vs. the actual score with different colors for each group
        """
        import matplotlib.pyplot as plt
        import rpy2.robjects as rob
        from rpy2.robjects import FloatVector as FV
        from rpy2.robjects.packages import importr
        import numpy as np
        import os
        stats = importr('stats')
        base = importr('base')
        self.get_rmse()
        predscores, actualscores, meanerr, rmsqerr = self.prediction, self.actual, np.mean(abs(self.pred_errors)), self.rmse
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
        plt.title("Predicted vs. Actual LSAS_delta score")
        plt.axis([0,100,0,100])
        axes = plt.axes()
        axes.text(0.05,0.8,"meanerr: %.2f\nrmse: %.2f\nr: %.2f (Rsqrd: %.2f)"%(meanerr,rmsqerr,np.sqrt(rsqrd),rsqrd),transform=axes.transAxes)
        plt.legend()
        plt.savefig(os.path.join(savedir,"%s_%s_lsas_delta_act_pred.pdf"%(modelname,con)),dpi=300,format="pdf")
        


