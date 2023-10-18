import numpy  
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

import utils 
import MVG
import SVM
import logreg
import BayesDecision
import GMM


def plotting_features(DTR, LTR):
    utils.plot_features(DTR, LTR)
    DTRg, _ = utils.features_gaussianization(DTR, DTE)
    utils.plot_features(DTRg, LTR)
    utils.plot_correlation_heat_map(DTR, LTR)

def plotting_data(DTR, LTR):
    DTR2, _ = utils.apply_PCA(DTR, LTR, 2)
    utils.plot_scatter(DTR2, LTR)
    DTR3, _ = utils.apply_PCA(DTR, LTR, 3)
    utils.plot_scatter3D(DTR3, LTR)

def MVG_training(DTR, LTR):
    covType = ['Full-Cov', 'Diag-Cov', 'Tied Full-Cov', 'Tied Diag-Cov']
    for g in [0, 1]:
        for pi in [.1, .5, .9]:
            for m in range(7, 13):
                if m == 7:
                    print("\t\t\t   \u03C0 : %1.1f \t\t k : %d" % (pi, 5))
                for Ct in covType:
                    MVG.validate_MVG(DTR, LTR, pi, Ct, g, m)
                print()

def Linear_LogReg_training(DTR, LTR):
    logreg.plot_logreg_lambda(DTR, LTR)
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f \t\t k : %d" % (pi, 5))
                logreg.validate_linear_logreg(1e-5, DTR, LTR, pi, 0, m)

def Quad_LogReg_training(DTR, LTR):
    logreg.plot_Quad_logreg_lambda(DTR, LTR)
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f \t\t k : %d" % (pi, 5))
                logreg.validate_Quad_logreg(1e-5, DTR, LTR, pi, 0, m)
    
def Linear_SVM_training(DTR, LTR):
    SVM.plot_linearSVM_C(DTR, LTR)
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f \t\t k : %d" % (pi, 5))
            SVM.validate_linearSVM(1e-1, DTR, LTR, pi, 0, m)

def Quad_SVM_training(DTR, LTR):
    SVM.plot_QuadSVM_C(DTR, LTR)
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f \t\t k : %d" % (pi, 5))
            SVM.validate_kernelSVM(0.00188739, 0, DTR, LTR, pi, 0, 'Quad', m)

def RBF_SVM_training(DTR, LTR):
    SVM.plot_rbfSVM_C(DTR, LTR)
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f \t\t k : %d" % (pi, 5))
            SVM.validate_kernelSVM(1, 1e-3, DTR, LTR, pi, 0, 'RBF', m)

def GMM_training(DTR, LTR):
    GMM.plot_GMM(DTR, LTR)
    covType = ['Full-Cov', 'Diag-Cov', 'Tied Full-Cov', 'Tied Diag-Cov']
    NG = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for pi in [.1, .5, .9]:
        print("\t\t\t   \u03C0 : %1.1f \t\t k : %d" % (pi, 5))
        for Ct in covType:
            for ng in NG: 
                GMM.validate_GMM(DTR, LTR, Ct, ng, pi, 12, 0)

def Score_calibration(llr, Lva):
    S, Lva = utils.score_calibarion(llr, Lva, 0.5)
    BayesDecision.plotBayes_error(S, Lva)

def BayesPlot(DTR, LTR):
    _, llr, Lva = GMM.validate_GMM(DTR, LTR, 'Tied Full-Cov', 512, 0.5, 12, 0)
    BayesDecision.plotBayes_error(llr, Lva)
    return llr, Lva

def MVG_evaluation(DTR, LTR, DTE, LTE):
    covType = ['Full-Cov', 'Diag-Cov', 'Tied Full-Cov', 'Tied Diag-Cov']
    for g in [0, 1]:
        for pi in [.1, .5, .9]:
            for m in range(8, 13):
                if m == 8:
                    print("\t\t\t   \u03C0 : %1.1f" % (pi))
                for Ct in covType:
                    MVG.evaluate_MVG(DTR, LTR, DTE, LTE, pi, Ct, g, m)
                print()

def Linear_LogReg_evaluation(DTR, LTR, DTE, LTE):
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f" % (pi))
            logreg.evaluate_logreg(DTR, LTR, DTE, LTE, pi, 0, m)

def Quad_LogReg_evaluation(DTR, LTR, DTE, LTE):
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f" % (pi))
            logreg.evaluate_Quad_logreg(DTR, LTR, DTE, LTE, pi, 0, m)

def Linear_SVM_evaluation(DTR, LTR, DTE, LTE):
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f" % (pi))
            SVM.evaluate_linearSVM(1e-1, DTR, LTR, DTE, LTE, pi, 0, m)
    
def Quad_SVM_evaluation(DTR, LTR, DTE, LTE):
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f" % (pi))
            SVM.evaluate_kernelSVM(0.00188739, 0, DTR, LTR, DTE, LTE, pi, 0, 'Quad', m)

def RBF_SVM_evaluation(DTR, LTR, DTE, LTE):
    for pi in [.1, .5, .9]:
        for m in [8, 12]:
            if m == 8:
                print("\t\t\t   \u03C0 : %1.1f" % (pi))
            SVM.evaluate_kernelSVM(1, 1e-3, DTR, LTR, DTE, LTE, pi, 0, 'RBF', m)

def GMM_evaluation(DTR, LTR, DTE, LTE):
    covType = ['Full-Cov', 'Diag-Cov', 'Tied Full-Cov', 'Tied Diag-Cov']
    NG = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for pi in [.1, .5, .9]:
        print("\t\t\t   \u03C0 : %1.1f" % (pi))
        for Ct in covType:
            for ng in NG: 
                GMM.evaluate_GMM(DTR, LTR, DTE, LTE, Ct, ng, pi, 12, 0)

if __name__ == '__main__':
    DTR, LTR = utils.loadTrain('GenderRecognition/Train.txt')       
    DTE, LTE = utils.loadTest('GenderRecognition/Test.txt')

    '''Features analyzing'''
    plotting_features(DTR, LTR)
    plotting_data(DTR, LTR)

    '''Training phase'''
    MVG_training(DTR, LTR)
    Linear_LogReg_training(DTR, LTR)
    Quad_LogReg_training(DTR, LTR)
    Linear_SVM_training(DTR, LTR)
    Quad_SVM_training(DTR, LTR)
    RBF_SVM_training(DTR, LTR)
    GMM_training(DTR, LTR)

    '''Bayes error plot of the best model'''
    llr, Lva = BayesPlot(DTR, LTR)

    '''Score calibration and plotting'''
    Score_calibration(llr, Lva)

    '''Evaluation phase'''
    MVG_evaluation(DTR, LTR, DTE, LTE)
    Linear_LogReg_evaluation(DTR, LTR, DTE, LTE)
    Quad_LogReg_evaluation(DTR, LTR, DTE, LTE)
    Linear_SVM_evaluation(DTR, LTR, DTE, LTE)
    Quad_SVM_evaluation(DTR, LTR, DTE, LTE)
    RBF_SVM_evaluation(DTR, LTR, DTE, LTE)
    GMM_evaluation(DTR, LTR, DTE, LTE)



   

                    

            