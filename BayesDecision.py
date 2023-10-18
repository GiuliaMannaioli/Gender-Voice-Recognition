import numpy  
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
import sys


def compute_confmatrix(LTE_P, LTE):
    Conf = numpy.zeros((len(set(LTE)), len(set(LTE))), dtype = int)
    for i in set(LTE):
        for j in set(LTE):
            Conf[i, j] = ((LTE_P == i) * (LTE == j)).sum() 
    return Conf

def compute_BayesMatrix(llr, LTE, pi, Cfn=1, Cfp=1):    
    t = - numpy.log((pi*Cfn)/((1-pi)*Cfp))     
    LTE_P = llr > t                             
    Conf = compute_confmatrix(LTE_P, LTE)
    return Conf

def compute_BayesRisk(BayesMatrix, pi, Cfn=1, Cfp=1):
    FNR = BayesMatrix[0, 1] / (BayesMatrix[0, 1] + BayesMatrix[1, 1])
    FPR = BayesMatrix[1, 0] / (BayesMatrix[1, 0] + BayesMatrix[0, 0])
    DCFu = pi*Cfn*FNR + (1-pi)*Cfp*FPR
    DCF = DCFu/numpy.min([pi*Cfn, (1-pi)*Cfp])
    return DCFu, DCF

def compute_minimum_detection_cost(pi, llr, LTE):
    thresholds = numpy.array(llr)        
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    minDCFs = numpy.zeros(thresholds.size)
    for i,t in enumerate(thresholds):
       LTE_P = numpy.int32(llr > t )
       Conf = compute_confmatrix(LTE_P, LTE)
       _, minDCFs[i] = compute_BayesRisk(Conf, pi)
    minDCF = numpy.min(minDCFs)
    return minDCF

def plot_ROC(llr, LTE):
    thresholds = numpy.array(llr)      
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    FNR = numpy.zeros(thresholds.size)
    FPR = numpy.zeros(thresholds.size)
    for i,t in enumerate(thresholds):
       LTE_P = numpy.int32(llr > t )
       Conf = compute_confmatrix(LTE_P, LTE)
       FNR[i] = Conf[0, 1] / (Conf[0, 1] + Conf[1, 1])
       FPR[i] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0])
    TPR = 1 - FNR
    plt.plot(FPR, TPR)
    plt.xlabel('FPR', fontsize=8)
    plt.ylabel('TPR', fontsize=8)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(8)
    plt.show()

def plotBayes_error(llr, LTE):
    effPriorLogOdds = numpy.linspace(-4, 4, 101)
    effPrior = 1 / (1 + numpy.exp(-effPriorLogOdds))
    minDCF, DCF = numpy.zeros(effPrior.size), numpy.zeros(effPrior.size)
    for i, effp in enumerate(effPrior):
        BayesMatrix = compute_BayesMatrix(llr, LTE, effp)
        _, DCF[i] = compute_BayesRisk(BayesMatrix, effp)
        minDCF[i] = compute_minimum_detection_cost(effp, llr, LTE)
    plt.plot(effPriorLogOdds, DCF, label='GMM - act DCF', color='r', linewidth=1.5)
    plt.plot(effPriorLogOdds, minDCF, label='GMM - min DCF', color='b', linewidth=1.5)
    # plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('', fontsize=8)
    plt.ylabel('DCF', fontsize=8)
    plt.legend(loc='lower left', prop={"size":8})
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(8)
    plt.show()
    return DCF, minDCF



