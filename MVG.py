import numpy  
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

import utils
import BayesDecision

def validate_MVG(DTR, LTR, pi, covType, gauss, m, k=5):
    h = {}
    S = numpy.zeros((len(set(LTR)), DTR.shape[1])) 
    C = 0          
    for i, j in enumerate(range(0, DTR.shape[1], DTR.shape[1]//k)):     # i is the i-th fold
        (Dtr, Ltr), (Dva, Lva) = utils.k_fold(DTR, LTR, i, k)
        if (gauss == 1):
            Dtr, Dva = utils.features_gaussianization(Dtr, Dva)               # guassianization first
        if (m < 12):
            Dtr, Dva = utils.apply_PCA(Dtr, Dva, m) 
        if 'Tied' in covType:
            for lab in [0,1]:
                h[lab] = utils.compute_empirical_moments(Dtr[:,Ltr == lab])
                if 'Tied Diag-Cov' == covType:
                    C += h[lab][1] * numpy.identity(Dtr.shape[0]) * float(Dtr[:,Ltr == lab].shape[1]/Dtr.shape[1])
                else:
                    C += h[lab][1] * float(Dtr[:,Ltr == lab].shape[1]/Dtr.shape[1]) 

        for lab in [0,1]:
            h[lab] = utils.compute_empirical_moments(Dtr[:,Ltr == lab])    
            if 'Tied' in covType:
                mu,_ = h[lab]       
            elif 'Diag-Cov' == covType:   
                mu, C = h[lab][0], h[lab][1]*numpy.identity(Dtr.shape[0])  
            elif 'Full-Cov' == covType:
                mu, C = h[lab]       
            S[lab, j: j + Dva.shape[1]] = utils.logpdf_GAU_ND_2(Dva, mu, C) 

    llr = S[1, :] - S[0, :]

    BayesMatrix = BayesDecision.compute_BayesMatrix(llr, Lva, pi)   
    _, DCF = BayesDecision.compute_BayesRisk(BayesMatrix, pi)
    minDCF = BayesDecision.compute_minimum_detection_cost(pi, llr, Lva)

    if m < 12: 
        if gauss == 1:
            print("%s\t-\t  minDCF :  %1.3f \t m : %d - Gaussianization" % (covType, minDCF, m))
        else:
            print("%s\t-\t  minDCF :  %1.3f \t m : %d" % (covType, minDCF, m))
    else:
        if gauss == 1:
            print("%s\t-\t  minDCF :  %1.3f \t Raw data - Gaussianization" % (covType, minDCF))
        else:
            print("%s\t-\t  minDCF :  %1.3f \t Raw data" % (covType, minDCF))
    return DCF, minDCF

def evaluate_MVG(DTR, LTR, DTE, LTE, pi, covType, gauss, m):
    h = {}
    S = numpy.zeros((len(set(LTR)), DTE.shape[1])) 
    C = 0          

    if (gauss == 1):
        DTR, DTE = utils.features_gaussianization(DTR, DTE)               # guassianization first
    if (m < 12):
        DTR, DTE = utils.apply_PCA(DTR, DTE, m) 
    if 'Tied' in covType:
        for lab in [0,1]:
            h[lab] = utils.compute_empirical_moments(DTR[:,LTR == lab])
            if 'Tied Diag-Cov' == covType:
                C += h[lab][1] * numpy.identity(DTR.shape[0]) * float(DTR[:,LTR == lab].shape[1]/DTR.shape[1])
            else:
                C += h[lab][1] * float(DTR[:,LTR == lab].shape[1]/DTR.shape[1]) 

    for lab in [0,1]:
        h[lab] = utils.compute_empirical_moments(DTR[:,LTR == lab])    
        if 'Tied' in covType:
            mu,_ = h[lab]       
        elif 'Diag-Cov' == covType:   
            mu, C = h[lab][0], h[lab][1]*numpy.identity(DTR.shape[0])  
        elif 'Full-Cov' == covType:
            mu, C = h[lab]       
        S[lab,:] = utils.logpdf_GAU_ND_2(DTE, mu, C) 

    llr = S[1, :] - S[0, :]
  
  
    minDCF = BayesDecision.compute_minimum_detection_cost(pi, llr, LTE)
    # print("MVG_llr_kfold \t-\t DCF : \t\t %1.3f \t  minDCF :  %1.3f \t m : %d" % (DCF, minDCF, m))
    if m < 12: 
        if gauss == 1:
            print("%s\t-\t  minDCF :  %1.3f \t m : %d - Gaussianization" % (covType, minDCF, m))
        else:
            print("%s\t-\t  minDCF :  %1.3f \t m : %d" % (covType, minDCF, m))
    else:
        if gauss == 1:
            print("%s\t-\t  minDCF :  %1.3f \t Raw data - Gaussianization" % (covType, minDCF))
        else:
            print("%s\t-\t  minDCF :  %1.3f \t Raw data" % (covType, minDCF))

