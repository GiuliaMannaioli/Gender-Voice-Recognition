import numpy  
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

import utils
import BayesDecision


def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        w, b = utils.vcol(v[:-1]), v[-1]                
        J = l/2*numpy.dot(w.T, w) + numpy.sum(numpy.logaddexp(0, -(2*LTR - 1)*(numpy.dot(w.T,DTR) + b)))/DTR.shape[1]
        return J
    return logreg_obj

def logreg_obj_wrap_balancing(DTR, LTR, l, pi):
    def logreg_obj(v):
        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1                                                 
        Z[LTR == 0] = -1
        w, b = utils.vcol(v[:-1]), v[-1]  
        s =  numpy.dot(w.T,DTR) + b         
        J = l/2*numpy.dot(w.T, w) + (numpy.logaddexp(0, -s[:,LTR == 1]*Z[LTR == 1])).mean()*(pi) + (numpy.logaddexp(0, -s[:,LTR == 0]*Z[LTR == 0])).mean()*(1-pi)
        return J
    return logreg_obj

def expanded_features_space(DTR, DTE):
    Dtr_exp = []
    for i in range(DTR.shape[1]):
            x = DTR[:,i:i+1]    
            xxt = utils.vcol(numpy.dot(x, x.T).flatten(order='F'))
            x_exp = numpy.vstack([xxt, x])
            Dtr_exp.append(x_exp)
    Dtr_exp = numpy.hstack(Dtr_exp)
    
    Dte_exp = []
    for i in range(DTE.shape[1]):
            x = DTE[:,i:i+1]    
            xxt = utils.vcol(numpy.dot(x, x.T).flatten(order='F'))
            x_exp = numpy.vstack([xxt, x])
            Dte_exp.append(x_exp)
    Dte_exp = numpy.hstack(Dte_exp)
    return Dtr_exp, Dte_exp 

def compute_scoreMatrix_logreg(w, b, DTE):
    S = numpy.dot(w.T, DTE) + b
    return S.ravel()

def validate_linear_logreg(l, DTR, LTR, pi, gauss, m, k=5):
    S = numpy.zeros((DTR.shape[1]))
    for i, j in enumerate(range(0, DTR.shape[1], DTR.shape[1]//k)):   
        (Dtr, Ltr), (Dva, Lva) = utils.k_fold(DTR, LTR, i, k)
        if gauss == 1:
            Dtr, Dva = utils.features_gaussianization(Dtr, Dva) 
        if m < 12:
            Dtr, Dva = utils.apply_PCA(Dtr, Dva, m) 

        v = numpy.zeros((Dtr.shape[0]+1))   
        logreg_obj = logreg_obj_wrap(Dtr, Ltr, l)

        (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v, approx_grad = True, factr=1.0, maxfun=50000)
        w, b = utils.vcol(x[:-1]), x[-1]

        S[j: j + Dva.shape[1]] = compute_scoreMatrix_logreg(w, b, Dva)
 
    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S, Lva)
    if m < 12: 
        print("Log Reg \t-\t minDCF :  %1.3f (\u03BB = %5.5f) \t m : %d" % (minDCF, l, m))
    else:
        print("Log Reg \t-\t minDCF :  %1.3f (\u03BB = %5.5f) \t Raw data" % (minDCF, l))
    return minDCF

def plot_logreg_lambda(DTR, LTR):
    nlambda = 50
    minDCF_pi = numpy.zeros((3, nlambda))
    lambdas = numpy.logspace(-5, 5, num=nlambda, endpoint=True)
    for i, pi in enumerate([.1, .5, .9]):
        for j, l in enumerate(lambdas):
            minDCF_pi[i, j] = validate_linear_logreg(l, DTR, LTR, pi, 0, 12)

    x1 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3, 1e4, 1e5]
    plt.plot(lambdas, minDCF_pi[0,:], label="minDCF($\widetilde{\pi}$ = 0.1)", color='blue', linewidth=1)
    plt.plot(lambdas, minDCF_pi[1,:], label="minDCF($\widetilde{\pi}$ = 0.5)", color='red', linewidth=1)
    plt.plot(lambdas, minDCF_pi[2,:], label="minDCF($\widetilde{\pi}$ = 0.9)", color='green', linewidth=1)
    plt.xscale("log")
    plt.xlim([1e-5, 1e5])
    plt.gca().xaxis.set_ticks(x1)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(8)
    plt.xlabel('\u03BB', fontsize=8)
    plt.ylabel('DCF', fontsize=8)
    plt.legend(prop={"size":8})
    plt.show()

def logreg_on_score(DTR, LTR, pi, k=5):
    l = 1e-5
    S = numpy.zeros((DTR.shape[1]))
    for i, j in enumerate(range(0, DTR.shape[1], DTR.shape[1]//k)):   
        (Dtr, Ltr), (Dva, Lva) = utils.k_fold(DTR, LTR, i, k)

        v = numpy.zeros((Dtr.shape[0]+1))   
        logreg_obj = logreg_obj_wrap_balancing(Dtr, Ltr, l, pi)

        (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v, approx_grad = True, factr=1.0, maxfun=50000)
        w, b = utils.vcol(x[:-1]), x[-1]

        S[j: j + Dva.shape[1]] = compute_scoreMatrix_logreg(w, b, Dva)

    BayesMatrix = BayesDecision.compute_BayesMatrix(S, Lva, pi)  
    _, DCF = BayesDecision.compute_BayesRisk(BayesMatrix, pi)
    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S, Lva)

    print("Log Reg on score \t-\tminDCF : %1.3f \tDCF : %1.3f" % (minDCF, DCF))
    return S, Lva

def validate_Quad_logreg(l, DTR, LTR, pi, gauss, m, k=5):
    # l = 1e-5            best parameter founded
    
    S = numpy.zeros((DTR.shape[1]))
    for i, j in enumerate(range(0, DTR.shape[1], DTR.shape[1]//k)):   
        (Dtr, Ltr), (Dva, Lva) = utils.k_fold(DTR, LTR, i, k)
        Dtr, Dva = expanded_features_space(Dtr, Dva)

        if gauss == 1:
            Dtr, Dva = utils.features_gaussianization(Dtr, Dva) 
        if m < 12:
            Dtr, Dva = utils.apply_PCA(Dtr, Dva, m) 

        v = numpy.zeros((Dtr.shape[0]+1))   
        logreg_obj = logreg_obj_wrap(Dtr, Ltr, l)

        (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v, approx_grad = True, factr=1.0, maxfun=50000)
        w, b = utils.vcol(x[:-1]), x[-1]
        
        S[j: j + Dva.shape[1]] = compute_scoreMatrix_logreg(w, b, Dva)
 
    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S, Lva)


    if m < 12: 
        print("Quad Reg \t-\t minDCF :  %1.3f (\u03BB = %5.5f) \t m : %d" % (minDCF, l, m))
    else:
        print("Quad Reg \t-\t minDCF :  %1.3f (\u03BB = %5.5f) \t Raw data" % (minDCF, l))
    return minDCF

def plot_Quad_logreg_lambda(DTR, LTR):
    nlambda = 30
    minDCF_pi = numpy.zeros((3, nlambda))
    lambdas = numpy.logspace(-5, 5, num=nlambda, endpoint=True)
    for i, pi in enumerate([.1, .5, .9]):
        for j, l in enumerate(lambdas):
            minDCF_pi[i, j]= validate_Quad_logreg(l, DTR, LTR, pi, 0, 12)

    x1 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3, 1e4, 1e5]
    plt.plot(lambdas, minDCF_pi[0,:], label="minDCF($\widetilde{\pi}$ = 0.1)", color='blue', linewidth=1)
    plt.plot(lambdas, minDCF_pi[1,:], label="minDCF($\widetilde{\pi}$ = 0.5)", color='red', linewidth=1)
    plt.plot(lambdas, minDCF_pi[2,:], label="minDCF($\widetilde{\pi}$ = 0.9)", color='green', linewidth=1)
    plt.xscale("log")
    plt.xlim([1e-5, 1e5])
    plt.gca().xaxis.set_ticks(x1)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(8)
    plt.xlabel('\u03BB', fontsize=8)
    plt.ylabel('DCF', fontsize=8)
    plt.legend(prop={"size":8})
    plt.show()

def evaluate_logreg(DTR, LTR, DTE, LTE, pi, gauss, m):
    l = 1e-5         
    S = numpy.zeros((DTR.shape[1]))
    if gauss == 1:
        DTR, DTE = utils.features_gaussianization(DTR, DTE) 
    if m < 12:
        DTR, DTE = utils.apply_PCA(DTR, DTE, m) 

    v = numpy.zeros((DTR.shape[0]+1))   
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)

    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v, approx_grad = True, factr=1.0, maxfun=50000)

    w, b = utils.vcol(x[:-1]), x[-1]

    S= compute_scoreMatrix_logreg(w, b, DTE)
    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S, LTE)

    if m < 12: 
        print("Log Reg \t-\t minDCF :  %1.3f (\u03BB = %2.0e) \t m : %d" % (minDCF, l, m))
    else:
        print("Log Reg \t-\t minDCF :  %1.3f (\u03BB = %2.0e) \t Raw data" % (minDCF, l))

def evaluate_Quad_logreg(DTR, LTR, DTE, LTE, pi, gauss, m): 
    l = 1e-5          
    S = numpy.zeros((DTR.shape[1])) 
    DTR, DTE = expanded_features_space(DTR, DTE) 

    if m < 12: 
        DTR, DTE = utils.apply_PCA(DTR, DTE, m)  

    v = numpy.zeros((DTR.shape[0]+1))     
    logreg_obj = logreg_obj_wrap(DTR, LTR, l) 

    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v, approx_grad = True, factr=1.0, maxfun=50000) 
    w, b = utils.vcol(x[:-1]), x[-1] 
        
    S  = compute_scoreMatrix_logreg(w, b, DTE) 

    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S, LTE)   

    if m < 12:  
        print("Quad Reg \t-\t minDCF :  %1.3f (\u03BB = %2.0e) \t m : %d" % (minDCF, l, m)) 
    else: 
        print("Quad Reg \t-\t minDCF :  %1.3f (\u03BB = %2.0e) \t Raw data" % (minDCF, l)) 
