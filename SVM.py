import numpy  
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt


import BayesDecision 
import utils 



def compute_H_ext_linear(DTR, Z):
    z = numpy.dot(utils.vcol(Z), utils.vrow(Z))
    G = numpy.dot(DTR.T, DTR)
    H = z*G
    return H

def compute_H_ext_kernel(DTR, LTR, k):
    z = numpy.dot(utils.vcol(LTR), utils.vrow(LTR))
    G = k(DTR, DTR)
    H = z*G
    return H

def compute_w(alpha, DTR, Z):
    w = numpy.sum(alpha * Z * DTR, axis=1)
    return w

def compute_score_linear(w_ext, DTE, K):
    DTE_ext = numpy.vstack((DTE, K*numpy.ones((1, DTE.shape[1]))))
    S = numpy.dot(utils.vrow(w_ext), DTE_ext)    
    return S

def compute_score_kernel(alpha, LTR, DTR, DTE, k): 
    S = numpy.sum(utils.vcol(alpha) * utils.vcol(LTR) * k(DTR, DTE), axis=0)
    return S

def Quad_wrap(c, xi):
    def compute_polynomial_kernel_ext_matrix(X1, X2, degree=2):
        k = (numpy.dot(X1.T, X2) + c) ** degree 
        k_ext = k + xi
        return k_ext
    return compute_polynomial_kernel_ext_matrix

def RBF_wrap(gamma, xi):
    def compute_RBF_kernel_ext_matrix(X1, X2):
        k = numpy.zeros((X1.shape[1], X2.shape[1]))
        for i in range(X1.shape[1]):
            for j in range(X2.shape[1]):
                x_dist = X1[:,i:i+1] - X2[:,j:j+1]
                k[i,j] = numpy.exp(- gamma * numpy.dot(x_dist.T, x_dist))
        k_ext = k + xi
        return k_ext
    return compute_RBF_kernel_ext_matrix

def LDual_wrap(H_ext):
    def LDual(alpha):
        Ha = numpy.dot(H_ext, utils.vcol(alpha))
        aHa = numpy.dot(utils.vrow(alpha), Ha)
        a1 = alpha.sum()
        return 0.5 * aHa.ravel() - a1, Ha.ravel() - numpy.ones(alpha.size)
    return LDual

def validate_linearSVM(C, DTR, LTR, pi, gauss, m, k=5):
    K = 1
    S = numpy.zeros((DTR.shape[1]))
    for i, j in enumerate(range(0, DTR.shape[1], DTR.shape[1]//k)):    
        (Dtr, Ltr), (Dva, Lva) = utils.k_fold(DTR, LTR, i, k)
        if gauss == 1:
            Dtr, Dva = utils.features_gaussianization(Dtr, Dva) 
        if m < 12:
            Dtr, Dva = utils.apply_PCA(Dtr, Dva, m) 
        Z = numpy.zeros(Ltr.shape)
        Dtr_ext = numpy.vstack((Dtr, K*numpy.ones((1, Dtr.shape[1]))))     
        Z[Ltr == 1] = 1                                                 
        Z[Ltr == 0] = -1

        H_ext = compute_H_ext_linear(Dtr_ext, Z)

        LDual = LDual_wrap(H_ext)
        (x, f, d) = scipy.optimize.fmin_l_bfgs_b(LDual, numpy.zeros(Dtr.shape[1]), bounds=[(0, C)] * Dtr.shape[1], factr=1000.0)
        w_ext = compute_w(x, Dtr_ext, Z)

        S[j: j + Dva.shape[1]] = compute_score_linear(w_ext, Dva, K)

    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S, Lva)

    if m < 12: 
        print("Linear SVM \t-\t minDCF : %1.3f (C = %2.0e) \t m = %d" % (minDCF, C, m))
    else:
        print("Linear SVM \t-\t minDCF : %1.3f (C = %2.0e) \t Raw Data" % (minDCF, C))
    return minDCF

def plot_linearSVM_C(DTR, LTR):
    nC = 30
    minDCF_pi = numpy.zeros((3, nC))
    Cs = numpy.logspace(-3, 1, num=nC, endpoint=True)
    for i, pi in enumerate([.1, .5, .9]):
        for j, C in enumerate(Cs):
            minDCF_pi[i, j] = validate_linearSVM(C, DTR, LTR, pi, 0, 12)

    x1 = [1e-2, 1e-1, 1e-0]
    plt.plot(Cs, minDCF_pi[0,:], label="minDCF($\widetilde{\pi}$ = 0.1)", color='blue', linewidth=1)
    plt.plot(Cs, minDCF_pi[1,:], label="minDCF($\widetilde{\pi}$ = 0.5)", color='red', linewidth=1)
    plt.plot(Cs, minDCF_pi[2,:], label="minDCF($\widetilde{\pi}$ = 0.9)", color='green', linewidth=1)
    plt.xscale("log")
    plt.xlim([1e-3, 1e1])
    plt.gca().xaxis.set_ticks(x1)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(8)
    plt.xlabel('C', fontsize=8)
    plt.ylabel('DCF', fontsize=8)
    plt.legend(prop={"size":8})
    plt.show()

def validate_kernelSVM(C, gamma, DTR, LTR, pi, gauss, kernelType, m, k=5):
    xi = 1
    c = 1.0

    if kernelType == 'Quad':
        kernel = Quad_wrap(c, xi)
    elif kernelType == 'RBF':
        kernel = RBF_wrap(gamma, xi)

    S = numpy.zeros((DTR.shape[1]))
    for i, j in enumerate(range(0, DTR.shape[1], DTR.shape[1]//k)):    
        (Dtr, Ltr), (Dva, Lva) = utils.k_fold(DTR, LTR, i, k)
        if gauss == 1:
            Dtr, Dva = utils.features_gaussianization(Dtr, Dva) 
        if m < 12:
            Dtr, Dva = utils.apply_PCA(Dtr, Dva, m) 
        Z = numpy.zeros(Ltr.shape)
        Z[Ltr == 1] = 1                                                 
        Z[Ltr == 0] = -1
        H_ext = compute_H_ext_kernel(Dtr, Z, kernel)    
        LDual = LDual_wrap(H_ext)
        (x, f, d) = scipy.optimize.fmin_l_bfgs_b(LDual, numpy.zeros(Dtr.shape[1]), bounds=[(0, C)] * Dtr.shape[1], factr=1000.0)

        S[j: j + Dva.shape[1]] =  compute_score_kernel(x, Z, Dtr, Dva, kernel)

    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S, Lva)

    if kernelType == 'Quad':
        print("Quadratic SVM \t-\t minDCF : %1.3f \u221a\u03BE: %.1f C: %.3f c: %d" % (minDCF, xi, C, c))  
    else:
        print("RBF SVM \t-\t minDCF : %1.3f \u221a\u03BE: %.1f C: %.5f \u03B3: %.3f" % (minDCF, xi, C, gamma))  
    return minDCF

def plot_QuadSVM_C(DTR, LTR):
    nC = 30
    minDCF_pi = numpy.zeros((3, nC))
    Cs = numpy.logspace(-3, 1, num=nC, endpoint=True)
    for i, pi in enumerate([.1, .5, .9]):
        for j, C in enumerate(Cs):
            minDCF_pi[i, j] = validate_kernelSVM(C, 0, DTR, LTR, pi, 0, 'Quad', 12)

    x1 = [1e-2, 1e-1, 1e-0]
    plt.plot(Cs, minDCF_pi[0,:], label="minDCF($\widetilde{\pi}$ = 0.1)", color='blue', linewidth=1.5)
    plt.plot(Cs, minDCF_pi[1,:], label="minDCF($\widetilde{\pi}$ = 0.5)", color='red', linewidth=1.5)
    plt.plot(Cs, minDCF_pi[2,:], label="minDCF($\widetilde{\pi}$ = 0.9)", color='green', linewidth=1.5)
    plt.xscale("log")
    plt.xlim([1e-3, 1e1])
    plt.gca().xaxis.set_ticks(x1)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(8)
    plt.xlabel('C', fontsize=8)
    plt.ylabel('DCF', fontsize=8)
    plt.legend(prop={"size":8})
    plt.show()

def plot_rbfSVM_C(DTR, LTR):
    nC = 30
    minDCF_pi = numpy.zeros((3, nC))
    Cs = numpy.logspace(-2, 3, num=nC, endpoint=True)
    for i, gamma in enumerate([1e-4, 1e-3, 1e-2]):
        for j, C in enumerate(Cs):
            minDCF_pi[i, j] = validate_kernelSVM(C, gamma, DTR, LTR, .5, 0, 'RBF', 12)
            
    x1 = [1e-1, 1e0, 1e1, 1e2]
    plt.plot(Cs, minDCF_pi[0,:], label="log\u03B3 = -4", color='gold', linewidth=1.5)
    plt.plot(Cs, minDCF_pi[1,:], label="log\u03B3 = -3", color='darkorange', linewidth=1.5)
    plt.plot(Cs, minDCF_pi[2,:], label="log\u03B3 = -2", color='darkred', linewidth=1.5)
    plt.xscale("log")
    plt.xlim([1e-2, 1e3])
    plt.gca().xaxis.set_ticks(x1)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(8)
    plt.xlabel('C', fontsize=8)
    plt.ylabel('DCF', fontsize=8)
    plt.legend(prop={"size":8})
    plt.show()

def evaluate_kernelSVM(C, gamma, DTR, LTR, DTE, LTE, pi, gauss, kernelType, m):
    xi = 1
    c = 1.0

    if kernelType == 'Quad':
        kernel = Quad_wrap(c, xi)
    elif kernelType == 'RBF':
        kernel = RBF_wrap(gamma, xi)

    S = numpy.zeros((DTR.shape[1]))

    if gauss == 1:
        DTR, DTE = utils.features_gaussianization(DTE, DTE) 
    if m < 12:
        DTR, DTE = utils.apply_PCA(DTR, DTE, m) 
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1                                                 
    Z[LTR == 0] = -1

    H_ext = compute_H_ext_kernel(DTR, Z, kernel)               

    LDual = LDual_wrap(H_ext)
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(LDual, numpy.zeros(DTR.shape[1]), bounds=[(0, C)] * DTR.shape[1], factr=1000.0)

    S =  compute_score_kernel(x, Z, DTR, DTE, kernel)

    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S.ravel(), LTE)

    if kernelType == 'Quad':
        print("Quadratic SVM \t-\t minDCF : %1.3f \u221a\u03BE: %.1f C: %.3f c: %d" % (minDCF, xi, C, c))  
    else:
        print("RBF SVM \t-\t minDCF : %1.3f \u221a\u03BE: %.1f C: %.5f \u03B3: %.3f" % (minDCF, xi, C, gamma))  
    return minDCF

def evaluate_linearSVM(C, DTR, LTR, DTE, LTE, pi, gauss, m):
    K = 1
    S = numpy.zeros((DTE.shape[1]))
    if gauss == 1:
        DTR, DTE = utils.features_gaussianization(DTR, DTE) 
    if m < 12:
        DTR, DTE = utils.apply_PCA(DTR, DTE, m) 
    Z = numpy.zeros(LTR.shape)
    Dtr_ext = numpy.vstack((DTR, K*numpy.ones((1, DTR.shape[1]))))              # Building the extended matrix of training data     
    Z[LTR == 1] = 1                                                 
    Z[LTR == 0] = -1

    H_ext = compute_H_ext_linear(Dtr_ext, Z)

    LDual = LDual_wrap(H_ext)
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(LDual, numpy.zeros(DTR.shape[1]), bounds=[(0, C)] * DTR.shape[1], factr=1000.0)
    w_ext = compute_w(x, Dtr_ext, Z)

    S = compute_score_linear(w_ext, DTE, K)

    minDCF = BayesDecision.compute_minimum_detection_cost(pi, S.ravel(), LTE)

    if m < 12: 
        print("Linear SVM \t-\t minDCF : %1.3f (C = %2.0e) \t m = %d" % (minDCF, C, m))
    else:
        print("Linear SVM \t-\t minDCF : %1.3f (C = %2.0e) \t Raw Data" % (minDCF, C))
    return minDCF