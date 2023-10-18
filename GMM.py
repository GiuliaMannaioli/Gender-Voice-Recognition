import numpy  
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

import utils
import BayesDecision

def validate_GMM(DTR, LTR, covType, NG, pi, m, gauss, k=5):
    GMM = []
    S_GMM = numpy.zeros((len(set(LTR)), DTR.shape[1]))
    for i, j in enumerate(range(0, DTR.shape[1], DTR.shape[1]//k)):    
        (Dtr, Ltr), (Dva, Lva) = utils.k_fold(DTR, LTR, i, k)
        if gauss == 1:
            Dtr, Dva = utils.features_gaussianization(Dtr, Dva)
        if m < 12:
            Dtr, Dva = utils.apply_PCA(Dtr, Dva, m)
        for lab in set(Ltr):
            mu, C = utils.compute_empirical_moments(Dtr[:, Ltr == lab])
            GMM_1 = [(1.0, mu, C)]

            gmm = LBG(Dtr[:, Ltr == lab], GMM_1, NG, covType)
            GMM.append(gmm)

        for lab in set(LTR):
            S_GMM[lab, j: j + Dva.shape[1]] =  scipy.special.logsumexp(logpdf_GMM(Dva, GMM[lab]), axis=0)

    llr = S_GMM[1, :] - S_GMM[0, :]

    BayesMatrix = BayesDecision.compute_BayesMatrix(llr, Lva, pi)    
    _, DCF = BayesDecision.compute_BayesRisk(BayesMatrix, pi)
    minDCF = BayesDecision.compute_minimum_detection_cost(pi, llr, Lva)

    if m < 12: 
        print("%s\t-\tminDCF : %1.3f \tDCF : %1.3f \t \u03C0_eff : %1.1f \t Ng = %d \tm = %d" % (covType, minDCF, DCF, pi, NG, m))
    else:
        print("%s\t-\tminDCF : %1.3f \tDCF : %1.3f \t \u03C0_eff : %1.1f \t Ng = %d \t Raw Data" % (covType, minDCF, DCF, pi, NG))
    return minDCF, llr, Lva

def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    for i, g in enumerate(gmm):
        w, mu, C = g
        S[i,:] = numpy.log(w) + utils.logpdf_GAU_ND_2(X, mu, C)        # we can interpret Gaussian components G as representing sub-classes (i.e. clusters) of our data
    return S

def E_step(logS):
    logdens = scipy.special.logsumexp(logS, axis=0)
    Post = numpy.exp(logS - logdens)        
    all = numpy.sum(logdens)/Post.shape[1]
    return Post, all

def compute_stats(XND, Post):
    Zg = numpy.sum(Post, axis=1)
    Fg = numpy.dot(Post, XND.T)
    Sg = []
    for g in range(Post.shape[0]):
        Sg.append(numpy.dot(Post[g,:] * XND, XND.T))
    return Zg, Fg, Sg, XND

def M_step(Stats, covType, psi=0.01):
    Zg, Fg, Sg, XND = Stats
    gmm = []
    for g in range(Zg.size):
        mug = utils.vcol(Fg[g,:]/Zg[g])
        wg = Zg[g]/numpy.sum(Zg)
        Cg = Sg[g]/Zg[g] - numpy.dot(mug, mug.T) 
        if 'Diag-Cov' in covType:
            Cg = Cg * numpy.eye(Cg.shape[0])
        gmm.append([wg, mug, Cg])

    if 'Tied' in covType:
        C = 0
        for g in range(len(gmm)):
            C += Zg[g]*gmm[g][2]

    for g in gmm:                                  
        if 'Tied' in covType:
            g[2] = C/XND.shape[1]
        U, s, _ = numpy.linalg.svd(g[2])             # constraints application
        s[s < psi] = psi
        g[2] = numpy.dot(U, utils.vcol(s)*U.T)

    return gmm    

def split_GMM(GMM_pre, alpha=.1):
    gmm = []
    for g in GMM_pre:
        wg, mug, Cg = g
        U, s, _ = numpy.linalg.svd(Cg)
        d = U[:, 0:1] * s[0]**0.5 * alpha 
        gmm.append((wg/2, mug - d, Cg)) 
        gmm.append((wg/2, mug + d, Cg))  
    return gmm

def LBG(X, gmm, NG, covType):
    while len(gmm) <= NG:
        gmm = GMM_EM(X, gmm, covType)
        if len(gmm) == NG:
            break
        gmm = split_GMM(gmm)
    return gmm

def GMM_EM(X, gmm, covType, llDiff=1e-6):
    llNew = None
    llOld = None
    while llOld is None or llNew - llOld > llDiff:
        llOld = llNew
        Post, llNew = E_step(logpdf_GMM(X, gmm))
        gmm = M_step(compute_stats(X, Post), covType, psi=0.01)
    # print(llNew)
    return gmm

def plot_GMM(DTR, LTR):
    ng_max = 1024
    NGs = numpy.power(2, range(int(numpy.log2(ng_max))+1))
    x_axis = [i*2 for i in range(len(NGs))]
    minDCF_ct = numpy.zeros((4, len(NGs)))
    for i, covType in enumerate(['Full-Cov', 'Diag-Cov', 'Tied Full-Cov', 'Tied Diag-Cov']):
        for j, ng in enumerate(NGs):
            minDCF, _, _ = validate_GMM(DTR, LTR, covType, ng, 0.5, 12, 0)
            minDCF_ct[i,j] = minDCF

    barWidth = .3
    br1 = x_axis
    br2 = [x + barWidth+0.05 for x in br1]
    br3 = [x + barWidth+0.05 for x in br2]
    br4 = [x + barWidth+0.05 for x in br3]
    plt.bar(br1, minDCF_ct[0,:], label="minDCF($\widetilde{\pi}$ = 0.5) - Full-Cov", color='gold', linewidth=.5, width = barWidth, edgecolor ='black')
    plt.bar(br2, minDCF_ct[1,:], label="minDCF($\widetilde{\pi}$ = 0.5) - Diag-Cov", color='darkorange', linewidth=.5, width = barWidth, edgecolor ='black')
    plt.bar(br3, minDCF_ct[2,:], label="minDCF($\widetilde{\pi}$ = 0.5) - Tied Full-Cov", color='red', linewidth=.5, width = barWidth, edgecolor ='black')
    plt.bar(br4, minDCF_ct[3,:], label="minDCF($\widetilde{\pi}$ = 0.5) - Tied Diag-Cov", color='darkred', linewidth=.5, width = barWidth, edgecolor ='black')
    br_mat = numpy.vstack([br1, br2, br3, br4]).T
    plt.gca().xaxis.set_ticks(br_mat.mean(1), NGs)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(8)
    plt.xlabel('GMM components', fontsize=8)
    plt.ylabel('DCF', fontsize=8)
    plt.legend(prop={"size":8})
    plt.show()

def evaluate_GMM(DTR, LTR, DTE, LTE, covType, NG, pi, m, gauss):
    GMM = []
    S_GMM = numpy.zeros((len(set(LTR)), DTE.shape[1]))
    if gauss == 1:
        DTR, DTE = utils.features_gaussianization(DTR, DTE)
    if m < 12:
        DTR, DTE = utils.apply_PCA(DTR, DTE, m)
        
    for lab in set(LTR):
        mu, C = utils.compute_empirical_moments(DTR[:, LTR == lab])
        GMM_1 = [(1.0, mu, C)]

        gmm = LBG(DTR[:, LTR == lab], GMM_1, NG, covType)
        GMM.append(gmm)

    for lab in set(LTR):
        S_GMM[lab,:] =  scipy.special.logsumexp(logpdf_GMM(DTE, GMM[lab]), axis=0)

    llr = S_GMM[1, :] - S_GMM[0, :]
  
    minDCF = BayesDecision.compute_minimum_detection_cost(pi, llr, LTE)

    if m < 12: 
        print("%s\t-\tminDCF : %1.3f \t \u03C0_eff : %1.1f \t Ng = %d \tm = %d" % (covType, minDCF, pi, NG, m))
    else:
        print("%s\t-\tminDCF : %1.3f \t \u03C0_eff : %1.1f \t Ng = %d \t Raw Data" % (covType, minDCF, pi, NG))