import numpy  
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

import logreg

def loadTrain(fname):
    DTR = []
    LTR = []

    with open(fname) as f:
        for line in f:
            try:
                sample = numpy.array([float(i) for i in line.rstrip("\n").split(',')[0:12]]).reshape(12,1)
                # label = h_labels[line.rstrip("\n").split(',')[-1]]
                label = int(line.rstrip("\n").split(',')[-1])
                DTR.append(sample)     
                LTR.append(label)
            except:
                pass
    return numpy.hstack(DTR), numpy.array(LTR)     

def loadTest(fname):
    DTE = []
    LTE = []

    with open(fname) as f:
        for line in f:
            try:
                sample = numpy.array([float(i) for i in line.rstrip("\n").split(',')[0:12]]).reshape(12,1)
                # label = h_labels[line.rstrip("\n").split(',')[-1]]      if needed
                label = int(line.rstrip("\n").split(',')[-1]) 
                DTE.append(sample)     
                LTE.append(label)
            except:
                pass
    return numpy.hstack(DTE), numpy.array(LTE) 

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def plot_scatter(D,L):
    D0 = D[:, L==0]                       # sample of male
    D1 = D[:, L==1]                       # sample of female
        
    plt.scatter(D0[0,:],D0[1,:], label="Male", c="tab:blue", ec="dodgerblue", alpha=0.9, marker='o')
    plt.scatter(D1[0,:],D1[1,:], label="Female", c="tab:orange", ec="crimson", alpha=0.9, marker='o')
    plt.legend()
    plt.show()

def plot_scatter3D(D,L):
    D0 = D[:, L==0]                       # sample of male
    D1 = D[:, L==1]                       # sample of female
        
    ax = plt.axes(projection ="3d")
    ax.scatter3D(D0[0,:],D0[1,:],D0[2,:], label="Male", c="tab:blue", ec="dodgerblue", alpha=0.9, marker='o')
    ax.scatter3D(D1[0,:],D1[1,:],D0[2,:], label="Female", c="tab:orange", ec="crimson", alpha=0.9, marker='o')
    plt.legend()
    plt.show()

def compute_empirical_moments(D):
    mu = vcol(D.mean(1))                
    C = numpy.dot((D - mu),(D - mu).T)/D.shape[1]
    return mu,C

def apply_PCA(DTR, DTE, m):
    mu, C = compute_empirical_moments(DTR)
    s, U = numpy.linalg.eigh(C)              
    P = U[:, ::-1][:, 0:m] 
    return numpy.dot(P.T, DTR-mu), numpy.dot(P.T, DTE-mu)

def compute_heat_map(DTR): 
    _,C = compute_empirical_moments(DTR)
    M = numpy.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
                M[i,j] = abs(C[i,j]/((C[i,i]**.5) * (C[j,j]**.5)))
    return M

def plot_correlation_heat_map(DTR, LTR):
    M = compute_heat_map(DTR)
    M_f = compute_heat_map(DTR[:, LTR==1])
    M_m = compute_heat_map(DTR[:, LTR==0])
    LM = [M, M_f, M_m]
    cmap = ['binary', 'Reds', 'Blues']

    fig, axs = plt.subplots(1, 3)
    for i in range(len(LM)):   
        axs[i].imshow(LM[i], cmap=cmap[i])
        axs[i].set_xticks(range(M.shape[0]), range(1, M.shape[0] + 1))
        axs[i].set_yticks(range(M.shape[0]), range(1, M.shape[0] + 1))
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontsize(8)
    plt.show()

def plot_features(DTR, LTR):
    D_m = DTR[:, LTR==0]                     
    D_f = DTR[:, LTR==1]                  
    
    fig, axs = plt.subplots(3,4)
    for i in range(DTR.shape[0]//4):             
        for j in range(DTR.shape[0]//3):              
            axs[i][j].hist(D_m[j+i*(DTR.shape[0]//4 + 1),:], bins=50, density=True, label="male", color="tab:blue", ec="dodgerblue", alpha=0.5)
            axs[i][j].hist(D_f[j+i*(DTR.shape[0]//4 + 1),:], bins=50, density=True, label="female", color="tab:orange", ec="crimson", alpha=0.5)
            # axs[i][j].set_title(" %d " % (i+j+1))
            axs[i][j].text(0.04, 0.94, str(j+i*(DTR.shape[0]//4 + 1)+1), bbox=dict(facecolor = 'white', alpha = 0.4), fontsize=7, horizontalalignment='left', verticalalignment='top', transform=axs[i][j].transAxes)
            # axs[i][j].text(-15, 0.05, str(i), fontsize=8)
            for label in (axs[i][j].get_xticklabels() + axs[i][j].get_yticklabels()):
                label.set_fontsize(8)
    plt.show()

def features_gaussianization(DTR, DTE):  
    r_DTR = numpy.ones((DTR.shape)) 
    for i in range(DTR.shape[1]):
        r_DTR[:, i] += (DTR < vcol(DTR[:, i])).sum(axis=1)
    r_DTR /= DTR.shape[1] + 2

    r_DTE = numpy.ones((DTE.shape)) 
    for i in range(DTE.shape[1]):
        r_DTE[:, i] += (DTR < vcol(DTE[:, i])).sum(axis=1)
    r_DTE /= DTR.shape[1] + 2
    return scipy.stats.norm.ppf(r_DTR), scipy.stats.norm.ppf(r_DTE)

def k_fold(D, L, test_pos, k, seed=0):
    fold_size = D.shape[1]//k                       
    numpy.random.seed(seed)
    DTR = []
    LTR = []
    idx = numpy.random.permutation(D.shape[1])   
    # idx = list(range(D.shape[1]))            
    for i in range(k):
        idx_folds = idx[i*fold_size:(i+1)*fold_size]    
        if (i == test_pos):
            DVA = (D[:, idx_folds])
            LVA = L[idx]
        else:  
            DTR.append(D[:, idx_folds])
            LTR.append(L[idx_folds])
    return (numpy.hstack(DTR), numpy.hstack(LTR)), (DVA, LVA)

def logpdf_GAU_ND_2(XND, mu, C):
    logpdf = -XND.shape[0]/2*numpy.log(2*numpy.pi)-1/2*numpy.linalg.slogdet(C)[1]-1/2*((XND-mu)*numpy.dot(numpy.linalg.inv(C), (XND-mu))).sum(0)    # con sum(0) somma le due righe 
    return logpdf

def single_fold_score(D, L, seed=0):
    nTrain = int(D.size*2.0/3.0)              
    numpy.random.seed(seed)                      
    idx = numpy.random.permutation(D.size)

    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]       
    LTR_score = L[idxTrain] 
    LVA_score = L[idxTest] 
    DTR_score = D[idxTrain]                  
    DVA_score = D[idxTest]                      
    return (DTR_score, LTR_score), (DVA_score, LVA_score)

def score_calibarion(llr, Lva, pi):
    S, Lva = logreg.logreg_on_score(vrow(numpy.array(llr)), Lva, pi)
    return S, Lva
