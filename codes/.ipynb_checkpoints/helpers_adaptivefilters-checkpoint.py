import numpy as np
from tqdm import tqdm

def sb_lms(x, d, mu, K, normalized=False, a=0.0, type="standard", beta=None, tqdm_print=False):
    """
    Implement sequential block least mean square algorithm which is the assigned adaptive filter algorithm
    d[n] = s[n] + (h * x)[n],
    d[n]: received signal
    x[n]: input noise signal
    s[n]: source signal (tried to be estimated)
    
    :param x: input noise signal, x[n]
    :param d: measured signal, d[n]
    :param mu: step size
    :param K: adaptive filter order
    :param normalized: normalized LMS or not (default: False)
    :param a: leaky factor (default: 0.0)
    :param type: type of lms variant (default: "standard")
    
    :return: error signal e[n] (estimated s[n])
    """
    f = np.zeros(K)
    x_buffer = np.zeros(K)
    
    N = len(x)
    e = np.zeros(N)
    
    func_norm = (lambda x: 0.001+np.dot(x,x)) if normalized else (lambda x: 1)
    
    func_tqdm = (lambda x: tqdm(x, desc="Progress")) if tqdm_print else (lambda x: x)
    
    if type=="standard":
        func_e = lambda x: x
        func_x = lambda x: x
    elif type=="sign-error":
        func_e = np.sign
        func_x = lambda x: x
    elif type=="sign-data":
        func_e = lambda x: x
        func_x = np.sign
    elif type=="sign-sign":
        func_e = np.sign
        func_x = np.sign
    elif type=="momentum":
        return momentum_lms(x, d, mu, K, beta)
    else:
        raise ValueError("Type must be either standard, sign-error, sign-data or sign-sign.")
    
    for i in func_tqdm(range(N)):
        x_buffer = np.r_[x[i], x_buffer[:-1]]
        e[i] = d[i] - np.dot(x_buffer, f)
        f = f * (1 - mu * a) + mu * func_e(e[i]) * func_x(x_buffer) / func_norm(x_buffer)
    
    return e

def momentum_lms(x, d, mu, K, beta):
    """
    Implements momentum sequential block least mean square algorithm which is the variant of assigned adaptive filter algorithm
    d[n] = s[n] + (h * x)[n],
    d[n]: received signal
    x[n]: input noise signal
    s[n]: source signal (tried to be estimated)
    
    :param x: input noise signal, x[n]
    :param d: measured signal, d[n]
    :param mu: step size
    :param K: adaptive filter order
    :return: error signal e[n] (estimated s[n])
    """
    prev_fs = [np.zeros(K), np.zeros(K)]
    f = np.zeros(K)
    x_buffer = np.ones(K)
    
    N = len(x)
    e = np.zeros(N)
    
    for i in tqdm(range(N), desc="Progress"):
        x_buffer = np.r_[x[i], x_buffer[:-1]]
        e[i] = d[i] - np.dot(x_buffer, f)
        prev_fs = [f, prev_fs[0]]
        f = f + mu * e[i] * x_buffer + beta * (prev_fs[0] - prev_fs[1])
    
    return e


def rls(x, d, mu, eps, K):
    # initial guess for the filter
    w = np.zeros(K)
    R = 1/eps*np.identity(K)
    X = np.zeros(K)

    # number of iterations
    L = len(d)
    e = np.zeros(L)
    
    # run the adaptation
    for n in range(L):
        
        X = np.concatenate(([x[n]], X[:K-1]))
        
        R1 = np.inner(np.inner(np.inner(R,X),X.T),R)
        R2 = mu + np.dot(np.dot(X,R),X.T)
        R = 1/mu * (R - R1/R2)
        w = w + np.dot(R, X.T) * (d[n] - np.dot(X.T, w))
            
        # estimate output and error
        e[n] = d[n] - np.dot(w.transpose(), X) 
        
    return e

