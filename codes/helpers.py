import numpy as np

def sb_lms(x, d, mu, K):
    """
    Implements sequential block least mean square algorithm which is the assigned adaptive filter algorithm
    d[n] = s[n] + (h * x)[n],
    d[n]: received signal
    x[n]: noise signal
    s[n]: source signal
    
    :param x: noise signal, x[n]
    :param x: measured signal, d[n]
    :param mu: step size
    :param K: adaptive filter order
    :return: error signal e[n]
    """
    f = np.zeros(K)
    x_buffer = np.ones(K)
    
    N = len(x)
    e = np.zeros(N)
    
    for i in range(N):
        x_buffer = np.r_[x[i], x_buffer[:-1]]
        e[i] = d[i] - np.dot(x_buffer, f)
        f = f + mu * e[i] * x_buffer
    
    return e
