import numpy as np

def sb_lms(x, d, mu, K, normalized=False, a=0.0, type="standard"):
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
    
    func_norm = (lambda x: np.dot(x,x)) if normalized else lambda x: 1
    
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
    else:
        raise ValueError("Type must be either standard, sign-error, sign-data or sign-sign.")
    
    for i in range(N):
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
    
    for i in range(N):
        x_buffer = np.r_[x[i], x_buffer[:-1]]
        e[i] = d[i] - np.dot(x_buffer, f)
        prev_fs = [f, prev_fs[0]]
        f = f + mu * e[i] * x_buffer + beta * (prev_fs[0] - prev_fs[1])
    
    return e

"""
def block_lms(x, d, mu, K, L):
"""
"""
    Implement block least mean square algorithm
    d[n] = s[n] + (h * x)[n],
    d[n]: received signal
    x[n]: input noise signal
    s[n]: source signal (tried to be estimated)
    
    :param x: input noise signal, x[n]
    :param d: measured signal, d[n]
    :param mu: step size
    :param K: adaptive filter order
    :param L: block length
    
    :return: error signal e[n] (estimated s[n])
"""
"""
    f = np.zeros(K)
    x_buffer = np.zeros(K)
    x_buffer2 = np.zeros(L)
    x_buffblock = np.zeros(L)
    
    N = len(x)
    e = np.zeros(N)
    
    for i in range(int(np.floor(N/L))):
        for j in range(L):
            x_buffer = np.r_[x[i*L:j], x_buffer[:-1]]
            x_buffer2 = np.r_[x[i*L:j], x_buffer2[:-1]]
            x_buffblock(j) = np.dot(x_buffer2, f)
        e[i*L:i*(L+1)] = d[i*L:i*(L+1)] - x_buffblock
        f = f + mu * e[i] * x_buffer
    
    return e
"""