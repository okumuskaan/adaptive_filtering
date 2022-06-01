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


def rls(x, d, mu = .999 , eps = .00001, K=5):
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

def fast_block_lms(x, d, mu, K):
    f = np.zeros(2*K) + 0j
    f = np.random.randn(2*K) + 0j
    x_buffer = np.zeros(2*K) + 0j
    
    N = len(x)
    e = np.zeros(N) + 0j
    e_buffer = np.zeros(2*K) + 0j
    zeros_vec = np.zeros(K) + 0j
    
    for i in range(int(N/K)):
        
        x_buffer = np.r_[x_buffer[:K], x[i*K:(i+1)*K]] # 1
        X = np.fft.fft(x_buffer) # 2
        y = np.fft.ifft(f * X) # 3 & 4
        e[i*K:(i+1)*K] = d[i*K:(i+1)*K] - y[K:] # 5 & 6
        
        e_buffer[K:] = e[i*K:(i+1)*K] # 7
        E = np.fft.fft(e_buffer) # 7 & 8
        
        y = np.fft.ifft(E*np.conj(X)) # 9 & 10
        y[K:] = zeros_vec # 11
        f = f + mu * np.fft.fft(y) # 12 & 13

    return np.real(e)


def get_transform_matrix(size):
    transform_matrix = np.array(range(size), np.float64).repeat(size).reshape(size, size)
    transform_matrixT = np.pi* (transform_matrix.transpose() + 0.5) / size
    transform_matrixT = (1.0/np.sqrt( size / 2.0)) * np.cos(transform_matrix * transform_matrixT)
    transform_matrixT[0] = transform_matrixT[0] * (np.sqrt(2.0)/2.0)

    return transform_matrixT

def dct_lms(x, d, mu, K, alpha=0.01, gamma = 1e-8, init_power=1):
    
    N = len(x)
    
    e = np.zeros(N, dtype=complex)
    f = np.zeros(K, dtype=complex)
    f_dct = np.zeros(K, dtype=complex)
    T = get_transform_matrix(K)
    T_herm = np.conj(np.transpose(T))
    
    power_vec = init_power * np.ones(K)
    
    f_dct = np.dot(T, f)
    
    x_buffer = np.zeros(K, dtype=complex)
        
    for i in np.arange(N):
        x_buffer = np.roll(x_buffer, 1)
        x_buffer[0] = x[i]
        
        x_dct = np.dot(T, x_buffer)
        
        power_vec = alpha * np.multiply(x_dct, np.conj(x_dct)) + (1 - alpha) * power_vec
        
        e[i] = d[i] - np.dot(np.conj(f_dct), x_dct)

        aux_numerator = np.dot(np.conj(e[i]), x_dct)
        aux_denominator = gamma + power_vec          
        f_dct = f_dct + mu * np.divide(aux_numerator, aux_denominator)        
        f = np.dot(T_herm, f_dct)
        
    return np.real(e)

def affine_projection(x, d, mu, K, L, gamma=0.001):
    
    N = len(x)
    
    d_ap = np.zeros(L+1, dtype=complex)
    e_ap = np.zeros(L+1, dtype=complex)
    
    e = np.zeros(N, dtype=complex)
    f = np.zeros(K, dtype=complex)
    
    x_buffer = np.zeros(K, dtype=complex)
    x_ap = np.zeros([L+1, K], dtype=complex)
        
    for i in np.arange(N):
        x_buffer = np.roll(x_buffer, 1)
        x_buffer[0] = x[i]
        
        x_ap = np.roll(x_ap, 1, axis=0)
        x_ap[0] = x_buffer
        
        d_ap = np.roll(d_ap, 1)
        d_ap[0] = d[i]
        
        e_ap = d_ap - np.matmul(np.conj(f), np.transpose(x_ap))
        
        f = f + mu * np.matmul(np.conj(e_ap), np.matmul(np.transpose(np.linalg.inv(np.matmul(np.conj(x_ap), np.transpose(x_ap)) + gamma * np.eye(L+1, dtype = complex))), x_ap))
        
        e[i] = e_ap[0]
        
        
    return np.real(e)

