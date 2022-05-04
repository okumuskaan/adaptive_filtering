import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from helpers_adaptivefilters import sb_lms
from tqdm import tqdm
import time

def generate_signals(f, T, fs, sigma_x=1, h=None, h_len=5, sigma_h=1, plot=True):
    t = np.arange(0,T,1/fs)
    N = len(t)
    
    s = np.sin(2*np.pi*f*t)
    x = sigma_x * np.random.randn(N)
    if h is None:
        h = sigma_h * np.random.randn(h_len)
    else:
        h_len = len(h)
        sigma_h = np.std(h)
    
    noise = np.convolve(h, x)
    d = s + noise[:N]
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16,5)) 
        axes[0].plot(t,s)
        axes[0].set_title("Source signal: s[n]")
        axes[1].plot(t, d) 
        axes[1].set_title("Received signal: d[n]")
        plt.show()
    
    return t, s, x, d
    

def find_misadjustment(aee, conv_point, fs, margin=100, print_misad=True):
    if int(conv_point*fs)+margin>len(aee):
        return -1
    temp = aee[int(conv_point*fs)+margin:]
    max_misadjustment = np.max(temp)
    if print_misad:
        print("Max Misadjustment: \t", max_misadjustment)
    return max_misadjustment#np.max(temp)#, np.mean(temp)

def find_converged_point(aee, fs, conv_div=20, bufsize=100, print_point=True):
    means = []
    for i in range(len(aee)-bufsize):
        means.append(np.mean(aee[i:i+bufsize]))
    th = max(means)/conv_div
    for i in range(len(means)):
        if means[i]<th:
            if print_point:
                print("Converged Point: \t", (i+bufsize)/fs, "sec")
            return (i+bufsize)/fs
    if print_point:
        print("Not Converged!")
    return -1

def denoising_plots(t, e, s, fs, T, margin=100, conv_div=20, bufsize=100, type=None):
    aee = np.abs(s-e) # absolute estimation error
    max_e = max(e)
    max_aee = max(aee)
    max_t = max(t)
    
    conv_point = find_converged_point(aee, fs, conv_div=conv_div, bufsize=bufsize)
        
    fig, axes = plt.subplots(2, 1, figsize=(16,12))
    fig.tight_layout(h_pad=5.0)
    axes[0].plot(t, s, linewidth = 2.0, label = 'source')
    axes[0].plot(t, e, linewidth = 1.0, linestyle = '--', label = 'estimated')
    axes[0].legend()
    axes[0].set_title("Source and Estimated Sound Signal")
    axes[0].set_xlabel("Time (sec)")

    
    axes[1].plot(t, aee)
    axes[1].set_xlabel("Time (sec)")
    axes[1].set_title("Absolute Estimation Error: |e[n]-s[n]|")
    
    if conv_point!=-1:
        max_misadjustment = find_misadjustment(aee, conv_point, fs, margin)
        axes[0].axvline(x=conv_point, linestyle='--', color="r")
        axes[0].text(conv_point, -.07, 'Convergence\nPoint', color='red', transform=axes[0].get_xaxis_transform(), ha='center', va='top', fontweight='bold', fontsize='large')
    
        axes[1].axvline(x=conv_point, linestyle='--', color="r")
        axes[1].text(conv_point, -.07, 'Convergence\nPoint', color='red', transform=axes[1].get_xaxis_transform(), ha='center', va='top', fontweight='bold', fontsize='large')
        axes[1].hlines(y=0, xmin=conv_point, xmax=T, linestyle="--", color="red")
        axes[1].hlines(y=max_misadjustment, xmin=conv_point, xmax=T, linestyle="--", color="red")
        myArrow = FancyArrowPatch(posA=(max_t, 0), posB=(max_t, max_misadjustment), arrowstyle='<|-|>', color='red', mutation_scale=20, shrinkA=0, shrinkB=0)
        axes[1].add_artist(myArrow)
        axes[1].text(max_t, max_misadjustment+max_aee/30, "Misadjustment", color="red", ha='right', fontweight='bold', fontsize='large')
    
    else:
        axes[0].text(max_t, max_e*0.75, 'Not Converged!', color='red', ha='right', va='top', fontweight='bold', fontsize='large')
        axes[1].text(max_t, max_aee*0.75, 'Not Converged!', color='red', ha='right', va='bottom', fontweight='bold', fontsize='large')
    plt.show()
    
def denoising_plots_across_params(t, s, x, d, fs, T, normalized=False, a=0.0, lms_type="standard", Ks=None, mus=None, margin=100, conv_div=30, bufsize=100, beta=None):
    params = None
    param_name = " "
    if Ks is not None:
        params = Ks
        param_name = "K"
    elif mus is not None:
        params = mus
        param_name = "mu"
    if params is None:
        raise ValueError("Only one parameter must be given. Either mu or K.")
    
    nparams = len(params)
    
    fig, axes = plt.subplots(2, nparams, figsize=(16,12))
    fig.tight_layout(h_pad=10.0)
    
    for ind, param in enumerate(params):
        if param_name=="K":
            mu=0.001
            K=param
        else:
            mu=param
            K=5
            
        e = sb_lms(x, d, mu=mu, K=K, normalized=normalized, a=a, type=lms_type, beta=beta)
        
        aee = np.abs(s-e) # absolute estimation error
        max_e = max(e)
        max_aee = max(aee)
        max_t = max(t)
        
        conv_point = find_converged_point(aee, fs, conv_div=conv_div, bufsize=bufsize)
        
        axes[0, ind].plot(t, s, linewidth = 2.0, label = 'source')
        axes[0, ind].plot(t, e, linewidth = 1.0, linestyle = '--', label = 'estimated')
        axes[0, ind].legend()
        axes[0, ind].set_title("Source and Estimated \nSound Signal with "+param_name+"="+str(param))
        axes[0, ind].set_xlabel("Time (sec)")


        axes[1, ind].plot(t, aee)
        axes[1, ind].set_xlabel("Time (sec)")
        axes[1, ind].set_title("Absolute Estimation Error: \n|e[n]-s[n]| with "+param_name+"="+str(param))
        
        if conv_point!=-1:
            max_misadjustment = find_misadjustment(aee, conv_point, fs, margin)
            axes[0, ind].axvline(x=conv_point, linestyle='--', color="r")
            axes[0, ind].text(conv_point, -.1, 'Convergence\nPoint', color='red', transform=axes[0, ind].get_xaxis_transform(), ha='center', va='top', fontweight='bold', fontsize='large')

            axes[1, ind].axvline(x=conv_point, linestyle='--', color="r")
            axes[1, ind].text(conv_point, -.1, 'Convergence\nPoint', color='red', transform=axes[1, ind].get_xaxis_transform(), ha='center', va='top', fontweight='bold', fontsize='large')
            axes[1, ind].hlines(y=0, xmin=conv_point, xmax=T, linestyle="--", color="red")
            axes[1, ind].hlines(y=max_misadjustment, xmin=conv_point, xmax=T, linestyle="--", color="red")
            myArrow = FancyArrowPatch(posA=(max_t, 0), posB=(max_t, max_misadjustment), arrowstyle='<|-|>', color='red', mutation_scale=20, shrinkA=0, shrinkB=0)
            axes[1, ind].add_artist(myArrow)
            axes[1, ind].text(max_t, max_misadjustment+max_aee/30, "Misadjustment", color="red", ha='right', fontweight='bold', fontsize='large')

        else:
            axes[0, ind].text(max_t/2, max_e, 'Not Converged!', color='red', ha='center', va='top', fontweight='bold', fontsize='large')
            axes[1, ind].text(max_t/2, max_aee, 'Not Converged!', color='red', ha='center', va='top', fontweight='bold', fontsize='large')
    plt.show()
    
    
    
def compare_with_params(f, T, fs, mus=None, Ks=None, normalized=False, a=0.0, type="standard", margin=100, conv_div=20, bufsize=100):
    params = None
    param_name = " "
    if Ks is not None:
        params = Ks
        param_name = "K"
    elif mus is not None:
        params = mus
        param_name = "mu"
    if params is None:
        raise ValueError("Only one parameter must be given. Either mu or K.")
    
    nparams = len(params)
    
    t, s, x, d = generate_signals(f, T, fs=fs, h=np.array([-0.86101471, 0.39594833, -1.04287894, -1.00737516, -0.331485]), plot=False)
    
    conv_points = np.zeros((nparams,))
    max_misads = np.zeros((nparams,))
    
    for ind in tqdm(range(nparams), desc="Progress"):
        if param_name=='K':
            K=params[ind]
            mu=0.001
        else:
            K=5
            mu=params[ind]
                
        e = sb_lms(x, d, mu=mu, K=K, normalized=False, a=0.0, type="standard")
        aee = np.abs(s-e)
        conv_points[ind] = find_converged_point(aee, fs, conv_div=conv_div, bufsize=bufsize, print_point=False)
        max_misads[ind] = find_misadjustment(aee, conv_points[ind], fs, margin, print_misad=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(16,6))
    ax.plot(params, conv_points, linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(params, max_misads, 'r', linewidth=2)
    if param_name=="K":
        xlab = "Adaptive Filter Order, K"
        tit = "Convergence Points and Max Misadjustments vs Adaptive Filter Order K"
        opt = 5
        text_posx = 5.2
        text_posy = (max(conv_points)+min(conv_points))/2
        text_desc = "Optimum K=5"
        ax.text(4, text_posy, "Not Converged Region!", color='black', ha='center', va='center', fontweight='heavy', fontsize='x-large', rotation=90)
    else:
        xlab = "Step Size, mu"
        tit = "Convergence Points and Max Misadjustments vs Step Sizes"
        opt = 0.0015
        text_posx = 0.0016
        text_posy = max(conv_points)/2
        text_desc = 'Optimum mu=1.5e-3'
        ax.set_xscale('log')
    ax.axvline(x=opt, linestyle='--', color="k", linewidth=2)
    ax.text(text_posx, text_posy, text_desc, color='black', ha='left', va='bottom', fontweight='bold', fontsize='large')
    ax.set_xlabel(xlab)
    ax.set_ylabel("convergence points", color='tab:blue')
    ax2.set_ylabel("max misadjustments", color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.set_title(tit)
    plt.show()


