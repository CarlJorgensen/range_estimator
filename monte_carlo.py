#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def s1(t):
    return np.exp(-0.1 * (t ** 2))

def s2(t):
    return np.exp(-0.1 * (t ** 2))*np.cos(t)

def calc_const():
    tau = 0.1
    Trange = np.arange(-15, 15+tau, tau)
    signal1 = s1(Trange)
    signal2 = s2(Trange)

    E1 = np.sum(np.abs(signal1)**2)
    E2 = np.sum(np.abs(signal2)**2)

    a1 = 1/np.sqrt(E1)
    a2 = 1/np.sqrt(E2)

    return a1, a2


def plot_signal():
    # a1 = 0.252313
    # a2 = 0.501249

    a1 = 0.25
    a2 = 0.5

    t = np.linspace(-15, 15, 1000)
    signal1 = s1(t)
    signal2 = s2(t)

    plt.figure(figsize=(10, 6))
    plt.plot(t, signal1, 'r', label="s1(t)")
    plt.plot(t, signal2, 'g', label="s2(t)")
    plt.xlabel("t [s]", loc='right')  # Move x-axis label to the far right
    plt.ylabel("s_n(t)", loc='top')  
    plt.legend(loc="upper right")

    ax = plt.gca()
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True)
    plt.show()


def CRLB():

    a1, a2 = calc_const()
    tau = 0.1
    Trange = np.arange(-15, 15, tau)

    signal1 = a1 * s1(Trange)
    signal2 = a2 * s2(Trange)

    ds1 = np.gradient(signal1, tau)
    ds2 = np.gradient(signal2, tau)

    SNR_range = np.linspace(10, 30, 100)
    SNR = 10**(SNR_range/10)

    F1 = np.sum((ds1**2))
    F2 = np.sum((ds2**2))
    
    CRLB_1 = 1/(F1 * SNR)
    CRLB_2 = 1/(F2 * SNR)

    plt.loglog(SNR, np.sqrt(CRLB_1), 'r', label="sqrt(CRLB_1)")
    plt.loglog(SNR, np.sqrt(CRLB_2), 'g', label="sqrt(CRLB_2)")    
    plt.xlabel("SNR [dB]")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def maximum_likelihood(sigma: float, grid_size: float):
    """
    T_ML = argmax_{T} sum_{n} x[n*tau]s[n*tau-T]
    """

    a1, a2 = calc_const()
    tau = 0.1
    Trange = np.arange(-15, 15, tau)
    T_est_range = np.arange(-5, 5, grid_size)

    T = np.random.uniform(-5, 5)
    T1_ML = None
    T2_ML = None

    sum1 = -1000
    sum2 = -1000

    e = np.random.normal(loc=0, scale=sigma**2, size=len(Trange))

    x1 = a1 * s1(Trange - T) + e
    x2 = a2 * s2(Trange - T) + e

    signal1_matrix = a1 * s1(Trange[:, None] - T_est_range)
    signal2_matrix = a2 * s2(Trange[:, None] - T_est_range)

    sum1 = np.sum(x1[:, None] * signal1_matrix, axis=0)
    sum2 = np.sum(x2[:, None] * signal2_matrix, axis=0)

    T1_ML = T_est_range[np.argmax(sum1)]
    T2_ML = T_est_range[np.argmax(sum2)]



    # for T_est in T_est_range:
    #     signal1 = a1 * s1(Trange - T_est)
    #     signal2 = a2 * s2(Trange - T_est)

    #     curr_sum1 = np.sum(x1 * signal1)
    #     curr_sum2 = np.sum(x2 * signal2)

    #     if curr_sum1 > sum1:
    #         sum1 = curr_sum1
    #         T1_ML = T_est
        
    #     if curr_sum2 > sum2:
    #         sum2 = curr_sum2
    #         T2_ML = T_est

    #print(f"True T: {T} \n T1_ML: {T1_ML} \n T2_ML: {T2_ML}")

    return T1_ML, T2_ML, T

def monte_carlo():

    nr_iter = 10
    SNR_range = np.linspace(10, 30, 100)

    RMSE1 = np.zeros(len(SNR_range))
    RMSE2 = np.zeros(len(SNR_range))

    for i, SNR in tqdm(enumerate(SNR_range), desc="SNR"):
        T1_ML = np.zeros(nr_iter)
        T2_ML = np.zeros(nr_iter)
        T_true = np.zeros(nr_iter)
        for j in tqdm(range(nr_iter), desc="Iteration", leave=False):
            sigma = 1/(10**(SNR/10))
            T1, T2, T = maximum_likelihood(sigma, 0.1)
            T1_ML[j] = T1
            T2_ML[j] = T2
            T_true[j] = T
        
        RMSE1[i] = np.sqrt(((T1_ML - T_true)**2))
        RMSE1[i] = np.sqrt(((T1_ML - T_true)**2))
    
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_range, RMSE1, 'r', label="RMSE1")
    plt.plot(SNR_range, RMSE2, 'g', label="RMSE2")
    plt.show()


#maximum_likelihood(1, 0.001)
monte_carlo()






