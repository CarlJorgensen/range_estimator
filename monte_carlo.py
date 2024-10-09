#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def s1(t, const=0.25):
    return const*np.exp(-0.1 * (t ** 2))

def s2(t, const=0.5):
    return const*np.exp(-0.1 * (t ** 2))*np.cos(t)


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


def CRLB(T: float, nr_sample: int):
    a1 = 0.25
    a2 = 0.5
    tau = 0.1

    SNR_range = np.linspace(10, 50, 40)

    CRLB_1 = np.zeros(40)
    CRLB_2 = np.zeros(40)
    for i, SNR in enumerate(SNR_range):    
        denominator_1 = 0
        denominator_2 = 0
        for n in range(nr_sample):
            exp_term =  np.exp(-0.1 * ((n*tau - T) ** 2))
            ds1 = -0.2 * (n*tau - T) * a1 * exp_term
            ds2 = -0.2* a2 *  exp_term * ( (n*tau - T) * np.cos(n*tau-T) + 5*np.sin(n*tau - T) )
            denominator_1 += ds1**2
            denominator_2 += ds2**2

        std_dev = 1/SNR    
        CRLB_1[i] = std_dev/denominator_1
        CRLB_2[i] = std_dev/denominator_2
    
    # Should be log scale...

    SNR_range = np.log10(SNR_range)
    CRLB_1 = np.log10(CRLB_1)
    CRLB_2 = np.log10(CRLB_2)


    plt.figure(figsize=(10, 6))
    plt.plot(SNR_range, CRLB_1, 'r', label="CRLB_1")
    plt.plot(SNR_range, CRLB_2, 'g', label="CRLB_2")
    plt.xlabel("SNR [dB]")
    plt.ylabel("CRLB")  
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def maximum_likelihood(nr_samples: int):
    """
    T_ML = argmax_{T} sum_{n} x[n*tau]s[n*tau-T]
    """

    a1 = 0.25
    a2 = 0.5
    tau = 0.1

    T_est = []
    T_true = []
    
    for _ in range(nr_samples):
        T = np.random.uniform(-5, 5)
        sample_times = np.arrange(-15, 15, tau)
    return 0


plot_signal()


