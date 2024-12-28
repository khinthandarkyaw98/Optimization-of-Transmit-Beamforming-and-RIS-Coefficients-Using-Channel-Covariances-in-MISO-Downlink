"""
Compare the sum rate between the neural network and the Zero-forcing
Author    : Khin Thandar Kyaw
Date : 21 OCT 2023
Last Modified  : 15 Nov 2023
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from nn_utils import *


# tf_version: 2.15.0
print(tf.__version__)
print("Loading...")

# --------------------------- Start --------------------------------
snr_fixed = fixed_snr()
Nt, N, _, _, _, _, _ = parameters(6) # 6 is just a placeholder

global_ymin = 0
global_ymax = 70

M1 = 3
K1 = 3

M2 = 5
K2 = 5


###################
# 6 users
###################
# load the data
rate_NN_unsuper_6 = np.load('Plotting/6users/sumRateSuper.npy')
rate_WF_6 = np.load('Plotting/6users/sumRateWF.npy')
rate_NN_unsuper_ADMM_6 = np.load('Plotting/6users/sumRateSuper_ADMM.npy')

x_new = snr_fixed


###################
# 10 users
###################
rate_NN_unsuper_10 = np.load('Plotting/10users/sumRateSuper.npy')
rate_WF_10 = np.load('Plotting/10users/sumRateWF.npy')
rate_NN_unsuper_ADMM_10 = np.load('Plotting/10users/sumRateSuper_ADMM.npy')


print('Loading...')
print_line()

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.figure(figsize=(7, 6))


# Plot lines
plot_line(rate_NN_unsuper_6, r'BNN + random RIS $(M = 3)$' , 'dashed', 'blue', 's')
plot_line(rate_WF_6, r'ZF beam w/ WF pwr + random RIS $(M = 3)$', 'dotted', 'blue', 's')
plot_line(rate_NN_unsuper_ADMM_6, r'BNN + RIS CNN $(M = 3)$', 'solid', 'blue', 's')


ellipse1 = Ellipse((20, 45), width=2, height=10, edgecolor='grey', facecolor='none', linestyle='--', linewidth=1.5)
plt.gca().add_patch(ellipse1)

plt.annotate(r'$M$ = {}, $K$ = {}'.format(M1, K1), xy=(20, 40), xytext=(15, 30),
             arrowprops=dict(facecolor='grey', shrink=0.05, width=1, headwidth=6),
             fontsize=14, ha='left')

ellipse2 = Ellipse((20, 60), width=2, height=10, edgecolor='grey', facecolor='none', linestyle='solid', linewidth=1.5)
plt.gca().add_patch(ellipse2)

plt.annotate(r'$M$ = {}, $K$ = {}'.format(M2, K2), xy=(19, 62), xytext=(22, 67),
             arrowprops=dict(facecolor='grey', shrink=0.05, width=1, headwidth=6),
             fontsize=14, ha='right')

# Legend
plt.legend(loc='upper left', ncol=1, fontsize=14)
plt.ylim([global_ymin, global_ymax])

# Axes labels
plt.rc('text', usetex=True)
plt.xlabel(r'$P_{\mathrm{T}}/\sigma_n^2$ (dB)', fontsize=14)
plt.ylabel('Approximate sum rate (bps/Hz)', fontsize=14)

# Title
plt.title(r'$N_t$ = {}, $N$ = {}'.format(Nt, N), fontsize=14)

plt.grid(True) 
plt.tight_layout()  # Adjust layout to prevent clipping of legend
#plt.savefig('Plotting/fig2.tiff')
plt.savefig('Plotting/fig3.png', dpi=500)  
plt.savefig('Plotting/fig3.eps', format='eps', dpi=500)
plt.close()

print("Done!")  