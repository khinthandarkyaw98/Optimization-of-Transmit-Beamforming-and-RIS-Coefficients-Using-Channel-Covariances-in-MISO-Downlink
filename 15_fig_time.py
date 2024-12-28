"""
Plot the time difference
Author    : Khin Thandar Kyaw
Last Modified  : 28 Dec 2024
"""

import numpy as np

# N = 30

optimal_time_100 = np.load('test/6users/timeArrayOptimalSolution_coral_2_plot_7.npy')
# print(nn_time)
optimal_time_100 = np.sum(optimal_time_100)
print(f"NN time: {optimal_time_100}")


bnn_ris_cnn_time_100 = np.load('test/6users/timeArraySuper_plot_7.npy')
# print(zwf_time)
# print(len(zwf_time))
bnn_ris_cnn_time_100 = np.sum(bnn_ris_cnn_time_100)
print(f"ZWF time: {bnn_ris_cnn_time_100}")

import matplotlib.pyplot as plt

# Data for plotting
methods = ['Optimal BS beams \n+ ADMM RIS', 'BNN \n+ RIS CNN']
times = [optimal_time_100, bnn_ris_cnn_time_100]

# Bar width and positions
bar_width = 0.35
positions = range(len(methods))

# Create the figure and axis
plt.figure(figsize=(15, 7))
ax = plt.gca()

# Draw the grid first as separate lines
for y in np.logspace(np.log10(min(times) * 0.1), np.log10(max(times) * 10), num=10):
    ax.axhline(y, color='gray', linestyle='--', linewidth=0.5, zorder=1)
    
# Set y-axis to log scale
plt.yscale('log')

# Plot the bars with no axis grid interaction
bars = plt.bar(positions, times, width=0.5, color=['deepskyblue', 'slateblue'], hatch=['X', '+'], zorder=3)

# Add data labels to each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, zorder=4)


# Add titles and labels
plt.ylabel('Computation time (seconds)', fontsize=14)
plt.xticks(positions, methods, fontsize=12)

# Add legend
plt.legend(fontsize=14)
plt.savefig('Plotting/6usersBar_Time.png', dpi=500)
plt.savefig('Plotting/6users/Bar_Time.eps', format='eps', bbox_inches='tight', dpi=500)
plt.close()

print("Done!")





