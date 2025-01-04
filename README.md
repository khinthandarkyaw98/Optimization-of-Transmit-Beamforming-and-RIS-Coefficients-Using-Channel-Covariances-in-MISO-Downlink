# Neural Network Based Optimization of Transmit Beamforming and RIS Coefficients Using Channel Covariances in MISO Downlink

<b>Authors</b>: Khin Thandar Kyaw, Wiroonsak Santipacha, Kritsada Mamat, Kamol Kaemarungsi, Kazuhiko Fukawa, Lunchakorn Wuttisittikulkij

## Citation

```
@article{KYAW2025155656,
title = {Neural network based optimization of transmit beamforming and RIS coefficients using channel covariances in MISO downlink},
journal = {AEU - International Journal of Electronics and Communications},
volume = {191},
pages = {155656},
year = {2025},
issn = {1434-8411},
doi = {https://doi.org/10.1016/j.aeue.2024.155656},
url = {https://www.sciencedirect.com/science/article/pii/S1434841124005429},
author = {Khin Thandar Kyaw and Wiroonsak Santipach and Kritsada Mamat and Kamol Kaemarungsi and Kazuhiko Fukawa and Lunchakorn Wuttisittikulkij},
keywords = {Beamforming, Optimization, Downlink, RIS, Channel covariance, MISO, Neural network, Unsupervised learning, Supervised learning},
}
```

---
<div align="justify">
We propose an <b> unsupervised </b> beamforming neural network (BNN) and a <b> supervised </b> reconfigurable intelligent surface (RIS) convolutional neural network (CNN) to optimize transmit beamforming and RIS coefficients of multi-input single-output (MISO) downlink with RIS assistance. To avoid frequent beam updates, the proposed BNN and RIS CNN are based on slow-changing channel covariances and are different from most other neural networks that utilize channel instances. Numerical simulations show that for a small or moderate signal-to-noise ratio (SNR), the proposed BNN with RIS CNN can achieve a sum rate close to that of a system with optimal beams and RIS coefficients. Furthermore, the proposed scheme significantly reduces the computation time.
</div>

---

## System Model
<div align="center">
    <img src="./Plots/fig1.png" alt="system model figure">
</div>


## Implementation
Please refer the paper for implementation details.

## Numerical Results
<div align="center">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/fig2.png" alt="fig2" style="width:400px; height:350px">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/fig3.png" alt="fig3" style="width:400px; height:350px">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/fig4.png" alt="fig4" style="width:400px; height:350px">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/fig5.png" alt="fig5" style="width:400px; height:350px">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/fig6.png" alt="fig6" style="width:400px; height:350px">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/fig8.png" alt="fig8" style="width:400px; height:350px">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/fig9.png" alt="fig9">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/Bar_Time.png" alt="time-diff">
    <img src="https://github.com/khinthandarkyaw98/Optimization-of-Transmit-Beamforming-and-RIS-Coefficients-Using-Channel-Covariances-in-MISO-Downlink/blob/main/Plots/table.png" alt="table">
</div>