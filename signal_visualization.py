import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

fs=2000.; np.random.seed(77); N=int(fs*10); t=np.arange(N)/fs
# EMG sintetis: bandpass broadband + burst pattern
sos_emg=sig.butter(4,[20.,500.],btype='bandpass',fs=fs,output='sos')

emg_base=sig.sosfilt(sos_emg, np.random.randn(N))*0.5

burst_mask=np.zeros(N)
for st in [500,1500,2500,4000,5000,6500,8000,9000]: # 8 burst @ 500-600ms
 burst_mask[st:st+600]=1
x_signal = emg_base * (1 + 3*burst_mask) # amplitudo 4× lebih tinggi saat burst
x_noise = (0.80*np.sin(2*np.pi*50*t) + # PLN 50Hz
 0.30*np.sin(2*np.pi*100*t) + # PLN 100Hz
 sig.sosfilt(sig.butter(4,20./(fs/2),btype='low',output='sos'),
 0.40*np.random.randn(N)) + # motion artefak
 0.10*np.random.randn(N)) # elektronik
x_raw = x_signal + x_noise

# add visualization

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x_raw)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Raw EMG Signal')
plt.grid(True)

plt.show()