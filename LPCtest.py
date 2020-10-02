import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np 
y, sr = librosa.load(r"C:\Users\Lala\Desktop\clean_testset_wav\clean_testset_wav\p232_002.wav")
y = y[:15000]
y = y.reshape((10,1500))
a = librosa.lpc(y, 2)
b = np.hstack([[0], -1 * a[1:]])
#y_hat = scipy.signal.lfilter(b, [1], y)
y = y[:1500]
y_hat = np.zeros_like(y)
y_hat[0] = y[0]
y_hat[1] = y[1]
for  i in range(2,1500):
    y_hat[i] =   y_hat[i-1] * b[1] + y_hat[i-2] * b[2]
fig, ax = plt.subplots()
ax.plot(y)
ax.plot(y_hat, linestyle='--')
ax.legend(['y', 'y_hat'])
ax.set_title('LP Model Forward Prediction')
plt.show()