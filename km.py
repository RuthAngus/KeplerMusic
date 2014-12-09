import h5py
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, freqz
from sklearn.linear_model import Lasso
from IPython.display import Audio

with h5py.File("dataset.h5", "r") as f:
    k = np.arange(f["lcs"].shape[0])
    np.random.seed(1234)
    np.random.shuffle(k)
    lc = f["lcs"][np.sort(k[:5000]), :]

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

for i in range(len(lc)):
    lc[i] = butter_lowpass_filter(lc[i], 1, 10.)

lc_comp = np.concatenate((np.zeros(10000), lc[800][10000:20000], np.zeros(10000), lc[800][20000:30000], np.zeros(10000)))
Audio(data=lc_comp, rate=44100)
rate, samps = wavfile.read("sample2.wav")

from sklearn.linear_model import Lasso
k = 10  # downsampling]
n = lc.shape[1]  # the number of data points in each light curve = 44100
# y = samps[44100*10 + n:, 0][::k]  # downsample
y = samps[:, 0][::k]  # downsample
X = np.concatenate((lc, np.ones((1, lc.shape[1]))))[:, ::k].T
print np.shape(X)

# from sklearn.ensemble import RandomForestRegressor
# model = Lasso(alpha=1e-2)
model = Lasso(alpha=1e-10)
model.fit(X, y)

p = model.predict(X)
pl.plot(y)
pl.plot(p, alpha=0.7)

Audio(data=p, rate=4410)

#k = 10  # downsampling]
#n = lc.shape[1]  # the number of data points in each light curve = 44100
## y = samps[44100*10 + n:, 0][::k]  # downsample
#y = samps[:, 0][::k]  # downsample
#X = np.concatenate((lc, np.ones((1, lc.shape[1]))))[:, ::k].T
rate, samps = wavfile.read("sample2.wav")
#Audio(data=samps[44100*0:44100*(1+1), 1], rate=rate)
for i in range(15):
    print i
    #Audio(data=samps[44100*i:44100*(i+1), 1], rate=rate)
    y = samps[44100*i:44100*(i+1), 0][::k]  # downsample
    X = np.concatenate((lc, np.ones((1, lc.shape[1]))))[:, ::k].T
    model = Lasso(alpha=1e-10)
    model.fit(X, y)
    p = model.predict(X)
    wavfile.write('%s2.wav' % i, rate, p.astype(np.int16))

rate, samps = wavfile.read("sample.wav")
a = range(-16, -1)
for i in range(15):
    print i
    y = samps[44100*i:44100*(i+1), 0][::k]  # downsample
    X = np.concatenate((lc, np.ones((1, lc.shape[1]))))[:, ::k].T
    print a[i]
    model = Lasso(alpha=10**a[i])
    model.fit(X, y)
    p = model.predict(X)
    wavfile.write('%salpha.wav' % i, rate, p.astype(np.int16))

k = 1000  # downsampling
n = lc.shape[1]  # the number of data points in each light curve = 44100
y = samps[44100*10 + n:, 0][::k]  # downsample
X = np.concatenate((lc, np.ones((1, lc.shape[1]))))[:, ::k].T
rate, samps = wavfile.read("sample.wav")

for i in range(15):
    print i
    y = samps[44100*i:44100*(i+1), 0][::k]  # downsample
    X = np.concatenate((lc, np.ones((1, lc.shape[1]))))[:, ::k].T
    model = Lasso(alpha=1e-10)
    model.fit(X, y)
    p = model.predict(X)
    wavfile.write('%s1000.wav' % i, rate, p.astype(np.int16))
