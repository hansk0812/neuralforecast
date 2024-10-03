import pywt

from trend import df_gap

from matplotlib import pyplot as plt

print (pywt.families())
print (pywt.families(short=False))

wavelet = pywt.Wavelet("haar")
(cA, cD) = pywt.dwt(df_gap["y"], wavelet, mode='zero', axis=-1)

fig, ax = plt.subplots(3)
ax[0].plot(df_gap["y"].to_numpy())
ax[1].plot(cA)
ax[2].plot(cD)
plt.show()
print (cA.shape, cD.shape)
