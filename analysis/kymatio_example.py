from kymatio.torch import Scattering2D, Scattering1D
from trend import df_gap
import torch

scattering = Scattering1D(J=1, shape=(1000))
scattering.cuda()

x = torch.tensor(df_gap["y"].to_numpy()).float().cuda()

Sx = scattering(x)

print (x.shape, '==>', Sx.shape)
