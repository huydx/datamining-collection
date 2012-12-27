import pyaudio as pa
import scipy as sp
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

wave_l = None
wave_r = None
fft_wave_l = None
fft_wave_r = None
csp = None

def input():
  global wave_l, wave_r
  wave_l = sp.loadtxt('left.dat', usecols=(1,), unpack=True)
  wave_r = sp.loadtxt('right.dat', usecols=(1,), unpack=True)
 
def _unittest():
  print fft_wave_l
  print fft_wave_r
  print csp

if __name__ == '__main__':
  input()
  fft_wave_l = fft(wave_l)
  fft_wave_r = fft(wave_r) 
  
  sz_l = fft_wave_l.shape[0]
  sz_r = fft_wave_r.shape[0]
  assert sz_l == sz_r 
  
  csp = sp.zeros(sz_l)

  for i in range(fft_wave_l.shape[0]):
    dividend = fft_wave_l[i] * (fft_wave_r[i].conj())
    divisor = abs(fft_wave_l[i]) * abs(fft_wave_r[i])
    csp[i] = dividend / divisor

  csp = ifft(csp)
  csp = sp.apply_along_axis(lambda x: abs(x), 0, csp)
  
  plotdt = sp.zeros((2, sz_l))
  plotdt[0] = sp.loadtxt("left.dat", usecols=(0,), unpack=True)
  plotdt[1] = csp
  
  plt.scatter(plotdt[0], plotdt[1], color='r')
  plt.show()
  #print t
  #_unittest()

