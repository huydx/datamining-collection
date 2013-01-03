import pyaudio as pa
import scipy as sp
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

wave_l = None
wave_r = None
fft_wave_l = None
fft_wave_r = None
csp = None
degree = 0
degree_n = 0

def input():
  global wave_l, wave_r
  wave_l = sp.loadtxt('left.dat', usecols=(1,), unpack=True)
  wave_r = sp.loadtxt('right.dat', usecols=(1,), unpack=True)
 
def _unittest():
  #print fft_wave_l
  #print fft_wave_r
  print csp
  print degree
  print degree_n
  
if __name__ == '__main__':
  input()
  fft_wave_l = fft(wave_l) #fourier transform of left wave
  fft_wave_r = fft(wave_r) #fourier transform of right wave
  
  sz_l = fft_wave_l.shape[0]
  sz_r = fft_wave_r.shape[0]
  assert sz_l == sz_r 
  
  csp = sp.zeros(sz_l)

  for i in range(fft_wave_l.shape[0]):
    dividend = fft_wave_l[i] * (fft_wave_r[i].conj())
    divisor = abs(fft_wave_l[i]) * abs(fft_wave_r[i])
    csp[i] = dividend / divisor ##calculate Cross-power Spectrum Phase analysis

  csp = ifft(csp)
  a_csp = sp.apply_along_axis(lambda x: abs(x), 0, csp)
  max = a_csp.max(0)/16000 ##max of argument

  degree = sp.arccos(max*34000/10) ##34000: speed of sound wave, 10: distance of microphone array

  ##find noise direction
  ##delete 2 max value
  max_idx = a_csp.argmax(0)
  a_csp[max_idx] = 0
  max_idx = a_csp.argmax(0)
  a_csp[max_idx] = 0
  
  #find second max value 
  max_n = a_csp.max(0)/16000 #noise time interval
  max_n_idx = a_csp.argmax(0)
  degree_n = sp.arccos(max_n*34000/10)

  ##shift left wave to max_n_idx to normalize
  noise_nml_wave_l = sp.roll(wave_l, max_n_idx, 0)   
  for i in range(max_n_idx):
    noise_nml_wave_l[i] = 0

  ##substract right wave
  substract_wave = noise_nml_wave_l
  for i in range(sz_l):
    substract_wave[i] = wave_r[i] + noise_nml_wave_l[i]

  time_axis = sp.loadtxt('left.dat', usecols=(0,), unpack=True)
  output = sp.zeros((2, sz_l))
  output[0] = time_axis
  output[1] = noise_nml_wave_l
  output = sp.transpose(output)

  sp.savetxt('output.dat', output, fmt='%.10f', delimiter='\t', newline='\n')

  #_unittest()
  
  plotdt = sp.zeros((2, sz_l))
  plotdt[0] = sp.loadtxt("left.dat", usecols=(0,), unpack=True)
  plotdt[1] = csp
  
  plt.plot(plotdt[0], plotdt[1], color='r', linewidth=0.5)
  plt.show()
