'''
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from timeit import default_timer as timer
 
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void func(float *a, float *b, size_t N)
{
 const int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= N)
 {
  return;
 }
 float temp_a = a[i];
 float temp_b = b[i];
 a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
 // a[i] = a[i] + b[i];
}
""")
 
func = mod.get_function("func")
 
def test(N):
  # N = 1024 * 1024 * 90  # float: 4M = 1024 * 1024
 
  print("N = %d" % N)
 
  N = np.int32(N)
 
  a = np.random.randn(N).astype(np.float32)
  b = np.random.randn(N).astype(np.float32)
  # copy a to aa
  aa = np.empty_like(a)
  aa[:] = a
  # GPU run
  nTheads = 256
  nBlocks = int( ( N + nTheads - 1 ) / nTheads )
  start = timer()
  func(
      drv.InOut(a), drv.In(b), N,
      block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
  run_time = timer() - start
  print("gpu run time %f seconds " % run_time)
  # cpu run
  start = timer()
  aa = (aa * 10 + 2 ) * ((b + 2) * 10 - 5 ) * 5
  run_time = timer() - start
 
  print("cpu run time %f seconds " % run_time)
 
  # check result
  r = a - aa
  print( min(r), max(r) )
 
def main():
 for n in range(1, 10):
  N = 1024 * 1024 * (n * 10)
  print("------------%d---------------" % n)
  test(N)
 
if __name__ == '__main__':
  main()

'''


import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import zepid
from zepid.graphics import EffectMeasurePlot

labs = ["ACA(Isq=41.37% Tausq=0.146 pvalue=0.039 )",
        "ICA0(Isq=25.75% Tausq=0.092 pvalue=0.16 )",
        "ICA1(Isq=60.34% Tausq=0.121 pvalue=0.00 )",
        "ICAb(Isq=25.94% Tausq=0.083 pvalue=0.16 )",
        "ICAw(Isq=74.22% Tausq=0.465 pvalue=0.00 )"]
measure = [2.09,2.24,1.79,2.71,1.97]
lower = [1.49,1.63,1.33,2.00,1.25]
upper = [2.92,3.07,2.42,3.66,3.11]
p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
p.labels(effectmeasure='RR')
p.colors(pointshape="D")
ax=p.plot(figsize=(7,3), t_adjuster=0.09, max_value=4, min_value=0.35 )
plt.title("Random Effect Model(Risk Ratio)",loc="right",x=1, y=1.045)
plt.suptitle("Missing Data Imputation Method",x=-0.1,y=0.98)
ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(False)
plt.savefig("Missing Data Imputation Method",bbox_inches='tight')
