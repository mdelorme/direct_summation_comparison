import sys, os
import subprocess

if '-N' in sys.argv:
    i = sys.argv.index('-N')
    N = sys.argv[i+1]
else:
    N = '1000'

if '--ic' in sys.argv:
    i = sys.argv.index('--ic')
    ic = sys.argv[i+1]
else:
    ic = 'data/ic_1000.dat'

if '-D' in sys.argv:
    i = sys.argv.index('-D')
    D = sys.argv[i+1]
else:
    D = '20'

if '--dt' in sys.argv:
    i = sys.argv.index('--dt')
    dt = sys.argv[i+1]
else:
    dt = '1.0e-4'

bins = ['01_serial', '02_kokkos_1d_vec']

ic = os.path.join('..', ic)

for b in bins:
    print('Running', b)
    os.chdir(b)
    bin_file = os.path.join('./', b)
    subprocess.call([bin_file, '-N', N, '-D', D, '--ic', ic, '--dt', dt])
    os.chdir('..')
    
