import numpy as np
import matplotlib.pyplot as plt
import os, sys, shutil

path = sys.argv[-1]
file_list = os.listdir(path)
file_list.sort()

render_path = os.path.join(path, 'render')
if os.path.exists(render_path):
    shutil.rmtree(render_path)
os.mkdir(render_path)

for f in file_list:
    if not f.endswith('.dat'):
        continue
    
    print('Rendering ', f)
    fpath = os.path.join(path, f)
    dat = np.loadtxt(fpath)

    plt.close('all')
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(dat[:,0], dat[:,1], s=1)
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.3, 1.3)

    outputf = os.path.splitext(f)[0] + '.png'
    path_out = os.path.join(render_path, outputf)
    plt.savefig(path_out)
    
    
