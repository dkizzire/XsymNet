import numpy
import os    

cwd = os.getcwd()
xy_file = os.path.join(cwd, 'bso', 'hrp27188_bs_alpha_neutron_60-120.xy')
new_xy_file = os.path.join(cwd, 'bso', 'hrp27188_bs_alpha_neutron_60-120_trim.xy')
data = numpy.loadtxt(xy_file)
x = data[:,0]
y = data[:,1]

with open(new_xy_file, 'w') as f:
    for i in range(len(x)):
        f.write(f'{x[i]:.3f} {y[i]:.6f}\n')
