import numpy
import math
import os



def tof2d(tof, tzero, difa, difc):
    if difa > 0:
        d = (-difc/(2*difa) + math.sqrt((difc/(2*difa))**2 + tof/difa - tzero/difa))
    elif difa < 0:
        d = (-difc/(2*difa) - math.sqrt((difc/(2*difa))**2 + tof/difa - tzero/difa))
    elif difa == 0:
        d = (tof - tzero) / difc
    return d

def d2tof(d, tzero, difa, difc):
    tof = (difa * d**2) + (difc * d) + tzero
    return tof

# read in xy file, find closest x to desired tof, return index
def find_closest_tof(xy_file, desired_tof, snip_name = '_snipped'):
    #remove comments and blank lines from xy file
    data_check = ('-','0','1','2','3','4','5','6','7','8','9')
    with open(xy_file, 'r') as f:    
        lines = f.readlines() 
        if not lines[0].startswith(data_check):
            lines = [line for line in lines if line.strip() and not line.startswith(data_check)]
            new_xy_file = (os.path.splitext(os.path.basename(xy_file))[0] + '_dataonly' + os.path.splitext(os.path.basename(xy_file))[1])
            new_xy_file = os.path.join(os.path.dirname(xy_file), new_xy_file)
            xy_file = new_xy_file
            with open(new_xy_file, 'w') as f:
                f.writelines(lines)
            print(f'Wrote new xy file without comments or blank lines: {new_xy_file}')

    data = numpy.loadtxt(xy_file)
    x = data[:,0]
    y = data[:,1]
    index = (numpy.abs(x - desired_tof)).argmin()
    # remove values before index
    snip_x = x[index:]
    snip_y = y[index:]
    snip_xy_file = new_xy_file.replace('_dataonly', snip_name)
    with open(snip_xy_file, 'w') as f:
        for i in range(len(snip_x)):
            f.write(f'{snip_x[i]:.3f} {snip_y[i]:.6f}\n')
    print(f'Wrote snipped xy file without comments or blank lines: {snip_xy_file}')
    return index, x[index], y[index]

tof = 60000

#* pzt 90
# tzero = 9.32399
# difa = -0.01750
# difc = 22571.96264

# #* bso beta
# tzero = -22.0105
# difa = -14.00371
# difc = 48283.52311

#* bso alpha
# tzero = -11.79112
# difa = -9.87666
# difc = 48255.09936

#* bso alpha - J.Evans Params
tzero = -22.0105
difa = -14.00371
difc = 48283.52311

#* bso beta - J.Evans Params
# tzero = 2.00496
# difa = -3.22266
# difc = 48255.09936

d = tof2d(tof, tzero, difa, difc)
print(f'\nTOF of {tof} us corresponds to d-spacing of {d:.6f} Ang\n')

desired_d = 1.0
snip_name = '_dmin_1p0'
desired_tof = d2tof(desired_d, tzero, difa, difc)

xy_file = os.path.abspath('./bso/Hrp27190_bs_NoComments.xye')
index, closest_tof, closest_y = find_closest_tof(xy_file, desired_tof, snip_name=snip_name)
print(f'For desired d-spacing of {desired_d} Ang, the corresponding tof is {desired_tof:.2f} us')
print(f'In the xy file, the closest tof is {closest_tof:.2f} us at index {index}, with intensity {closest_y}')


