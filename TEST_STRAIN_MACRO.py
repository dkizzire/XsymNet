import os
import glob
import numpy as np
import math
import write_TOPAS_funcs as funcs

def straincell(pa0, pb0, pc0, pal0, pbe0, pga0):
    radian = 57.2957795
    palpha0 = pal0/radian
    pbeta0  = pbe0/radian
    pgamma0 = pga0/radian
    # parent-lattice basis vectors in cartesian coords
    zzz  = (math.cos(palpha0)- math.cos(pbeta0)*math.cos(pgamma0))/math.sin(pgamma0)
    a0x = pa0
    b0x = pb0*math.cos(pgamma0)
    b0y = pb0*math.sin(pgamma0)
    c0x = pc0*math.cos(pbeta0)
    c0y = pc0*zzz
    c0z = pc0*math.sqrt(1 - math.cos(pbeta0)**2 - zzz**2)

    bv = [a0x, b0x, b0y, c0x, c0y, c0z]

    return bv

def celltransform(s_mat, p_mat, bv, tm):

    pe1, pe2, pe3, pe4, pe5, pe6 = p_mat[0], p_mat[1], p_mat[2], p_mat[3], p_mat[4], p_mat[5]

    a0x, b0x, b0y, c0x, c0y, c0z = bv[0], bv[1], bv[2], bv[3], bv[4], bv[5]

    t11, t12, t13, t21, t22, t23, t31, t32, t33 = tm[0], tm[1], tm[2], tm[3], tm[4], tm[5], tm[6], tm[7], tm[8]  

    radian = 57.2957795
    #parent-lattice basis vectors in cartesian coords
    pv11 = a0x*(pe1 + 1)
    pv21 = a0x*(pe6/2)
    pv31 = a0x*(pe5/2)
    pv12 = b0x*(pe1 + 1) + b0y*(pe6/2)
    pv22 = b0x*(pe6/2)   + b0y*(pe2 + 1)
    pv32 = b0x*(pe5/2)   + b0y*(pe4/2)
    pv13 = c0x*(pe1 + 1) + c0y*(pe6/2) + c0z*(pe5/2)
    pv23 = c0x*(pe6/2)+ c0y*(pe2 + 1) + c0z*(pe4/2)
    pv33 = c0x*(pe5/2)   + c0y*(pe4/2) + c0z*(pe3 + 1)
    #supercell basis vectors in cartesian coords
    sv11 = (t11)*pv11 + (t12)*pv12 + (t13)*pv13
    sv21 = (t11)*pv21 + (t12)*pv22 + (t13)*pv23
    sv31 = (t11)*pv31 + (t12)*pv32 + (t13)*pv33
    sv12 = (t21)*pv11 + (t22)*pv12 + (t23)*pv13
    sv22 = (t21)*pv21 + (t22)*pv22 + (t23)*pv23
    sv32 = (t21)*pv31 + (t22)*pv32 + (t23)*pv33
    sv13 = (t31)*pv11 + (t32)*pv12 + (t33)*pv13
    sv23 = (t31)*pv21 + (t32)*pv22 + (t33)*pv23
    sv33 = (t31)*pv31 + (t32)*pv32 + (t33)*pv33
    #distorted supercell parameters
    sa = math.sqrt(sv11**2+sv21**2+sv31**2)
    sb = math.sqrt(sv12**2+sv22**2+sv32**2)
    sc = math.sqrt(sv13**2+sv23**2+sv33**2)
    salpha = math.acos((sv12*sv13+sv22*sv23+sv32*sv33)/(sb*sc))
    sbeta  = math.acos((sv11*sv13+sv21*sv23+sv31*sv33)/(sa*sc))
    sgamma = math.acos((sv11*sv12+sv21*sv22+sv31*sv32)/(sa*sb))
    sal = salpha*radian
    sbe = sbeta*radian
    sga = sgamma*radian

    # print()
    # print(f's_vars = [{s1:.4f}, {s2:.4f}, {s3:.4f}, {s4:.4f}, {s5:.4f}, {s6:.4f}]')
    # print(f'a = {sa}')
    # print(f'b = {sb}')
    # print(f'c = {sc}')
    # print(f'al = {sal}')
    # print(f'be = {sbe}')
    # print(f'ga = {sga}')
    # print()

    return sa, sb, sc, sal, sbe, sga

lsa, lsb, lsc, lsal, lsbe, lsga = [], [], [], [], [], []

for i in range(10000):
    random_strain = np.random.uniform(low=-0.01, high=0.01, size=(6))

    s1 = random_strain[0]
    s2 = random_strain[1]
    s3 = random_strain[2]
    s4 = random_strain[3]
    s5 = random_strain[4]
    s6 = random_strain[5]

    # s1 = 0.1
    # s2 = -0.1
    # s3 = 0.1
    # s4 = 0.1
    # s5 = 0.1
    # s6 = 0.1

    # P1 
    # pe1 = 0.57735*s1 -  0.40825*s2 +  0.70711*s3
    # pe2 = 0.57735*s1 -  0.40825*s2 +  0.70711*s3
    # pe3 = 0.57735*s1 +  0.81650*s2
    # pe4 = 1.41421*s5
    # pe5 = 1.41421*s6
    # pe6 = 1.41421*s4
    # basis = [1,1,0,1,-1,0,0,0,-2]

    # Parent 
    # pe1 = 0.57735*s1
    # pe2 = 0.57735*s1
    # pe3 = 0.57735*s1
    # pe4 = 0
    # pe5 = 0
    # pe6 = 0
    # basis = [0,-1,0,-1,0,0,0,0,-1]

    # Beta 
    pe1 = 0.57735*s1 +  0.81650*s2
    pe2 = 0.57735*s1 -  0.40825*s2
    pe3 = 0.57735*s1 -  0.40825*s2
    pe4 = 1.41421*s3
    pe5 = 0
    pe6 = 0
    basis = [0,-0.5,0.5,-2,0,0,0,-1,-1]

    s_mat = [s1, s2, s3, s4, s5, s6]
    p_mat = [pe1, pe2, pe3, pe4, pe5, pe6]

    bv = straincell(10.72250,10.72250,10.72250,90.00000,90.00000,90.00000)

    sa, sb, sc, sal, sbe, sga = celltransform(s_mat, p_mat, bv, basis)
    lsa.append(sa)
    lsb.append(sb)
    lsc.append(sc)
    lsal.append(sal)
    lsbe.append(sbe)
    lsga.append(sga)

print('\nMin and Max of Values')
print(f'a = {min(lsa):.4f},  {max(lsa):.4f}')
print(f'b = {min(lsb):.4f},  {max(lsb):.4f}')
print(f'c = {min(lsc):.4f},  {max(lsc):.4f}')
print(f'al = {min(lsal):.4f},  {max(lsal):.4f}')
print(f'be = {min(lsbe):.4f},  {max(lsbe):.4f}')
print(f'ga = {min(lsga):.4f},  {max(lsga):.4f}')
print()

print(f'{7.58195*.999}, {7.58195*1.001}')
print(f'{21.44500*.999}, {21.44500*1.001}')
print(f'{15.16390*.999}, {15.16390*1.001}')



strs = sorted(glob.glob('./TOPAS/topas_strs_strain/*.str'))
cif_filenames = [os.path.basename(x) for x in strs]
cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
subgroup_nums = [x[5:] for x in cif_noext]

param_counts = {}
for i in range(len(strs)):
    count = funcs.count_refinable_parameters_in_topas_strs(strs[i])
    param_counts[subgroup_nums[i]] = count

funcs.save_dict_to_json(param_counts, './subgroup_refinable_parameter_counts.json')
