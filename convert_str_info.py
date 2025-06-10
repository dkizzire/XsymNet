import os


parent_cell = 'straincell(5.25850,5.25850,22.71890,90.00000,90.00000,90.00000)'
transform =   'celltransform(0,-1,0,1,0,0,0,0,1)'

parent_cell = parent_cell.replace('straincell', '').replace('(', '').replace(')', '')
transform = transform.replace('celltransform', '').replace('(', '').replace(')', '')
# extract numbers from above strings into a list
def extract_numbers(s):
    return [float(x) for x in s.split(',') if x.strip()]


cell = extract_numbers(parent_cell)
matrix = extract_numbers(transform)

print(f'prm !pa0 = {cell[0]};')
print(f'prm !pb0 = {cell[1]};')
print(f'prm !pc0 = {cell[2]};')
print(f'prm !pal0 = {cell[3]};')
print(f'prm !pbe0 = {cell[4]};')
print(f'prm !pga0 = {cell[5]};')
print()
print(f'prm !t11 = {matrix[0]};')
print(f'prm !t12 = {matrix[1]};')
print(f'prm !t13 = {matrix[2]};')
print(f'prm !t21 = {matrix[3]};')
print(f'prm !t22 = {matrix[4]};')
print(f'prm !t23 = {matrix[5]};')
print(f'prm !t31 = {matrix[6]};')
print(f'prm !t32 = {matrix[7]};')
print(f'prm !t33 = {matrix[8]};')
print()
print('straincell(pa0, pb0, pc0, pal0, pbe0, pga0)')
print('celltransform(t11, t12, t13, t21, t22, t23, t31, t32, t33)')
print()
