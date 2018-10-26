import numpy as np
import matplotlib.pyplot as plt
#import converted as TOCM
##basedir='../../../catherine/21cmStatistics/code/21cmFAST/Boxes/'
#basedir='21cmFAST-master\Boxes'
#myrun=TOCM.runio.loadslices(basedir,256,300)
#myrun.info()

filename = '21cmFAST-master/Boxes/delta_T_v3_no_halos_z006.80_nf0.651158_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb011.88_Pop-1_256_300Mpc'
#filename = '21cmFAST-master/Boxes/delta_T_v3_no_halos_z012.00_nf0.975151_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb030.04_Pop-1_256_300Mpc'

dtype='f'
fd=open(filename,'rb')
read_data=np.fromfile(fd,dtype)
fd.close()

print(len(read_data))
print(int(np.round(len(read_data)**(1./3))))

plt.hist(read_data, bins=200)
plt.show()

# %%


dim = int(np.round(len(read_data)**(1./3)))
row = []
rc = 0
layer = []
lc = 0
box = []
bc = 0

for i,n in enumerate(read_data):

    row.append(n)
    rc += 1

    if rc == dim:
        layer.append(row)
        lc += 1
        row = []
        rc = 0

        if lc == dim:
            box.append(layer)
            bc += 1
            layer = []
            lc = 0

print(bc)
#box is [layer,column,row]

# %%

plt.imshow(box[0], cmap='hot', interpolation='nearest')
plt.show()

# %%

for layer in range(dim):
    plt.imshow(box[layer], cmap='hot', interpolation='nearest')
    plt.savefig('image_outputs/' + str(layer) + '.png', format='png')
