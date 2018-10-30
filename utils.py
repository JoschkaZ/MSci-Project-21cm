import os
from PATH import get_path
import numpy as np
import matplotlib.pyplot as plt
from sys import platform

def read_box(filename, verbose=1):

    PATH = get_path()

    if platform == "darwin":
            boxpath = PATH + '/Boxes/' +  filename
    else:
            boxpath = PATH + '\\Boxes\\' +  filename
    dtype='f'
    fd=open(boxpath,'rb')
    read_data=np.fromfile(fd,dtype)
    fd.close()


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

    box = np.array(box)

    if verbose==1:
        print('read box @ ', boxpath)
        print('with dimensions ', box.shape)
        print('..........')
    return box

def show_box(box):
    plt.imshow(box[0], cmap='hot', interpolation='nearest')
    plt.show()
    return 1

def box_to_movie(box, verbose=1):

    PATH = get_path()
    savedirectory = PATH + '\\output_movie\\'
    if verbose==1: 'making movie'
    for layer in range(len(box)):
        print('saving frame #', layer)
        plt.imshow(box[layer], cmap='hot', interpolation='nearest')
        plt.savefig(savedirectory + str(layer) + '.png', format='png')

    print('..........')
    return 1

def change_parameter(parameter_name, new_value, verbose=1):

    PATH = get_path()

    #ZETA_X (double) (2.0e56) // 2e56 ~ 0.3 X-ray photons per stellar baryon

    if parameter_name == 'ZETA_X':
        filepath = PATH + '\\Parameter_files\\HEAT_PARAMS.H'

    else:
        print('WARNING - PARAMETER_NAME NOT RECOGNIZED!')
        return 0

    if verbose == 1:
        print('Changing paramter ', parameter_name, ' to ', value)


    # r'C:\21cmFAST\21cmFAST-master'
    #:\21cmFAST\21cmFAST-master\Parameter_files

if __name__ == '__main__':
    print(1)
