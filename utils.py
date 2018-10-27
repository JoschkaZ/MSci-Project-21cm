import os
from PATH import get_path
import numpy as np
import matplotlib.pyplot as plt

def read_box(filename, verbose=1):

    PATH = get_path()
    boxpath = PATH + '\\Boxes\\' +  filename
    boxpath = boxpath
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
        print()'saving frame #', layer)
        plt.imshow(box[layer], cmap='hot', interpolation='nearest')
        plt.savefig(savedirectory + str(layer) + '.png', format='png')

    print('..........')
    return 1
