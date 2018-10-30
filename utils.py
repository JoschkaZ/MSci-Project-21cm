import os
from PATH import get_path
import numpy as np
import matplotlib.pyplot as plt
from sys import platform

def what_platform():
    print(platform)
    return platform
def read_box(filename, verbose=1):

    PATH = get_path()

    if platform == "darwin": # its mac
        boxpath = PATH + '/Boxes/' +  filename

    elif platform == "linux":
        boxpath = PATH + '/Boxes/' + filename

    else: # its windows
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

    #ZETA_X (double) (2.0e56) // 2e56 ~ 0.3 X-ray photons per stellar baryon

    # get path to file
    PATH = get_path()
    if parameter_name == 'ZETA_X':
        filepath = PATH + '\\Parameter_files\\HEAT_PARAMS.H'
        if new_value == 'default':
            new_value = '2.0e56'
    elif parameter_name == '...':
        1
        #...
        #...
        #...
    else:
        print('WARNING - PARAMETER_NAME NOT RECOGNIZED!')
        return 0

    if verbose == 1: print('Changing paramter ', parameter_name, ' to ', new_value)

    # read old text
    with open(filepath, "r") as f:
         old = f.readlines() # read everything in the file

    # create new text
    new = []
    for line in old:
        #print(line)
        if '#define ' + parameter_name + ' (' in line:
            #print('CHANGING LINE')
            #print(line)
            newline = '('.join(line.split('(')[0:2]) + '(' + new_value + ')' + ')'.join(line.split(')')[2::])
            #print(newline)
        else:
            newline=line
        new.append(newline)

    #print('#####PRINTING NEW FILE#####')
    #for line in new:
    #    print(line)
    #    1

    # save new text
    '''
    print(filepath[0:-3]+'_mod.H')
    with open(filepath[-3:-1]+'_mod.H', 'w') as f:
        for line in new:
            print(line)
            f.write("%s\n" % line)
    '''

    new = ''.join(new)
    text_file = open(filepath, "w")
    text_file.write(new)
    text_file.close()




    if verbose == 1: print('Parameter file has been modified')

    # r'C:\21cmFAST\21cmFAST-master'
    #:\21cmFAST\21cmFAST-master\Parameter_files

    return 1

def run_commands(commands, verbose=1):
    for command in commands:
        print('Running command: ' + command)
        os.system(command)
    return 1





if __name__ == '__main__':
    print(1)
