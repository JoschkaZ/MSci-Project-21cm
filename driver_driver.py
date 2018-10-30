# IMPORTS
import importlib
import numpy as np
import utils
importlib.reload(utils)


loopz = [0]
for loop in loopz:

    #clean box folder
    utils.clear_box_directory()

    #change some parameter
    utils.change_parameter('ZETA_X', '4.0e56')

    #run driver
    utils.cd_to_programs()
    commands = ['make', './drive_logZscroll_Ts']
    utils.run_commands(commands)


    '''
    next steps after driver has finished:
    1. process boxes
    2. downlaod boxes
    3. repeat for different settings
    '''
