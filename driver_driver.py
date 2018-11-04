# IMPORTS
import importlib
import numpy as np
import utils
importlib.reload(utils)


seeds = range(1)
for seed in seeds:
    print('using seed ', seed)

    #clean box folder
    utils.clear_box_directory()

    #change some parameters
    utils.change_parameter('ZETA_X', 'default')
    utils.change_param('SEED', str(seed))
    utils.change_param('drive_zscroll_noTs ZSTART', 9)
    utils.change_param('drive_zscroll_noTs ZEND', 9)

    #run driver
    utils.cd_to_programs()
    #commands = ['make', './drive_logZscroll_Ts']
    commands = ['make', './drive_zscroll_noTs']
    utils.run_commands(commands)

    #zip all delta_T_boxes
    utils.cd_to_boxes()
    box_names = utils.get_delta_T_boxes()
    archive_name = 'my_archive ' + str(seed)
    utils.zip_boxes(box_names, archive_name)


    '''
    next steps after driver has finished:
    1. process boxes
    2. downlaod boxes
    3. repeat for different settings
    '''
