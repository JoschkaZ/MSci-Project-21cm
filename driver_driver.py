# IMPORTS
import importlib
import numpy as np
import utils
importlib.reload(utils)





#change some parameter
utils.change_parameter('ZETA_X', '4.0e56')

#run driver
utils.cd_to_programs()
commands = ['make', './drive_logZscroll_Ts']



'''
next steps after driver has finished:
1. process boxes
2. downlaod boxes
3. repeat for different settings
'''
