
# IMPORTS
import importlib
import numpy as np
import utils
importlib.reload(utils)


# %% READ A BOX
box = utils.read_box(r'delta_T_v3_no_halos_z007.20_nf0.715123_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb014.04_Pop-1_256_300Mpc')


# %% VISUALIZE A BOX
utils.show_box(box)


# %% MAKE A MOVIE
utils.box_to_movie(box)



# %% CHANGE A PARAMETER
change_parameter('ZETA_X', '3.0e56')
