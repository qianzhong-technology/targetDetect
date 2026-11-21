############### Add include ###################
list(APPEND ADD_INCLUDE "")
list(APPEND ADD_PRIVATE_INCLUDE "")
###############################################

############ Add source files #################
append_srcs_dir(ADD_SRCS "src")
###############################################

###### Add required/dependent components ######
list(APPEND ADD_REQUIREMENTS basic targetdetect_accel)
###############################################

register_component()
