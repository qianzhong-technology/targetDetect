############### Add include ###################
list(APPEND ADD_INCLUDE "")
list(APPEND ADD_PRIVATE_INCLUDE "")
###############################################

############ Add source files #################
append_srcs_dir(ADD_SRCS "src")
###############################################

###### Add required/dependent components ######
list(APPEND ADD_REQUIREMENTS basic vision)
###############################################

#### Add compile option for this component ####
list(APPEND ADD_DEFINITIONS_PRIVATE -O3 -ffast-math)
###############################################

# Build this component as shared library.
register_component(DYNAMIC)
