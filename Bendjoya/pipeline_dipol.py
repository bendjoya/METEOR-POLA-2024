#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:57:49 2024

@author: phil
"""

import lib_dipol_phil as ldp

#################################################################

filtres=['B','V','R']
dossier_root='/home/phil/POLA/RAW_DATA/'

sub=ldp.choose_folder_to_treat(dossier_root)
not_HD_nor_aster='2024MK'
names=ldp.identify_objects_in_sub(sub, not_HD_nor_aster=not_HD_nor_aster)
#ldp.create_sub_dirs(sub, names)
#ldp.dispatch_files_binned_in_subdir(sub,names)
#ldp.create_master_bias(sub,filtres=filtres)
exposures=ldp.get_exopure_times(sub,names=names,fitres=filtres)
#ldp.create_rescaled_master_dark(sub,exposures,filtre=filtres)
#ldp.subtract_scaled_dark(sub,names,filtres=filtres,exposures=exposures)

start_cube=0
end_cube= 2#-1 for entire cube
thresh_detection=5
box=100
PLOT=True
filtre='B'
COSMIC=False
CIRCULAR=False
if CIRCULAR:
    flux1,err_flux1,flux2,err_flux2=ldp.do_photom(
                            sub,names,filtre,
                            box,start_cube,end_cube,thresh_detection,
                            PLOT,COSMIC)
else:
    flux1,err_flux1,flux2,err_flux2=ldp.do_photom_ellipse(
                            sub,names,filtre,
                            box,start_cube,end_cube,thresh_detection,
                            PLOT,COSMIC)
