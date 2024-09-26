#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:09:41 2024

@author: phil
"""
filtres=['B','V','R']
dossier_root='/home/phil/POLA/RAW_DATA/'
from astropy.modeling import models, fitting
from photutils import aperture_photometry
from photutils import CircularAperture, EllipticalAperture
from photutils import DAOStarFinder
import corner
import emcee
from astropy.io import fits
import numpy as np
from photutils import aperture_photometry
from photutils import CircularAnnulus
import matplotlib.pyplot as plt
import glob
import os
import pathlib
from matplotlib.patches import Rectangle
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
import time
from lmfit import Parameters, minimize, report_fit,Model

##########################################################################

def choose_folder_to_treat(dossier_root, tous=False):
    if tous == False:

        list_dir = glob.glob(dossier_root+'*')
        list_dir.sort()
        for i, sub in enumerate(list_dir):
            print(i, ' ', sub)
        isub = int(input('Choose sub folder number=> '))
        sub = list_dir[isub]
        sub = sub+'/'
        return sub
    else:
        list_dir = glob.glob(dossier_root+'*')
        print(list_dir)
        return np.sort(list_dir)

##########################################################################

def identify_objects_in_sub(sub, filtre='B', not_HD_nor_aster='toto'):
    filenames = glob.glob(sub+'*'+filtre+'.f*t*')
    names = []
    for i, name in enumerate(filenames):
        name = name[len(sub):]
        if name.split('-')[0][0] == 'H' or name.split('-')[0] == 'dark' or name.split('-')[0] == 'bias' or name.split('-')[0] == not_HD_nor_aster:
            names.append(name.split('-')[0])
        else:
            names.append(name.split('-')[0]+'-'+name.split('-')[1])
    names = np.array(names)
    return np.unique(names)

##########################################################################

def create_sub_dirs(sub, names):
    sub_bin = sub +'BIN/'
    if not os.path.isdir(sub_bin):
        
        os.system(f"mkdir {sub_bin}")
        print(sub_bin,' created')
    for name in names:
        print('******')
        print(name)
        print('******')
        if not os.path.isdir(sub_bin+name):
            os.system(f"mkdir {sub_bin}{name}")
            print(sub_bin+name,' created')
            print(name)
#            if (name !='dark' and name != 'bias'):
            for filt in filtres:
                if not os.path.isdir(sub_bin+name+filt):
                
                    os.system(f"mkdir {sub_bin}{name}/{filt}")
                    print(sub_bin+name+'/'+filt, 'created')
    return
##########################################################################

def load_fits_image(filename):
    with fits.open(filename) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        hdul.close()
    return data, header
##########################################################################

def bin_image(image, bin_size=2):
    # Get the shape of the original image
    rows, cols = image.shape

    # Ensure the image dimensions are divisible by the bin_size
    binned_rows = rows // bin_size
    binned_cols = cols // bin_size

    # Reshape and compute the mean over the 2x2 blocks
    binned_image = image[:binned_rows * bin_size, :binned_cols * bin_size].reshape(
        binned_rows, bin_size, binned_cols, bin_size
    ).mean(axis=(1, 3))

    return binned_image

##########################################################################

def bin_fits_image(input_fits, output_fits, bin_factor=2,depth='int'):
    
    # Open the FITS file and load the image data
    
    image_data,header=load_fits_image(input_fits)

    # Bin the image
    if depth==None:
        binned_data = bin_image(image_data, bin_factor)
    elif depth=='int':
        binned_data = np.int16(bin_image(image_data, bin_factor))
        header['BITPIX']=16
    # Update header with new image dimensions
    header['NAXIS1'] = binned_data.shape[1]  # Update the width
    header['NAXIS2'] = binned_data.shape[0] 
    header['XBINNING']=bin_factor# Update the height
    header['YBINNING']=bin_factor
    hdu = fits.PrimaryHDU(binned_data,header)
    # Write the binned image to a new FITS file
    hdu.writeto(output_fits, overwrite=True)
    
    print(input_fits,' binned and saved')
    return
##########################################################################

def dispatch_files_binned_in_subdir(sub,names):
    sub_bin= sub +'BIN/'
    list_files=glob.glob(sub+'*')
    for name in names:
        print('******')
        print(name)
        print('******')
#        if name != 'dark' and name !='bias':
        indx=[]
        for i,nome in enumerate(list_files):
            if name in nome:
                indx.append(i)
        for i in indx:
            fitsfile=list_files[i]
            for filt in filtres:
                if filt in fitsfile.split('-')[-1]:
                    fitsfile_short=fitsfile[len(sub):]
                    output_fits=f"{sub_bin}{name}/{filt}/{fitsfile_short}"
                    bin_fits_image(fitsfile, output_fits, bin_factor=2)

    return
##########################################################################

def cree_cube(liste):
    """
    lit les images de la liste et forme un cube d image
    """
    
    b=liste[0]
    image,head=load_fits_image(b)
    cube=[image] # on cree le cube a partir de la premiere image
#    plt.imshow(image, cmap='gray', norm=LogNorm(),origin='lower')
  
    for b in liste[1:]:
        image,head=load_fits_image(b)
        cube.append(np.transpose(image))
    toto=np.array(cube)
    
#    print( 'Shape du cube ', np.shape(toto))
   
    return toto
##########################################################################

def create_master_bias(sub,filtres=filtres):
    sub_bin=sub+'BIN/'
    for filt in filtres:
        print('Filtre ',filt)
        bias_files=glob.glob(sub_bin+'bias/'+filt+'/*')
        if len(bias_files) !=0:
            cube_bias=cree_cube(bias_files)
            Master_Bias=np.median(cube_bias,axis=0)
            hdu = fits.PrimaryHDU(Master_Bias)
            outdir=sub_bin+'bias/'+filt+'/'
            output_fits=outdir+'Masterbias'+filt+'.fts'
            # Write the binned image to a new FITS file
            hdu.writeto(output_fits, overwrite=True)
            print(output_fits,' created in ',outdir)
        else:
            print('No data in filter ', filt)
    return 
##########################################################################

def get_exopure_times(sub,names=[],fitres=filtres):
    sub_bin=sub+'BIN/'
    
    all_exposures=[]
    for nom in names:
        if nom !='bias' and nom != 'dark':
            for filt in filtres:
                dossier=sub_bin+nom+'/'+filt+'/'
                fichiers=glob.glob(dossier+'*')
                for fich in fichiers:
                    data,header=load_fits_image(fich)
                    all_exposures.append(header['EXPOSURE'])
    return np.unique(np.array(all_exposures))
##########################################################################

def create_rescaled_master_dark(sub,exposures,filtre=filtres):
    sub_bin=sub+'BIN/'
    for filt in filtres:
        print('Master Dak rescaled for filter : ', filt)
        list_darks=glob.glob(sub_bin+'dark/'+filt+'/*')
        if list_darks==[]:
            print('No dark in filter : ',filt)
            continue
        else: 
            data0,head0=load_fits_image(list_darks[0])
            filemasterbias=sub_bin+'bias/'+filt+'/Masterbias'+filt+'.fts'
            masterbias,head_bias=load_fits_image(filemasterbias)
            cube=np.zeros((len(exposures),len(list_darks),np.shape(data0)[0],np.shape(data0)[1]))
            
            for i, dark in enumerate(list_darks):
                data,head=load_fits_image(dark)
                #print(dark,' loaded ', ' sphape =', np.shape(data))
                for iexp,exp in enumerate(exposures):
                
                    datarescale=(data-masterbias)*exp/head['EXPOSURE'] + masterbias
                    cube[iexp,i]=datarescale
            
                
            for iexp,exp in enumerate(exposures):
                print('Filt = ', filt,' Exp = ',exp)
                head0['EXPOSURE']=exp
                dark_rescale_med=np.median(cube[iexp],axis=0)
                hdu = fits.PrimaryHDU(dark_rescale_med,head0)
                outdir=sub_bin+'dark/'+filt+'/'
                output_fits=outdir+'Masterdark'+filt+'_'+str(exp)+'s.fts'
                # Write the binned image to a new FITS file
                hdu.writeto(output_fits,overwrite=True)
                print(output_fits,' created in ',outdir)
    return
            
##########################################################################
        
            
def subtract_scaled_dark(sub,names,filtres=filtres,exposures=[]):
    sub_bin=sub+'BIN/'
    for name in names:
        if name != 'bias' and name !='dark':
            for filt in filtres:
                print('*******')
                print(name,' ',filt)
                print('*******')
                subdirdata=sub_bin+name+'/'+filt+'/'
                listefiles=glob.glob(subdirdata+'*.f*ts')
                if listefiles != []:
                    listefiles.sort()
                    for i,filename in enumerate(listefiles):
                        data,head=load_fits_image(filename)
                        exp_data=head['EXPOSURE']
                        masterdarkfile=sub_bin+'dark'+'/'+filt+'/Masterdark'+filt+'_'+str(exp_data)+'s.fts'
                        dark,head_dark=load_fits_image(masterdarkfile)
                        data_dedarked=data-dark
                        head['COMMENT']='Dedarked by '+ masterdarkfile
                        hdu = fits.PrimaryHDU(data_dedarked,head)
                        outdir=subdirdata
                        output_fits=outdir+filename.split('/')[-1].split('.')[0]+'_d.fts'
                        # Write the binned image to a new FITS file
                        hdu.writeto(output_fits,overwrite=True)
                        print(output_fits,' created in ',outdir)
                else:
                    continue
    return
##########################################################################

def detect_sources(image, fwhm=3, threshold=3):

    mean, median, std = np.mean(image), np.median(image), np.std(image)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    sources = daofind(image - median)

    return sources
##########################################################################
def gaussian2D(x, y, cen_x, cen_y, sig_x, sig_y, offset):
    return np.exp(-(((cen_x-x)/sig_x)**2 + ((cen_y-y)/sig_y)**2)/2.0) + offset

##########################################################################     

def residuals(p, x, y, z):
    height = p["height"].value
    cen_x = p["centroid_x"].value
    cen_y = p["centroid_y"].value
    sigma_x = p["sigma_x"].value
    sigma_y = p["sigma_y"].value
    offset = p["background"].value
    return (z - height*gaussian2D(x,y, cen_x, cen_y, sigma_x, sigma_y,offset))
##########################################################################     
def fit_gauss2D(data,params):

    x, y = np.meshgrid(np.linspace(0,np.shape(data)[0],np.shape(data)[0]), np.linspace(0,np.shape(data)[1],np.shape(data)[1]))
    initial = Parameters()
    initial.add("height",value=params[0])
    initial.add("centroid_x",params[1])
    initial.add("centroid_y",params[2])
    initial.add("sigma_x",params[3])
    initial.add("sigma_y",params[4])
    initial.add("background",params[5])
    
    
    fit = minimize(residuals, initial, args=(x, y, data))
    
    return fit
###################################################################################################################################################

def gaussian2D_ellipse(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    g = offset + amplitude * np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g
##########################################################################
def fit_gauss2D_ellipse(data,params):
    x, y = np.meshgrid(np.linspace(0,np.shape(data)[0],np.shape(data)[0]), np.linspace(0,np.shape(data)[1],np.shape(data)[1]))

    model = Model(gaussian2D_ellipse, independent_vars=['x', 'y'])

    # Initialize parameters
    params = model.make_params(amplitude=params[0], 
                               xo=params[1], 
                               yo=params[2], 
                               sigma_x=params[3], 
                               sigma_y=params[4], 
                               theta=params[5], 
                               offset=params[6])

    # Fit the model to the data
    result = model.fit(data, params, x=x, y=y)

    # Print the fit results
    print(result.fit_report())

    # Generate fitted data
    fitted_data = result.eval(x=x, y=y)

# # Plot the results
    fig, axes = plt.subplots(1, 1, figsize=(9, 5))

    axes.imshow(data, origin='lower', cmap='viridis')
    axes.contour(fitted_data, origin='lower')
    return result
##########################################################################     


def plot_gauss_fit(image_box1,image_box2,box,sources1,
                   sources2,fitted1,fitted2,vmin=0,vmax=10):
    x, y = np.meshgrid(np.linspace(0,box,box), 
                      np.linspace(0,box,box))
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.imshow(image_box1,origin='lower',vmin=vmin,vmax=vmax)
    #for i in range(len(sources1)):
    ax1.contour(gaussian2D(x,y, sources1[0],sources1[1],fitted1.params['sigma_x'],fitted1.params['sigma_y'].value,fitted1.params['background'].value),colors='red')
   
    ax2.imshow(image_box2,origin='lower',vmin=vmin,vmax=vmax)
    #for i in range(len(sources1)):

    ax2.contour(gaussian2D(x,y, sources2[0],sources2[1],fitted2.params['sigma_x'],fitted2.params['sigma_y'].value,fitted2.params['background'].value),colors='red')
    fig.canvas.draw()
    fig.canvas.flush_events()
    ans=input('type enter to next')
    time.sleep(0.1)
    return       
#########################################################################
def plot_gauss_fit_ellipse(image_box1,image_box2,box,sources1,
                   sources2,fitted1,fitted2,vmin=0,vmax=10):
    x, y = np.meshgrid(np.linspace(0,box,box), 
                      np.linspace(0,box,box))
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.imshow(image_box1,origin='lower',vmin=vmin,vmax=vmax)
    for i in range(len(sources1)):
        yy=gaussian2D_ellipse(x, y,
                              fitted1.best_values['amplitude'],
                              sources1[0],
                              sources1[1],
                              fitted1.best_values['sigma_x'],
                              fitted1.best_values['sigma_y'],
                              fitted1.best_values['theta'], 
                              fitted1.best_values['offset'])

        ax1.contour(yy,colors='red')
   
    ax2.imshow(image_box2,origin='lower',vmin=vmin,vmax=vmax)
    for i in range(len(sources1)):
        yy=gaussian2D_ellipse(x, y,
                              fitted2.best_values['amplitude'],
                              sources2[0],
                              sources2[1],
                              fitted2.best_values['sigma_x'],
                              fitted2.best_values['sigma_y'],
                              fitted2.best_values['theta'], 
                              fitted2.best_values['offset'])

        ax2.contour(yy,colors='red')
    fig.canvas.draw()
    fig.canvas.flush_events()
    ans=input('type enter to next')
    time.sleep(0.1)
    return  
########################################################################     
def click_cut_sources(image,box=100,thresh_detection=5,PLOT=True):

    plt.close('all')
    print('box= ',box)
    vmin=np.mean(image)-3*np.std(image)
    vmax=np.mean(image)+3*np.std(image)
    figure, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image,origin='lower',vmin=vmin,vmax=vmax)
    Y1,Y2=plt.ginput(2,timeout=-1,show_clicks=True)
    ax.add_patch(Rectangle((Y1[0]-box/2,Y1[1]-box/2),box,box,
                        edgecolor='white',
                        facecolor='none',
                        lw=2))
    ax.add_patch(Rectangle((Y2[0]-box/2,Y2[1]-box/2),box,box,
                        edgecolor='pink',
                        facecolor='none',
                        lw=2))
    plt.scatter(Y1[0],Y1[1],marker='X',color='black')
    plt.scatter(Y2[0],Y2[1],marker='X',color='black')
    figure.canvas.draw()
    figure.canvas.flush_events()
    
    time.sleep(0.1)
    
    source1=[]
    source2=[]

    position1=(Y1[0],Y1[1])
    position2=(Y2[0],Y2[1])
    cutout1 = Cutout2D(image, position=position1, size=box)
    cutout2 = Cutout2D(image, position=position2, size=box)

    image_box1=cutout1.data
    image_box2=cutout2.data
    params1=[np.max(image_box1),
                box//2,
                box//2,
                10,
                10,
                np.std(image_box1)]
    params2=[np.max(image_box2),
                box//2,
                box//2,
                10,
                10,
                np.std(image_box2)]
    fitted1=fit_gauss2D(image_box1,params1)
    fitted2=fit_gauss2D(image_box2,params2)
    
    fwhm1=np.max([fitted1.params['sigma_x'].value,fitted1.params['sigma_y'].value])
#I take max and not mean in case of elongated psf
    fwhm1=2*np.sqrt(2*np.log(2))*fwhm1
    

    fwhm2=np.max([fitted2.params['sigma_x'].value,fitted2.params['sigma_y'].value])
    fwhm2=2*np.sqrt(2*np.log(2))*fwhm2
    print('fwhms ',fwhm1,fwhm2)
    source = detect_sources(image_box1, fwhm=fwhm1, threshold=thresh_detection)
    source1=(source['xcentroid'].data[0],source['ycentroid'].data[0])
    source=detect_sources(image_box2, fwhm=fwhm2, threshold=thresh_detection)
    source2=(source['xcentroid'].data[0],source['ycentroid'].data[0])

    
    if PLOT == True:
        plt.close('all')
        
        plot_gauss_fit(image_box1,image_box2,box,source1,source2,fitted1,fitted2,vmin=vmin,vmax=vmax)
    return source1,source2,fwhm1,fwhm2,image_box1,image_box2 
#################################################
def detect_sources_cube(cube,box,threshold,PLOT):      
    # sub_bin=sub+'BIN/'
    # subdir=sub_bin+name+'/'+filtre+'/'
    # files=glob.glob(subdir+'*_d.fts')
    # files.sort()
    # cube=cree_cube(files)
    sources1=[]
    sources2=[]
    fwhms1=[]
    fwhms2=[]
    cube_box1=[]
    cube_box2=[]
    for i in range(np.shape(cube)[0]):
        print('Image n° ', i)
        
        source1,source2,fwhm1,fwhm2,image_box1,image_box2=\
            click_cut_sources(cube[i],box=box,thresh_detection=threshold,PLOT=PLOT)
        sources1.append(source1)
        sources2.append(source2)
        fwhms1.append(fwhm1)
        fwhms2.append(fwhm2)
        cube_box1.append(image_box1)
        cube_box2.append(image_box2)
    return sources1,sources2,fwhms1,fwhms2,np.array(cube_box1),np.array(cube_box2)
#####################################################


def calculate_snr(texp,flux, sky_mean, aperture_area, sky_area):
    ccd_gain=0.9
    ccd_readout_noise=2 #in e/pix/sec
    ccd_dark_current=0.1
    signal = flux
    sky_noise = sky_mean * aperture_area
    readout_noise=aperture_area * ccd_readout_noise**2
    dark_noise=ccd_dark_current*aperture_area*texp
    noise = np.sqrt(signal + sky_noise + readout_noise+dark_noise)
    snr = signal / noise
    return snr
# def calculate_snr(R, fwhm, sky_mean,aperture_area):
#     nu=sky_mean * aperture_area
#     sigma=fwhm/2./np.sqrt(2.*np.log(2.))
#     f_star = 1 - np.exp( -0.5*(R/sigma)**2 )
#     snr = f_star / np.sqrt( f_star + np.pi*nu*R*R )
#     return snr

def decosmic(image):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 8))
    vmin=np.mean(image)-3*np.std(image)
    vmax=np.mean(image)+3*np.std(image)
    ax.imshow(image, vmin=vmin,vmax=vmax,origin='lower')
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)
    
    bb=4
    ans= 'n'
    ans=input('decosmicage y/[n] ? ==>')
    if ans !='y':
        return image
    else:
        
        Ycosm=[]
        again=''
        
        while(again == ''):
            print('click on comsic')
            Ycosm.append(plt.ginput(1,timeout=-1,show_clicks=True))
            again=input(' again ? [return]/n ==> ')
        for i, pos in enumerate(Ycosm):
            xcosm=int(np.array(Ycosm[i]).ravel()[0])
            ycosm=int(np.array(Ycosm[i]).ravel()[1])
            image[ycosm-bb//2:ycosm+bb//2,xcosm-bb//2:xcosm+bb//2]=\
                np.median(image[ycosm-bb//2:ycosm+bb//2,xcosm-bb//2:xcosm+bb//2])
    plt.close('all')
    return image
#####################################################
def photometry_best_aperture(texp,image,source, fwhm,box,PLOT,filename):
    # cosmics=np.where(image >= np.mean(image)+15.*np.std(image))
    # image[cosmics]=np.median(image)
    
    x=np.array(source).ravel()[0]
    y=np.array(source).ravel()[1]
    position=(x,y)
    print(position)
    offset=5
    width=2*offset
    aperture_radii=np.linspace(0.5*fwhm,box//2-width,box//2-width,dtype='int16')
    aperture_radii=np.unique(aperture_radii) # avec les entiers linspce peut repeter la valeur
    annulus_radii = [(r + offset, r + width) for r in aperture_radii]
    # Create the list of apertures and annuli)
    #annulus_radii=np.unique(annulus_radii)
    apertures = [CircularAperture(position, r=r) for r in aperture_radii]
    annuli = [CircularAnnulus(position, r_in=ann_r[0], r_out=ann_r[1]) for ann_r in annulus_radii]
    
    # Perform aperture photometry for all apertures and annuli
    phot_table = aperture_photometry(image, apertures + annuli)
    
    # Loop over the apertues and annuli to calculate SNR
    best_snr = 0
    best_aperture_radius = 0
    best_annulus_radii = (0, 0)
    jbest=0
    aperture_sum=np.zeros((len(aperture_radii),len(annulus_radii)))
    snr=np.zeros((len(aperture_radii),len(annulus_radii)))
    noise=np.zeros((len(aperture_radii),len(annulus_radii)))
    for i, r in enumerate(aperture_radii):
        # Extract annulus data and calculate background statistics
        #print('radius : ',i,' ',r)
        for j in range(i,len(aperture_radii),1):
            annulus_mask = annuli[j].to_mask(method='center')
            annulus_data = annulus_mask.multiply(image)
            annulus_data_1d = annulus_data[annulus_mask.data > 0]  # 1D array of the annulus pixel values
            bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(annulus_data_1d)
    
        # Subtract background from the aperture flux
            #aperture_sum_ij= phot_table['aperture_sum_' + str(i)].data[0] - bkg_median/annuli[j].area * apertures[i].area
            
            # I take bak_mean instead bkg_median so that if a star is in annulus this annulus will not be the best
            aperture_sum_ij= phot_table['aperture_sum_' + str(i)].data[0] - bkg_mean* apertures[i].area
            aperture_sum[i][j] =aperture_sum_ij
            #bkg_mean = bkg_table['aperture_sum'] / annulus_aperture.area
            #     bkg_total = bkg_mean * aperture.area
            #     flux = phot_table['aperture_sum'] - bkg_total
        # # Calculate the noise
        #     noise_ij=np.sqrt(aperture_sum_ij + (apertures[i].area * bkg_median/annuli[j].area)**2)
            
        #     noise[i][j] = noise_ij
        # # Calculate SNR
        #     snr_ij = aperture_sum_ij / noise_ij
            
            #snr_ij=calculate_snr(r, fwhm, bkg_mean,apertures[i].area)
            snr_ij=calculate_snr(texp,aperture_sum_ij, bkg_mean, apertures[i].area, annuli[j].area)
            snr[i][j] = snr_ij
           # print('radius : ',r,'annulus :',j,' snr: ',snr_ij)
        # Store the SNR in the phot_table
            phot_table['snr_' + str(r)+'_'+str(annulus_radii[j])] = snr_ij
    
        # Check if this is the best SNR
    indx=np.where(snr==np.max(snr))
    ibest=indx[0][0]
    jbest=indx[1][0]
    best_snr=snr[ibest][jbest]
    best_aperture_radius = aperture_radii[ibest]
    best_annulus_radii = annulus_radii[jbest]
    flux=aperture_sum[ibest][jbest]
    err_flux=flux/best_snr         
    
    print(f"Best aperture radius: {best_aperture_radius} pixels")
    print(f"Best annulus radii: {best_annulus_radii} pixels (inner, outer)")
    print(f"Highest SNR: {best_snr}")
    print(f"Best aperture radius: {best_aperture_radius} pixels")
    print(f"Best Flux: {flux}")
    print(f"err Flux: {err_flux}")
    
    # for r in aperture_radii:
    #     ax.add_patch(plt.Circle(position,
    #                                  radius=r,
    #                                  edgecolor='blue',
    #                                  facecolor='none',
    #                                  lw=2,
    #                                  linestyle='--',
    #                                  alpha=0.5))
    if PLOT:
        
        fig, ax = plt.subplots(figsize=(10, 8))
        vmin=np.mean(image)-3*np.std(image)
        vmax=np.mean(image)+3*np.std(image)
        ax.imshow(image, vmin=vmin,vmax=vmax,origin='lower')
        ax.add_patch(plt.Circle(position,
                                     radius=fwhm/2,
                                     edgecolor='blue',
                                     facecolor='none',
                                     lw=2,
                                     antialiased=True,
                                     label='fwhm'))
        ax.add_patch(plt.Circle(position,
                                     radius=best_aperture_radius,
                                     edgecolor='red',
                                     facecolor='none',
                                     lw=2,
                                     antialiased=True,
                                     label='best radius'))
        ax.add_patch(plt.Circle(position,
                                     radius=best_annulus_radii[0],
                                     edgecolor='orange',
                                     facecolor='none',
                                     lw=2,
                                     antialiased=True,
                                     label='annulus best'))
        ax.add_patch(plt.Circle(position,
                                     radius=best_annulus_radii[1],
                                     edgecolor='orange',
                                     facecolor='none',
                                     lw=2,
                                     antialiased=True,
                                     ))
        ax.legend()
        ax.set_title(filename)
        fig.canvas.draw()
        fig.canvas.flush_events()
        #ans=input('type enter to next')
        time.sleep(0.2)
        plt.close('all')
    return flux,err_flux,best_aperture_radius,\
            best_annulus_radii,best_snr,phot_table

#################################################################
def do_photom(sub,names,filtre,box,
              start_cube,end_cube,thresh_detection,
              PLOT,COSMIC):
    for i, name in enumerate(names):
        print(i,' ',name)
    indx=int(input('choose your # of target ==> '))
    name=names[indx]
    sub_bin=sub+'BIN/'
    #filtre='B'
    subdir=sub_bin+name+'/'+filtre+'/'
    files=glob.glob(subdir+'*_d.fts')
    files.sort()
    cube=cree_cube(files)
    cube=cube[start_cube:end_cube] # for test
    im,h=load_fits_image(files[0])
    texp=h['EXPOSURE'] #needed for SNR computation
    
    sources1,sources2,fwhms1,fwhms2,cube_box1,cube_box2\
        =detect_sources_cube(cube,box,thresh_detection,PLOT)
    flux1=[]
    flux2=[]
    err_flux1=[]
    err_flux2=[]
    for i in range(np.shape(cube)[0]):
        filename=files[i][len(subdir):]
        image1=cube_box1[i]
        if COSMIC:
            image1=decosmic(image1)
        fwhm1=fwhms1[i]
        source1=sources1[i]
        filename1=filename+'-box1'
        flux_1,err_flux_1,\
            best_aperture_radius1, best_annulus_radii1,\
            best_snr,phot_table1=\
                photometry_best_aperture(texp,image1,source1, fwhm1,box,PLOT,filename1)
        flux1.append(flux_1)
        err_flux1.append(err_flux_1)
        
        image2=cube_box2[i]
        if COSMIC:
            image2=decosmic(image2)
        fwhm2=fwhms2[i]
        source2=sources2[i]
        filename2=filename+'-box2'
        flux_2,err_flux_2,\
            best_aperture_radius2, best_annulus_radii2,\
            best_snr2,phot_table2=\
                photometry_best_aperture(texp,image2,source2, fwhm2,box,PLOT,filename2)
        flux2.append(flux_2)
        err_flux2.append(err_flux_2)
    return flux1,err_flux1,flux2,err_flux2
#################################################################
def click_cut_sources_ellipse(image,box=100,thresh_detection=5,PLOT=True):

    plt.close('all')
    print('box= ',box)
    vmin=np.mean(image)-3*np.std(image)
    vmax=np.mean(image)+3*np.std(image)
    figure, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image,origin='lower',vmin=vmin,vmax=vmax)
    Y1,Y2=plt.ginput(2,timeout=-1,show_clicks=True)
    ax.add_patch(Rectangle((Y1[0]-box/2,Y1[1]-box/2),box,box,
                        edgecolor='white',
                        facecolor='none',
                        lw=2))
    ax.add_patch(Rectangle((Y2[0]-box/2,Y2[1]-box/2),box,box,
                        edgecolor='pink',
                        facecolor='none',
                        lw=2))
    plt.scatter(Y1[0],Y1[1],marker='X',color='black')
    plt.scatter(Y2[0],Y2[1],marker='X',color='black')
    figure.canvas.draw()
    figure.canvas.flush_events()
    
    time.sleep(0.1)
    
    source1=[]
    source2=[]

    position1=(Y1[0],Y1[1])
    position2=(Y2[0],Y2[1])
    cutout1 = Cutout2D(image, position=position1, size=box)
    cutout2 = Cutout2D(image, position=position2, size=box)

    image_box1=cutout1.data
    image_box2=cutout2.data
    params1=[np.max(image_box1),
                box//2,
                box//2,
                10,
                10,
                45.*np.pi/180.,
                np.std(image_box1)]
    params2=[np.max(image_box2),
                box//2,
                box//2,
                10,
                10,
                45.*np.pi/180.,
                np.std(image_box2)]
    
    fitted1=fit_gauss2D_ellipse(image_box1,params1)
    fitted2=fit_gauss2D_ellipse(image_box2,params2)
    
    fwhm1x=fitted1.best_values['sigma_x']
    fwhm1y=fitted1.best_values['sigma_y']
    fwhm1x=2*np.sqrt(2*np.log(2))*fwhm1x
    fwhm1y=2*np.sqrt(2*np.log(2))*fwhm1y
    theta1=fitted1.best_values['theta']
    theta1=theta1%(2*np.pi) # theta entre o et 2pi radians
    
    fwhm2x=fitted2.best_values['sigma_x']
    fwhm2y=fitted2.best_values['sigma_y']
    fwhm2x=2*np.sqrt(2*np.log(2))*fwhm2x
    fwhm2y=2*np.sqrt(2*np.log(2))*fwhm2y
    theta2=fitted2.best_values['theta']
    theta2=theta2%(2*np.pi) # theta entre o et 2pi radians
    
    fwhm1=(fwhm1x,fwhm1y)
    fwhm2=(fwhm2x,fwhm2y)
    theta=(theta1,theta2)
    print('fwhms ',fwhm1x,fwhm1y,fwhm2x,fwhm2y)
    source = detect_sources(image_box1, fwhm=max(fwhm1x,fwhm1y), threshold=thresh_detection)
    source1=(source['xcentroid'].data[0],source['ycentroid'].data[0])
    source=detect_sources(image_box2, fwhm=max(fwhm2x,fwhm2y), threshold=thresh_detection)
    source2=(source['xcentroid'].data[0],source['ycentroid'].data[0])

    
    if PLOT == True:
        plt.close('all')
        
        plot_gauss_fit_ellipse(image_box1,image_box2,box,source1,source2,fitted1,fitted2,vmin=vmin,vmax=vmax)
    return source1,source2,fwhm1,fwhm2,theta,image_box1,image_box2
##############################################################"
def detect_sources_cube_ellipse(cube,box,threshold,PLOT):      
    # sub_bin=sub+'BIN/'
    # subdir=sub_bin+name+'/'+filtre+'/'
    # files=glob.glob(subdir+'*_d.fts')
    # files.sort()
    # cube=cree_cube(files)
    sources1=[]
    sources2=[]
    fwhms1=[]
    fwhms2=[]
    cube_box1=[]
    cube_box2=[]
    for i in range(np.shape(cube)[0]):
        print('Image n° ', i)
        
        source1,source2,fwhm1,fwhm2,theta,image_box1,image_box2=\
            click_cut_sources_ellipse(cube[i],box=box,thresh_detection=threshold,PLOT=PLOT)
        sources1.append(source1)
        sources2.append(source2)
        fwhms1.append(fwhm1)
        fwhms2.append(fwhm2)
        cube_box1.append(image_box1)
        cube_box2.append(image_box2)
    return sources1,sources2,fwhms1,fwhms2,theta,np.array(cube_box1),np.array(cube_box2)


##############################################################"
def photometry_best_aperture_ellipse(texp,image,source, fwhm,theta,box,PLOT,filename):
    
    x=np.array(source).ravel()[0]
    y=np.array(source).ravel()[1]
    position=(x,y)
    print(position)
    offset=5
    width=2*offset
    aperture_radii_a=np.linspace(1.1*fwhm[0],box//2-width,box//2-width,dtype='int16')
    aperture_radii_b=np.linspace(1.1*fwhm[1],box//2-width,box//2-width,dtype='int16')
    aperture_radii_a=np.unique(aperture_radii_a)# avec les entiers linspce peut repeter la valeur
    aperture_radii_b=np.unique(aperture_radii_b)
    annulus_radii_a= [(r + offset, r + width) for r in aperture_radii_a]
    annulus_radii_b= [(r + offset, r + width) for r in aperture_radii_b]
    # Create the list of apertures and annuli)
    #annulus_radii=np.unique(annulus_radii)
    apertures = [EllipticalAperture(position, a=r[0],b=r[1],theta=theta) for r in list(zip(aperture_radii_a,aperture_radii_b))]
    annuli = [EllipticalAnnulus(position, a_in=r[0][0],a_out=r[0][1],b_in=r[1][0],b_out=r[1][1], theta=theta) for r in list(zip(annulus_radii_a,annulus_radii_b))]
    
    # Perform aperture photometry for all apertures and annuli
    phot_table = aperture_photometry(image, apertures + annuli)
    
    # Loop over the apertues and annuli to calculate SNR
    best_snr = 0
    best_aperture_radius = 0
    best_annulus_radii = (0, 0)
    jbest=0
    print(len(aperture_radii_a),len(aperture_radii_b),)
    aperture_sum=np.zeros((len(aperture_radii_a),len(annulus_radii_a)))
    snr=np.zeros((len(aperture_radii_a),len(annulus_radii_a)))
    noise=np.zeros((len(aperture_radii_a),len(annulus_radii_a)))
    for i, r in enumerate(aperture_radii_a):
        # Extract annulus data and calculate background statistics
        #print('radius : ',i,' ',r)
        for j in range(i,len(aperture_radii_a)-2):
            annulus_mask = annuli[j].to_mask(method='center')
            annulus_data = annulus_mask.multiply(image)
            annulus_data_1d = annulus_data[annulus_mask.data > 0]  # 1D array of the annulus pixel values
            bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(annulus_data_1d)
    
        # Subtract background from the aperture flux
            #aperture_sum_ij= phot_table['aperture_sum_' + str(i)].data[0] - bkg_median/annuli[j].area * apertures[i].area
            
            # I take bak_mean instead bkg_median so that if a star is in annulus this annulus will not be the best
            aperture_sum_ij= phot_table['aperture_sum_' + str(i)].data[0] - bkg_mean* apertures[i].area
            aperture_sum[i][j] =aperture_sum_ij
            #bkg_mean = bkg_table['aperture_sum'] / annulus_aperture.area
            #     bkg_total = bkg_mean * aperture.area
            #     flux = phot_table['aperture_sum'] - bkg_total
        # # Calculate the noise
        #     noise_ij=np.sqrt(aperture_sum_ij + (apertures[i].area * bkg_median/annuli[j].area)**2)
            
        #     noise[i][j] = noise_ij
        # # Calculate SNR
        #     snr_ij = aperture_sum_ij / noise_ij
            
            #snr_ij=calculate_snr(r, fwhm, bkg_mean,apertures[i].area)
            snr_ij=calculate_snr(texp,aperture_sum_ij, bkg_mean, apertures[i].area, annuli[j].area)
            snr[i][j] = snr_ij
           # print('radius : ',r,'annulus :',j,' snr: ',snr_ij)
        # Store the SNR in the phot_table
            phot_table['snr_' + str(r)+'_'+str(annulus_radii_a[j])+'_'+str(annulus_radii_b[j])]= snr_ij
    
        # Check if this is the best SNR
    indx=np.where(snr==np.max(snr))
    ibest=indx[0][0]
    jbest=indx[1][0]
    best_snr=snr[ibest][jbest]
    best_aperture_radius_a = aperture_radii_a[ibest]
    best_aperture_radius_b = aperture_radii_b[ibest]
    best_annulus_radius_a = annulus_radii_a[jbest]
    best_annulus_radius_b = annulus_radii_b[jbest]
    flux=aperture_sum[ibest][jbest]
    err_flux=flux/best_snr         
    
    print(f"Best aperture radius a: {best_aperture_radius_a} pixels")
    print(f"Best aperture radius b: {best_aperture_radius_b} pixels")
    print(f"Best annulus radii a: {best_annulus_radius_a} pixels (inner, outer)")
    print(f"Best annulus radii b: {best_annulus_radius_b} pixels (inner, outer)")
    print(f"Highest SNR: {best_snr}")
    print(f"Best aperture radius: {best_aperture_radius} pixels")
    print(f"Best Flux: {flux}")
    print(f"err Flux: {err_flux}")
    
    # for r in aperture_radii:
    #     ax.add_patch(plt.Circle(position,
    #                                  radius=r,
    #                                  edgecolor='blue',
    #                                  facecolor='none',
    #                                  lw=2,
    #                                  linestyle='--',
    #                                  alpha=0.5))
    if PLOT:
        
        fig, ax = plt.subplots(figsize=(10, 8))
        vmin=np.mean(image)-3*np.std(image)
        vmax=np.mean(image)+3*np.std(image)
        ax.imshow(image, vmin=vmin,vmax=vmax,origin='lower')
        # ax.add_patch(plt.Ellipse(position,
        #                              width=2*best_aperture_radius_a,
        #                              height=2*best_aperture_radius_b,
        #                              angle=theta,
        #                              edgecolor='blue',
        #                              facecolor='none',
        #                              lw=2,
        #                              antialiased=True,
        #                              label='fwhm'))
        aperture_fwhm=EllipticalAperture(position, a=fwhm[0],b=fwhm[1],theta=theta)
        aperture_fwhm.plot(ax,
                                      edgecolor='blue',
                                      facecolor='none',
                                      lw=2,
                                      antialiased=True,
                                      label='fwhm')      
        apertures[ibest].plot(ax,
                                      edgecolor='red',
                                      facecolor='none',
                                      lw=2,
                                      antialiased=True,
                                      label='best aperture')
        annuli[jbest].plot(ax,
                                      edgecolor='orange',
                                      facecolor='none',
                                      lw=2,
                                      antialiased=True,
                                      )
        # ax.add_patch(plt.Ellipse(position,
        #                              width=2*best_annulus_radius_a,
        #                              height=2*best_annulus_radius_b,
        #                              angle=theta,
        #                              edgecolor='red',
        #                              facecolor='none',
        #                              lw=2,
        #                              antialiased=True,
        #                              label='best radius'))
        # ax.add_patch(plt.Ellipse(position,
        #                              radius=best_annulus_radii[0],
        #                              edgecolor='orange',
        #                              facecolor='none',
        #                              lw=2,
        #                              antialiased=True,
        #                              label='annulus best'))
        # ax.add_patch(plt.Circle(position,
        #                              radius=best_annulus_radii[1],
        #                              edgecolor='orange',
        #                              facecolor='none',
        #                              lw=2,
        #                              antialiased=True,
        #                              ))
        ax.legend()
        ax.set_title(filename)
        fig.canvas.draw()
        fig.canvas.flush_events()
        #ans=input('type enter to next')
        time.sleep(0.2)
        plt.close('all')
    return flux,err_flux,best_aperture_radius,\
            best_annulus_radii,best_snr,phot_table

##############################################################
def do_photom_ellipse(sub,names,filtre,box,
              start_cube,end_cube,thresh_detection,
              PLOT,COSMIC):
    for i, name in enumerate(names):
        print(i,' ',name)
    indx=int(input('choose your # of target ==> '))
    name=names[indx]
    sub_bin=sub+'BIN/'
    #filtre='B'
    subdir=sub_bin+name+'/'+filtre+'/'
    files=glob.glob(subdir+'*_d.fts')
    files.sort()
    cube=cree_cube(files)
    cube=cube[start_cube:end_cube] # for test
    im,h=load_fits_image(files[0])
    texp=h['EXPOSURE'] #needed for SNR computation
    
    sources1,sources2,fwhms1,fwhms2,theta,cube_box1,cube_box2\
        =detect_sources_cube_ellipse(cube,box,thresh_detection,PLOT)
    flux1=[]
    flux2=[]
    err_flux1=[]
    err_flux2=[]
    for i in range(np.shape(cube)[0]):
        filename=files[i][len(subdir):]
        image1=cube_box1[i]
        if COSMIC:
            image1=decosmic(image1)
        fwhm1=fwhms1[i]
        source1=sources1[i]
        filename1=filename+'-box1'
        flux_1,err_flux_1,\
            best_aperture_radius1, best_annulus_radii1,\
            best_snr1,phot_table1=\
            photometry_best_aperture_ellipse(
                texp,image1,source1, fwhm1,theta[0],
                box,PLOT,filename1)
        flux1.append(flux_1)
        err_flux1.append(err_flux_1)
        
        image2=cube_box2[i]
        if COSMIC:
            image2=decosmic(image2)
        fwhm2=fwhms2[i]
        source2=sources2[i]
        filename2=filename+'-box2'
        flux_2,err_flux_2,best_aperture_radius2,\
            best_annulus_radii2,best_snr2,phot_table2\
                =photometry_best_aperture_ellipse(texp,image2,\
                    source2, fwhm2,theta[1],box,PLOT,filename2)
        flux2.append(flux_2)
        err_flux2.append(err_flux_2)
    return flux1,err_flux1,flux2,err_flux2