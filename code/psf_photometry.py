import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.visualization import simple_norm
from astroquery.sdss import SDSS
from astroquery.vizier import Vizier
from astroquery.mast import Mast
import pyvo as vo
from utilities_az import visualization as vis
from photutils.psf import extract_stars, EPSFBuilder, PSFPhotometry
from photutils.background import Background2D, LocalBackground
from photutils.centroids import centroid_com
from astropy.modeling import models, fitting
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry, ApertureStats
from astropy.table import Table
from astropy.visualization import simple_norm

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import photutils

def wrapper(cat, filename, sn_coords, filter, ext=1, 
            bright_lim=18, faint_lim=20, sn_cutout_size=10, 
            fig_dir='../figures', poly=None,
            sn_background=None, nfwhm_aper=2, aper_annulus_offset=1):
    #can probably get filter by parsing the header (0th ext FILTER2)
    # Switch to Gaia catalog for PSF?
    img, epsf, psf_cat = build_psf(cat, filename, filter, ext=ext, visualize=True, bright_lim=bright_lim, faint_lim=faint_lim, fig_dir=fig_dir, poly=poly) 
    zeropoint_phot, phot_fig = do_photometry(psf_cat, img, epsf)
    phot_fig.savefig(os.path.join(fig_dir,f'{os.path.splitext(os.path.basename(filename))[0]}_psf_sub_residuals_zeropt.pdf'))
    plt.close(phot_fig)
    sn_cat = make_sn_coord_table(filename, sn_coords, img, cutout_size=sn_cutout_size, ext=ext)
    sn_phot, sn_fig = do_photometry(sn_cat, img, epsf, background=sn_background)
    sn_fig.savefig(os.path.join(fig_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_psf_sub_residuals_sn.pdf'))
    zeropoint = calc_zeropoint(filter, zeropoint_phot, psf_cat, filename, fig_dir=fig_dir)
    print('PSF Photometry')
    sn_phot[f'inst_mag_psf'] = calc_mag(sn_phot['flux_fit'])
    sn_phot[f'app_mag_psf'] = ((sn_phot[f'inst_mag_psf'][0]+zeropoint)).item()
    print(f"={sn_phot[f'app_mag_psf'][0]} mag")
    sn_phot.write(filename.replace('.fits', f'sn_phot_{filter}.csv'), overwrite=True)
    print('Aperture Photometry')
    #instrumental_mag
    sn_phot_aper, psf_aper_fig, fig_aper = do_aperture_photometry(img, epsf, sn_cat, nfwhm=nfwhm_aper, aper_annulus_offset=aper_annulus_offset)
    psf_aper_fig.savefig(os.path.join(fig_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_psf_gaussian_check.pdf'))
    plt.close(psf_aper_fig)
    fig_aper.savefig(os.path.join(fig_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_apertures.pdf'))
    plt.close(fig_aper)
    sn_phot_aper['inst_mag_aper'] = -2.5*np.log10(sn_phot_aper['aperture_sum_bkgsub'])
    sn_phot_aper[f'app_mag_aper'] = sn_phot_aper['inst_mag_aper']+zeropoint
    sn_phot_aper.write(filename.replace('.fits', f'sn_phot_aper_{filter}.csv'), overwrite=True)
    print(f"={sn_phot_aper[f'app_mag_aper'][0]} mag")
    return sn_phot, sn_phot_aper

def get_catalogs(sn_coords, sn_name, width=5.5, height=5.5, extended_mag_cut=0.1, cat_dir='../data'):
    apass_cat = query_apass(sn_coords, width=width, height=height)
    if not isinstance(apass_cat, type(None)):
        apass_cat.write(os.path.join(cat_dir, f'{sn_name}_apass.cat'), format='ascii.csv', overwrite=True)
    sdss_cat = query_sdss(sn_coords, width=width, height=height)
    if not isinstance(sdss_cat, type(None)):
        sdss_cat.write(os.path.join(cat_dir, f'{sn_name}_sdss.cat'), format='ascii.csv', overwrite=True)
    panstarrs_cat = query_panstarrs(sn_coords, width=width, height=height, extended_mag_cut=extended_mag_cut)
    if not isinstance(panstarrs_cat, type(None)):
        panstarrs_cat.write(os.path.join(cat_dir, f'{sn_name}_panstarrs.cat'), format='ascii.csv', overwrite=True)
    return apass_cat, sdss_cat, panstarrs_cat
    
def query_apass(sn_coords, width, height):
    viz = Vizier()
    apass_cat_num = 'II/336'
    #GMOS FOV is 5.5' sq 
    apass_cat = viz.query_region(sn_coords, width=f'{width}m', height=f'{height}m', catalog='II/336')[0]
    if not isinstance(apass_cat, type(None)):
        apass_cat['Bmag'] = apass_cat['B-V']+apass_cat['Vmag']
        apass_cat['e_Bmag'] = np.sqrt(apass_cat['e_B-V']**2 + apass_cat['e_Vmag']**2)
        apass_cat.rename_columns(["RAJ2000", "DEJ2000", "g'mag", "e_g'mag", "r'mag", "e_r'mag", "i'mag", "e_i'mag"], 
                                 ['ra', 'dec','gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag'])
        print(f'APASS Catalog contains {len(apass_cat)} stars')
    else:
        print('APASS Catalog returned no matches')
    return apass_cat

def query_sdss(sn_coords, width, height):
    sql_command = "select ra, dec, objID,u, err_u, g, err_g, r, err_r, i, err_i, z, err_z " + \
                  "from PhotoPrimary " + \
                  "where type=6 and ra>={} and ra < {} and dec>={} and dec < {}"
    sql_command = sql_command.format((sn_coords.ra-width*u.arcmin).value, (sn_coords.ra+width*u.arcmin).value,
                                (sn_coords.dec-height*u.arcmin).value, (sn_coords.dec+height*u.arcmin).value)
    sdss_cat = SDSS.query_sql(sql_command)
    if not isinstance(sdss_cat, type(None)):
        sdss_cat.rename_columns(['u', 'err_u', 'g', 'err_g', 'r', 'err_r', 'i', 'err_i', 'z', 'err_z'], 
                           ['umag', 'e_umag', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag'])
        print(f'SDSS Catalog contains {len(sdss_cat)} stars')
    else:
        print('SDSS Catalog returned no matches')
    return sdss_cat

def query_panstarrs(sn_coords, width, height, extended_mag_cut):
    TAP_service = vo.dal.TAPService("https://mast.stsci.edu/vo-tap/api/v0.1/ps1dr2/")
    sql_command = """
    SELECT objID, ramean, decmean, nDetections, ng,nr, ni, gmeanpsfmag,rmeanpsfmag, imeanpsfmag,gmeanpsfmagerr,rmeanpsfmagerr, imeanpsfmagerr, gmeankronmag,rmeankronmag, imeankronmag,gmeankronmagerr,rmeankronmagerr, imeankronmagerr
    FROM dbo.MeanObjectView
    WHERE
    ramean>={} and ramean < {} and decmean>={} and decmean < {}
    AND ((nr > 1) OR (ni > 1))
    """
    sql_command = sql_command.format((sn_coords.ra-width*u.arcmin).value, (sn_coords.ra+width*u.arcmin).value,
                                (sn_coords.dec-height*u.arcmin).value, (sn_coords.dec+height*u.arcmin).value)
    job = TAP_service.run_sync(sql_command)
    panstarrs_cat = job.to_table()
    if not isinstance(panstarrs_cat, type(None)):
        panstarrs_cat.rename_columns(['ramean', 'decmean', 'gmeanpsfmag','rmeanpsfmag', 'imeanpsfmag','gmeanpsfmagerr','rmeanpsfmagerr', 'imeanpsfmagerr'],
                                     ['ra'    , 'dec'   , 'gmag'      , 'rmag', 'imag', 'e_gmag', 'e_rmag', 'e_imag'])
        print(f'PanSTARRs Catalog contains {len(panstarrs_cat)} stars')
        #I set this fairly arbitrarily - the PS documentation doesn't say how it sets this - just that the difference indicates extendedness
        point_source_indx = (np.abs(panstarrs_cat[f'rmag']-panstarrs_cat[f'rmeankronmag'])<extended_mag_cut) & \
                            (np.abs(panstarrs_cat[f'imag']-panstarrs_cat[f'imeankronmag'])<extended_mag_cut) 
        panstarrs_cat = panstarrs_cat[point_source_indx]
        print(f'Removing {np.sum(~point_source_indx)} Extended stars')
    else:
        print('PanSTARRs catalog returned no matches')
    return panstarrs_cat

def select_psf_stars(cat, wcs, filt, img, filename, visualize=False, faint_lim=20, bright_lim=18, poly=None):
    
    x, y = wcs.wcs_world2pix(cat['ra'], cat['dec'], 0)
    if not poly:
        poly = mpl.patches.Polygon(np.array([[875, 625, 625,  875,  2350, 2575, 2575, 2350],
                                             [125,   325, 1850, 2075, 2075, 1850, 325,  125]]).T, closed=True, color='orange', alpha=0.4)
    
    #check that points are in image and between 15th and 20th mag
    in_frame_indx = poly.contains_points(np.array([x,y]).T)
    brightness_indx = (cat[f'{filt}mag']>bright_lim) & (cat[f'{filt}mag']<faint_lim)
    indx =  in_frame_indx & brightness_indx
    matched_cat = cat[indx]
    matched_cat['x'] = x[indx]
    matched_cat['y'] = y[indx]
    isolated_mask = np.bool_(np.ones(np.sum(indx)))
    
    #check that stars are more than 50 pixels from other PSF stars
    for rownum, irow in enumerate(matched_cat):
        dist = np.sqrt((x-irow['x'])**2 + (y-irow['y'])**2)#check for nearby stars in the full catalog, not just bright stars
        close_indx = (dist < 50) & (dist>0)
        if close_indx.sum()!= 0:
            isolated_mask[rownum]=False
    psf_cat = matched_cat[isolated_mask]
    if visualize:
        vmin, vmax =  vis.zscale(img)
        plt.figure()
        plt.imshow(img, vmin=vmin, vmax=vmax, cmap='bone', aspect='equal')
        plt.plot(x[in_frame_indx], y[in_frame_indx], 'o', mec='c', mfc='none', alpha=0.5, label='cat stars in frame')
        plt.plot(matched_cat['x'], matched_cat['y'], 'o', mec='y', mfc='none', alpha=0.5, label='cat stars meet brightness criteria')
        plt.plot(matched_cat['x'][isolated_mask], matched_cat['y'][isolated_mask], 'o', mec='r', mfc='none', label='selected stars')
        plt.gca().add_patch(poly)
        plt.legend(fontsize=3, loc='upper right')
        plt.savefig(f'../figures/{os.path.splitext(os.path.basename(filename))[0]}_selected_psf_stars.pdf')
        plt.close()
    return psf_cat

def visualize_extract_stars(stars):
    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                           squeeze=True)
    ax = ax.ravel()
    for i in range(min(len(stars), nrows*ncols)):
        vmin, vmax = vis.zscale(stars[i].data)
        ax[i].imshow(stars[i], vmin=vmin, vmax=vmax, aspect='equal')

def background_subtract_stars(img, length, psf_cat):
    for irow in psf_cat:
        #print(irow['x'], irow['y'], irow['rmag'])
        cutout = img[int(np.floor(irow['y']-length/2)):int(np.floor(irow['y']+length/2)),
                     int(np.floor(irow['x']-length/2)):int(np.floor(irow['x']+length/2))
                     ]
        if cutout.shape != (length,length):
            import pdb; pdb.set_trace()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
        vmin, vmax= vis.zscale(cutout)
        ax1.imshow(cutout, vmin=vmin, vmax=vmax, aspect='equal')
        mask = np.bool_(np.zeros(cutout.shape))
        mask[length//2-10:length//2+15, length//2-15:length//2+10]=True
        masked_cutout = np.copy(cutout)
        masked_cutout[mask]=np.nan
        ax2.imshow(masked_cutout,vmin=vmin, vmax=vmax, aspect='equal')
        bkg = Background2D(cutout, box_size=5, mask=mask, exclude_percentile=90)
        img[int(np.floor(irow['y']-length/2)):int(np.floor(irow['y']+length/2)),
                     int(np.floor(irow['x']-length/2)):int(np.floor(irow['x']+length/2))
                     ] = cutout-bkg.background

        ax3.imshow(bkg.background, vmin=vmin, vmax=vmax, aspect='equal')
        ax4.imshow(cutout-bkg.background, vmin=vmin, vmax=vmax, aspect='equal')
    return img

def background_subtract_img(img, length, psf_cat, filename, fig_dir='../figures'):
    mask = np.bool_(np.zeros(img.shape))
    for irow in psf_cat:
        mask[int(np.floor((irow['y'])))-15:int(np.floor((irow['y'])))+15, 
            int(np.floor(irow['x']))-15:int(np.floor(irow['x']))+15] = True
    masked_img = np.copy(img)
    masked_img[mask]=np.nan
    bkg = Background2D(img, box_size=50, mask=mask)
    img_bkg_sub = img - bkg.background
    fig, ax = plt.subplots(nrows=len(psf_cat),ncols=4, figsize=(5, len(psf_cat)))
    for indx, irow in enumerate(psf_cat):
        cutout = img[int(np.floor(irow['y']-length/2)):int(np.floor(irow['y']+length/2)),
                     int(np.floor(irow['x']-length/2)):int(np.floor(irow['x']+length/2))
                     ]
        masked_cutout = masked_img[int(np.floor(irow['y']-length/2)):int(np.floor(irow['y']+length/2)),
                     int(np.floor(irow['x']-length/2)):int(np.floor(irow['x']+length/2))
                     ]
        bkg_cutout = bkg.background[int(np.floor(irow['y']-length/2)):int(np.floor(irow['y']+length/2)),
                     int(np.floor(irow['x']-length/2)):int(np.floor(irow['x']+length/2))
                     ]
        img_bkg_sub_cutout = img_bkg_sub[int(np.floor(irow['y']-length/2)):int(np.floor(irow['y']+length/2)),
                     int(np.floor(irow['x']-length/2)):int(np.floor(irow['x']+length/2))
                     ]
        
        vmin, vmax= vis.zscale(cutout)
        vmin2, vmax2= vis.zscale(img_bkg_sub_cutout)
        ax[indx, 0].imshow(cutout, vmin=0, vmax=vmax, aspect='equal')
        ax[indx, 1].imshow(masked_cutout, vmin=0, vmax=vmax, aspect='equal')
        ax[indx, 2].imshow(bkg_cutout, vmin=0, vmax=vmax, aspect='equal')
        ax[indx, 3].imshow(img_bkg_sub_cutout, vmin=0, vmax=vmax2, aspect='equal')
    plt.savefig(os.path.join(fig_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_background_subtraction.pdf'))
    plt.close()
    return img_bkg_sub

def build_psf(cat, filename, filt, ext=1, visualize=False, backsub_size=108, cutout_size=51, faint_lim=20, bright_lim=18, fig_dir='../figures', poly=None):
    """
    backsub_size: size of region that is background subtracted in image
    cutout_size: size of region that is used to calculate the PSF
    """
    ofile = fits.open(filename)
    img = ofile[ext].data
    hdr1 = ofile[ext].header
    wcs = WCS(hdr1)
    psf_cat = select_psf_stars(cat, wcs, filt, img, filename, visualize=visualize, faint_lim=faint_lim, bright_lim=bright_lim, poly=poly)
    print(f'{len(psf_cat)} stars selected for PSF and zeropoint fitting')
    #img = background_subtract_stars(img, backsub_size, psf_cat)
    img = background_subtract_img(img, backsub_size, psf_cat, filename, fig_dir=fig_dir)
    nddata = NDData(img)
    stars = extract_stars(nddata, psf_cat[['x', 'y']], size=cutout_size)
    if visualize == True:
        visualize_extract_stars(stars)
        plt.savefig(os.path.join(fig_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_extracted_stars.pdf'))
        plt.close()
    epsf_builder = EPSFBuilder(oversampling=1, shape=(cutout_size,cutout_size),maxiters=15, progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    x_centroid=[]
    y_centroid=[]
    for istar in fitted_stars:
        x_centroid.append(istar.center[0])
        y_centroid.append(istar.center[1])
    psf_cat['x_centroid'] = x_centroid
    psf_cat['y_centroid'] = y_centroid
    if visualize == True:
        plt.figure()
        norm = simple_norm(epsf.data, 'log', percent=99.0)
        plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis', aspect='equal')
        plt.colorbar()
        plt.savefig(os.path.join(fig_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_psf.pdf'))
        plt.close()

    return img, epsf, psf_cat

def do_photometry(obj_cat, img, epsf, visualize=True, cutout_size=(51,51), background=None):
    psfphot = PSFPhotometry(epsf, cutout_size,  
                  aperture_radius=13, #for initial flux estimate
                  localbkg_estimator=background) #default is median - consider Background2D?
    phot = psfphot(img, init_params=obj_cat['x_centroid', 'y_centroid'])
    temp_cat = obj_cat['x_centroid', 'y_centroid']
    temp_cat.rename_columns(['x_centroid', 'y_centroid'], ['x', 'y'])
    if visualize:
        resid = psfphot.make_residual_image(img)
        stars = extract_stars(NDData(img), temp_cat, size=51)
        stars_mod = extract_stars(NDData(img-resid), temp_cat, size=51)
        stars_resid = extract_stars(NDData(resid), temp_cat, size=51)

        fig, ax = plt.subplots(ncols=3, nrows=len(stars), figsize=(5, 1.5*len(stars)))
        for irow, istar in enumerate(stars):
            vmin, vmax =  vis.zscale(istar.data)
            if len(stars) == 1:
                im = ax[0].imshow(istar.data, vmin=vmin, vmax=vmax, aspect='equal')
                ax[1].imshow(stars_mod[irow], vmin=vmin, vmax=vmax, aspect='equal')
                ax[2].imshow(stars_resid[irow], vmin=vmin, vmax=vmax, aspect='equal')
                ax[0].set_title('Data')
                #ax[0].contour(istar.data)
                plt.colorbar(mappable=im)
                ax[1].set_title('Model')
                ax[2].set_title('Residual Image')
            else:
                ax[irow, 0].imshow(istar.data, vmin=vmin, vmax=vmax, aspect='equal')
                ax[irow, 1].imshow(stars_mod[irow], vmin=vmin, vmax=vmax, aspect='equal')
                ax[irow, 2].imshow(stars_resid[irow], vmin=vmin, vmax=vmax, aspect='equal')

                ax[0, 0].set_title('Data')
                ax[0, 1].set_title('Model')
                ax[0, 2].set_title('Residual Image')
        plt.tight_layout()
        return phot, fig
    return phot

def calc_mag(flux):
    return -2.5*np.log10(flux)

def calc_zeropoint(filter, zeropoint_cat, psf_cat, filename, visualize=True, fig_dir='../figures'): 
    zeropoint_cat[f'inst_{filter}mag'] = calc_mag(zeropoint_cat['flux_fit'])
    zeropoint_cat[f'{filter}mag'] = psf_cat[f'{filter}mag']
    zeropoint = np.nanmedian(zeropoint_cat[f'{filter}mag'].value - zeropoint_cat[f'inst_{filter}mag'])
    if visualize:
        plt.figure()
        plt.plot(zeropoint_cat[f'{filter}mag'], zeropoint_cat[f'inst_{filter}mag'], '.')
        plt.savefig(os.path.join(fig_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_zeropoint.pdf'))
        plt.close()
    return float(zeropoint) #sometimes this is a MaskedNDArray with a single value

def make_sn_coord_table(filename, sn_coords, img, cutout_size=10, ext=1):
    wcs = WCS(fits.getheader(filename, ext))
    sn_pix = wcs.wcs_world2pix(sn_coords.ra, sn_coords.dec, 0)
    sn_cutout = img[int(np.floor(sn_pix[1]))-cutout_size:int(np.floor(sn_pix[1]))+cutout_size, int(np.floor(sn_pix[0]))-cutout_size:int(np.floor(sn_pix[0]))+cutout_size]
    x_centroid, y_centroid = centroid_com(sn_cutout)
    sn_cat = Table(names=['x', 'y'])
    sn_cat.add_row(sn_pix)
    sn_cat['x_centroid']=x_centroid + int(np.floor(sn_pix[0]))-cutout_size
    sn_cat['y_centroid']=y_centroid + int(np.floor(sn_pix[1]))-cutout_size
    if (np.abs(sn_cat['x_centroid']-sn_cat['x'])>5) | (np.abs(sn_cat['y_centroid']-sn_cat['y'])>5):
        print(f"Centroid ({sn_cat['x_centroid']:2.1f}, {sn_cat['y_centroid']:2.2f}) is >5 pix from estimated location ({sn_cat['x']:2.2f}, {sn_cat['y']:2.2f})")
    return sn_cat

def do_aperture_photometry(img, epsf, sn_cat, visualize=True, fig_dir='../figures', nfwhm=2, aper_annulus_offset=1):
    #fit a 2D Gaussian to ePSF image
    model = models.Gaussian2D(amplitude=np.argmax(epsf.data),x_mean=epsf.data.shape[1]/2, y_mean=epsf.data.shape[0]/2)
    y, x = np.mgrid[:epsf.data.shape[0], :epsf.data.shape[1]]
    fitter = fitting.LMLSQFitter()
    fit = fitter(model, x, y,epsf.data)
    fig, [ax1, ax2, ax3] = plt.subplots(figsize=(8, 2.5), ncols=3)
    if visualize:
        plt.figure()
        ax1.imshow(np.log(epsf.data), origin='lower', interpolation='nearest')
        ax1.set_title("Data")
        ax2.imshow(fit(x, y), origin='lower', interpolation='nearest', vmin=0,
                vmax=0.02)
        ax2.set_title("Model")
        ax3.imshow(epsf.data - fit(x, y), origin='lower', interpolation='nearest', vmin=0,
                vmax=0.02)
        ax3.set_title("Residual")

    #Use FWHM to define circular aperture and annulus aperture
    aper_radius = nfwhm
    source_aper = CircularAperture(r=aper_radius*np.mean([fit.x_fwhm, fit.y_fwhm]), 
                                positions=[(sn_cat['x'][0], sn_cat['y'][0])])
    bkg_annulus = CircularAnnulus(r_in=(aper_radius+aper_annulus_offset)*np.mean([fit.x_fwhm, fit.y_fwhm]), 
                                r_out=(aper_radius+aper_annulus_offset+1)*np.mean([fit.x_fwhm, fit.y_fwhm]), 
                                positions=[(sn_cat['x'][0], sn_cat['y'][0])])

    if visualize:
        fig_aper = plt.figure()
        vmin, vmax = vis.zscale(img[int(sn_cat['y'])-25:int(sn_cat['y'])+25, int(sn_cat['x'])-25:int(sn_cat['x'])+25])
        plt.imshow(img, interpolation='nearest', vmin=vmin, vmax=vmax)
        ap_patch = source_aper.plot(color='white', lw=1,
                           label='Photometry aperture')
        bkg_patch = bkg_annulus.plot(color='red', lw=1,
                           label='background annulus', ls=':')
        plt.xlim(sn_cat['x'][0]-75, sn_cat['x'][0]+75)
        plt.ylim(sn_cat['y'][0]-75, sn_cat['y'][0]+75)
    #Pass to aperture photometry
    sn_phot = aperture_photometry(img, source_aper)
    #background
    bkg_phot = aperture_photometry(img, bkg_annulus)
    #scale bkg counts to source aperture size
    aperstats = ApertureStats(img, bkg_annulus)
    bkg_mean = aperstats.mean
    aper_area = source_aper.area
    aperture_area = source_aper.area_overlap(img)
    total_bkg = bkg_mean * aperture_area
    #background subtract
    sn_phot['aperture_sum_bkgsub'] = sn_phot['aperture_sum']-total_bkg
    return sn_phot, fig, fig_aper

