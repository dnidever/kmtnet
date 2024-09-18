#!/usr/bin/env python

#AUTHORS: David Nidever (original author)
#         david.nidever@montana.edu
#         Katie Fasbender (adapted for analysis on MSU Tempest Research Cluster)
#         katiefasbender@montana.edu
#
# NSC_INSTCAL_MEAS.PY -- Run SExtractor and DAOPHOT on an exposure from the
# NOIRLab Astro Data Archive (NOIRLab Source Catalog measurements procedure)

#-------------------------------------------------
# Imports
#-------------------------------------------------
from argparse import ArgumentParser
from astropy.io import fits
import astropy.stats
from astropy.table import Table, Column,vstack
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from glob import glob
import logging
import numpy as np
import os
import re
from scipy.ndimage import convolve
#from scipy.ndimage.filters import convolve
import shutil
import socket
import struct
import subprocess
import sys
import time
import warnings
import requests
from dlnpyutils import utils as dln,coords
import prometheus as pm
from . import phot,utils

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

## Load default DECam chip data
#if os.path.exists(utils.datadir()+'params/decam_chip_data.fits'):
#    DECAM_DATA = Table.read(utils.datadir()+'params/decam_chip_data.fits')
#else:
#    print('Could not find decam_chip_data.fits file')
#    DECAM_DATA = None

    
#-------------------------------------------------
# Functions
#-------------------------------------------------


# Class to represent an exposure to process
class Exposure:

    # Initialize Exposure object
    def __init__(self,filename,host):
        filename = os.path.abspath(filename)
        # Check that the files exist
        if os.path.exists(filename) is False:
            print(filename+" NOT found")
            return
        #self.delete = delete  # delete original files
        # Setting up the object properties
        self.origfilename = filename
        self.host = host
        self.filename = None      # working files in temp dir
        base = os.path.basename(filename)
        base = os.path.splitext(os.path.splitext(base)[0])[0]
        self.base = base
        self.logfile = base+".log"
        self.logger = None
        self.origdir = None
        self.workdir = None     # the temporary working directory
        self.keepdir = None     # where to keep the final files before bundling
        self.outdir = None
        self.chip = None

        # Get number of extensions
        hdulist = fits.open(filename)
        nhdu = len(hdulist)
        hdulist.close()
        self.nexten = nhdu
        self.nchips = nhdu-1
        # Get night
        head0 = fits.getheader(filename,0)
        dateobs = head0.get("DATE-OBS")
        night = dateobs[0:4]+dateobs[5:7]+dateobs[8:10]
        self.night = night
        # Output directory
        basedir,tmpdir = utils.getdirs(self.host)
        self.outdir = os.path.join('/home/x51j468/kmtnet/',self.night,self.base)
        
    # Setup
    def setup(self):
        basedir,tmproot = utils.getdirs(self.host)
        print("dirs, setup = ",basedir,tmproot)
        # Prepare temporary directory
        tmpcntr = 1
        tmpdir = os.path.join(tmproot,self.base+"."+str(tmpcntr))
        print("temp dir = ",tmpdir)
        while (os.path.exists(tmpdir)):
            tmpcntr = tmpcntr+1
            tmpdir = os.path.join(tmproot,self.base+"."+str(tmpcntr))
            if tmpcntr > 20:
                print("Temporary Directory counter getting too high. Exiting")
                sys.exit()
        if os.path.exists(tmpdir)==False:
            os.makedirs(tmpdir)
        origdir = os.getcwd()
        self.origdir = origdir
        os.chdir(tmpdir)
        self.workdir = tmpdir
        self.keepdir = os.path.join(tmpdir,'keep')
        if os.path.exists(self.keepdir)==False:
            os.makedirs(self.keepdir)
        
        # Set up logging to screen and logfile
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger()
        # file handler
        fileHandler = logging.FileHandler(self.logfile)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        # console/screen handler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
        rootLogger.setLevel(logging.NOTSET)
        self.logger = rootLogger
        self.logger.info("Setting up in temporary directory "+tmpdir)
        self.logger.info("Starting logfile at "+self.logfile)

        # Copy over images from zeus1:/mss or Download images from Astro Data Archive
        filename = "bigfile.fits"
        shutil.copyfile(self.origfilename,os.path.join(tmpdir,os.path.basename(self.origfilename)))
        self.logger.info("  "+self.origfilename)
        if (os.path.basename(self.origfilename) != filename):
            os.symlink(os.path.basename(self.origfilename),filename)

        # Set local working filenames
        self.filename = filename
        
        # Make final output directory
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)   # will make multiple levels of directories if necessary
            self.logger.info("Making output directory: "+self.outdir)

    def __len__(self):
        return self.nchips
            
    # Load chip
    def loadchip(self,extension,filename="flux.fits"):
        # Load the data
        self.logger.info(" Loading chip "+str(extension))
        # Check that the working files set by "setup"
        if (self.filename is None):
            self.logger.warning("Local working filenames not set.  Make sure to run setup() first")
            return(False)
        try:
            flux,fhead = fits.getdata(self.filename,extension,header=True)
            fhead0 = fits.getheader(self.filename,0)  # add PDU info
            fhead.extend(fhead0,unique=True)
        except:
            self.logger.error("No extension "+str(extension))
            return(False)
        # Write the data to the appropriate files
        if os.path.exists(filename):
            os.remove(filename)
        fits.writeto(filename,flux,header=fhead,output_verify='warn')
        # Create the chip object
        self.chip = Chip(filename,self.base,self.host)
        self.chip.meta['ccdnum'] = extension
        self.chip._ccdnum = extension
        self.chip.bigextension = extension
        self.chip.outdir = self.outdir
        self.chip.keepdir = self.keepdir
        # Add logger information
        self.chip.logger = self.logger
        return True

    # Process all chips
    def process(self):
        self.logger.info("-------------------------------------------------")
        self.logger.info("Processing ALL extension images")
        self.logger.info("-------------------------------------------------")

        # LOOP through the HDUs/chips
        #----------------------------
        for i in range(1,self.nexten):
            t0 = time.time()
            self.logger.info(" ")
            self.logger.info("=== Processing subimage "+str(i)+" ===")
            # Load the chip
            bl = self.loadchip(i)
            self.logger.info("CCDNUM = "+str(self.chip.ccdnum))
            if bl==True:
                # Process it
                self.chip.process()
                # Clean up
                self.chip.cleanup()
            self.logger.info("dt = "+str(time.time()-t0)+" seconds")
            if 2==1:
                chiptimes = Table.read(basedir+'lists/nsc_dr3_chiptimes.fits')
                chiptimes.add_row([(str(self.filename).strip().split('/')[-1]).split('.')[0],i,
                                   int(self.chip.ccdnum),int(nsrc),int(t1_check-t0)])
                chiptimes = Table(np.unique(chiptimes))
                chiptimes.write(basedir+'lists/nsc_dr3_chiptimes.fits',overwrite=True)

    # Teardown
    def teardown(self):
        # Move the final log file
        shutil.move(self.logfile,os.path.join(self.keepdir,self.base+".log"))
        # Bundle files in the "keep" directory
        utils.concatmeas(self.keepdir,self.base)
        # Move the final bundled files
        finalfiles = [os.path.join(self.keepdir,self.base+f) for f in ['_meas.fits','.tgz','.log']]
        for f in finalfiles:
            if os.path.exists(f):
                self.logger.info('Moving '+f+' to '+self.outdir)
                shutil.move(f,os.path.join(self.outdir,os.path.basename(f)))
            else:
                self.logger.info(f+'not found')
        # Delete files and temporary directory
        self.logger.info("Deleting files and temporary directory.")
        self.logger.info('Removing '+self.workdir)
        shutil.rmtree(self.workdir)
        # CD back to original directory
        os.chdir(self.origdir)

    # RUN all steps to process this exposure
    def run(self):
        self.setup()
        self.process()
        self.teardown()

# Class to represent a single chip of an exposure
class Chip:

    def __init__(self,filename,bigbase,host):
        self.filename = filename
        self.bigbase = bigbase
        self.host = host
        if host=="tempest" or host=="tempest_group": self.bindir = "/home/x51j468/bin/"
        else: self.bindir = os.path.expanduser("~/bin/")
        self.bigextension = None
        base = os.path.basename(filename)
        base = os.path.splitext(os.path.splitext(base)[0])[0]
        self.dir = os.path.abspath(os.path.dirname(filename))
        self.base = base
        self.keepdir = None
        header = fits.getheader(filename)
        self.meta = phot.makemeta(header=header)        
        # Make wt and mask files
        im,head = fits.getdata(filename,header=True)
        # Saturated pixels
        self.meta['saturate'] = 60000
        badpix = (im>60000)
        # Create error image
        noise = np.sqrt(np.maximum(im,0)/self.gain + self.rdnoise**2)
        noise = np.maximum(noise,1)
        wt = 1/noise**2
        wt[badpix] = 0
        fits.writeto('wt.fits',wt)
        self.wtfile = 'wt.fits'
        # Create mask image
        mask = np.zeros(im.shape,np.int16)
        mask[badpix] = 1
        fits.writeto('mask.fits',mask)
        self.maskfile = 'mask.fits'
        self.sexfile = self.dir+"/"+self.base+"_sex.fits"
        self.daofile = self.dir+"/"+self.base+"_dao.fits"
        self.sexcatfile = None
        self.sexcat = None
        self.seeing = None
        self.apcorr = None
        # For the second run of SExtractor on the ALLSTAR PSF-subtracted file
        self.allsubfile = self.dir+"/"+self.base+"_daos.fits"
        self.smeta = None
        self.sexcatfile2 = None
        self.sexcat2 = None
        # Internal hidden variables
        self._rdnoise = None
        self._gain = None
        self._ccdnum = None
        self._pixscale = None
        self._saturate = None
        self._wcs = None
        self._exptime = None
        self._plver = None
        self._daomaglim = None    # set by daoaperphot()
        self._sexmaglim = None    # set by runsex()
        self.sexiter = 1          #ktedit:sex2; to keep track of which SExtractor run we're on
        # Logger
        self.logger = None
        # Estimate FWHM=
        im,head = fits.getdata(self.filename,header=True)
        im = pm.ccddata.CCDData(im,header=head)
        objects = pm.detection.detect(im,nsigma=10)
        objects = pm.aperture.aperphot(im,objects)
        fwhmpix = pm.utils.estimatefwhm(objects)
        fwhmarcsec = fwhmpix*self.meta['pixscale']
        self._fwhm = fwhmarcsec
        self.meta['fwhm'] = fwhmarcsec
        
    def __repr__(self):
        return "Chip object"        
        
    @property
    def rdnoise(self):
        # We have it already, just return it
        if hasattr(self,'_rdnoise') and self._rdnoise is not None:
            return self._rdnoise
        # Can't get rdnoise, no header yet
        if self.meta is None:
            self.logger.warning("Cannot get RDNOISE, no header yet")
            return None
        # Get rdnoise from the header
        for name in ['RDNOISE','READNOIS','ENOISE']:
            # We have this key, set _rndoise and return
            if name in self.meta.keys():
                self._rdnoise = float(self.meta[name])
                return self._rdnoise
        self.logger.warning('No RDNOISE found')
        return None
            
    @property
    def gain(self):
        # We have it already, just return it
        if hasattr(self,'_gain') and self._gain is not None:
            return self._gain
        try:
            gain = float(self.meta['gain'])
        except:
            raise Exception('no gain')
        self._gain = gain
        return self._gain
            
    @property
    def ccdnum(self):
        # We have it already, just return it
        if self._ccdnum is not None:
            return self._ccdnum
        # Can't get ccdnum, no header yet
        if self.meta is None:
            self.logger.warning("Cannot get CCDNUM, no header yet")
            return None
        # Get ccdnum from the header
        # We have this key, set _rndoise and return
        if 'CCDNUM' in self.meta.keys():
            self._ccdnum = self.meta['CCDNUM']
            return self._ccdnum
        self.logger.warning('No CCDNUM found')
        return None
            
    @property
    def pixscale(self):
        # We have it already, just return it
        if self._pixscale is not None:
            return self._pixscale
        pixmap = { 'c4d': 0.27, 'k4m': 0.258, 'ksb': 0.45 }
        try:
            pixscale = pixmap[self.instrument]
            self._pixscale = pixscale
            return self._pixscale
        except:
            self._pixscale = np.max(np.abs(self.wcs.pixel_scale_matrix))
            return self._pixscale
            
    @property
    def saturate(self):
        # We have it already, just return it
        if self._saturate is not None:
            return self._saturate
        # Can't get saturate, no header yet
        if self.meta is None:
            self.logger.warning("Cannot get SATURATE, no header yet")
            return None
        # Get saturate from the header
        # We have this key, set _saturate and return
        if 'SATURATE' in self.meta.keys():
            self._saturate = self.meta['SATURATE']
            return self._saturate
        self.logger.warning('No SATURATE found')
        return None
    
    @property
    def wcs(self):
        # We have it already, just return it
        if self._wcs is not None:
            return self._wcs
        # Can't get wcs, no header yet
        if self.meta is None:
            self.logger.warning("Cannot get WCS, no header yet")
            return None
        try:
            self._wcs = WCS(self.meta)
            return self._wcs
        except:
            self.logger.warning("Problem with WCS")
            return None
            
    @property
    def exptime(self):
        # We have it already, just return it
        if self._exptime is not None:
            return self._exptime
        # Can't get exptime, no header yet
        if self.meta is None:
            self.logger.warning("Cannot get EXPTIME, no header yet")
            return None
        # Get rdnoise from the header
        # We have this key, set _rndoise and return
        if 'EXPTIME' in self.meta.keys():
                self._exptime = self.meta['EXPTIME']
                return self._exptime
        print('No EXPTIME found')
        return None

    @property
    def fwhm(self):
        # We have it already, just return it
        if self._fwhm is not None:
            return self._fwhm
        from prometheus import prometheus as pm
        im,head = fits.getdata(self.filename,header=True)
        im = pm.ccddata.CCDData(im,header=head)
        objects = pm.detection.detect(im,nsigma=10)
        objects = pm.aperture.aperphot(im,objects)
        fwhm = pm.utils.estimatefwhm(objects)
        self._fwhm = fwhm
        self.meta['fwhm'] = fwhm
        return self._fwhm

    @property
    def maglim(self):
        # We have it already, just return it
        if self._daomaglim is not None:
            return self._daomaglim
        if self._sexmaglim is not None:
            return self._sexmaglim
        self.logger.warning('Maglim not set yet')
        return None

    # Write SE catalog in DAO format
    #-------------------------------
    def sextodao(self,cat=None,outfile=None,format="coo",meta=None):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        if outfile is None:
            outfile = daobase+".coo"
        if cat is None:
            cat = self.sexcat
        if meta is None:
            meta = self.meta
        else:
            offs = 0
        phot.sextodao(cat,meta,outfile=outfile,format=format,logger=self.logger)

    # Run Source Extractor
    #---------------------
    def runsex(self,dthresh=1.1,bindir="~/bin/",outfile=None):
        if self.sexiter==1: 
            infile = self.filename
            meta = self.meta
            sexcatfile = "flux_sex.cat.fits"
            offset = 0
        else:
            daobase = os.path.basename(self.daofile)
            daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]             
            infile = daobase+str(self.sexiter-1)+"s.fits"
            self.smeta = phot.makemeta(header=fits.getheader(infile,0))  #should this be self.smeta?  probably.
            meta = self.smeta
            sexcatfile = "flux_sex"+str(self.sexiter)+".cat.fits"
            if self.sexcat is not None: offset=int(self.sexcat['NUMBER'][-1]) #ktedit:sex2
        basedir, tmpdir = utils.getdirs(self.host)
        configdir = basedir+"config/"
        sexcat, maglim = phot.runsex(infile,self.wtfile,self.maskfile,meta,sexcatfile,configdir,
                                     offset=offset,sexiter=self.sexiter,dthresh=dthresh,
                                     logger=self.logger,bindir=self.bindir) #ktedit:sex2
        sexcat.add_column(np.repeat(self.sexiter,len(sexcat)),name="NDET_ITER") # keep track of what SExtractor iteration each source is from
        sexcat.add_column(np.zeros(len(sexcat)),name="REPEAT")                  # keep track of sources that were detected in multiple iterations
        # Rules of "REPEAT" column: 
        #  0 = source only detected once
        #  1 = source detected in multiple iterations (all iterations but last), will be removed from sexcat
        #  2 = source detected in multiple iterations (last iteration source was detected in)
        temp_sexcat = Table.read(sexcatfile,2)
        temp_sexcat.add_column(np.repeat(self.sexiter,len(temp_sexcat)),name="NDET_ITER")
        temp_sexcat.add_column(np.zeros(len(temp_sexcat)),name="REPEAT")
        temp_sexcat.write(sexcatfile,overwrite=True)
        # --If first SExtractor iteration, define cat
        if self.sexiter==1:
            self.sexcatfile = sexcatfile
            self.sexcat = sexcat
            self._sexmaglim = maglim
            # Set the FWHM as well
            fwhm = phot.sexfwhm(sexcat,logger=self.logger)
            self.meta['FWHM'] = fwhm
        # --If 2nd+ SExtractor iteration, compare sources with
        # those from previous iteration and combine catalogs 
        else: 
            sexcat = vstack([self.sexcat,sexcat])
            # lastsex -> newsexcat, restsex -> prevsexcat   
            newsexcat = sexcat[sexcat['NDET_ITER']==self.sexiter]
            prevsexcat = sexcat[sexcat['NDET_ITER']==(self.sexiter-1)]
            for newsource in newsexcat:
                dpix = 2
                prevsexcat_close = prevsexcat[(prevsexcat['X_IMAGE']<(newsource['X_IMAGE']+dpix)) & 
                                              (prevsexcat['X_IMAGE']>(newsource['X_IMAGE']-dpix)) & 
                                              (prevsexcat['Y_IMAGE']<(newsource['Y_IMAGE']+dpix)) & 
                                              (prevsexcat['Y_IMAGE']>(newsource['Y_IMAGE']-dpix))]
                if len(prevsexcat_close)>0:
                    for oldsource in prevsexcat_close:
                        d_btwn_centers = np.sqrt((newsource['X_IMAGE']-oldsource['X_IMAGE'])**2 + 
                                                 (newsource['Y_IMAGE']-oldsource['Y_IMAGE'])**2)
                        if d_btwn_centers <= dpix:
                            old_repeat_index = int(np.where(sexcat['NUMBER']==oldsource['NUMBER'])[0])
                            new_repeat_index = int(np.where(sexcat['NUMBER']==newsource['NUMBER'])[0])
                            sexcat[old_repeat_index]['REPEAT'] = 1
                            sexcat[new_repeat_index]['REPEAT'] = 2
            self.sexcat = sexcat[sexcat['REPEAT']!=1]

    # Determine FWHM using SE catalog
    #--------------------------------
    def sexfwhm(self):
        self.seeing = sexfwhm(self.sexcat)
        return self.seeing

    # Pick PSF candidates using SE catalog
    #-------------------------------------
    def sexpickpsf(self,nstars=100):
        base = os.path.basename(self.sexfile)
        base = os.path.splitext(os.path.splitext(base)[0])[0]
        fwhm = self.sexfwhm() if self.seeing is None else self.seeing
        psfcat = phot.sexpickpsf(self.sexcat,fwhm,self.meta,base+".lst",
                                 nstars=nstars,logger=self.logger)

    # Make DAOPHOT option files
    #--------------------------
    def mkopt(self):
        base = os.path.basename(self.daofile)
        base = os.path.splitext(os.path.splitext(base)[0])[0]
        phot.mkopt(base,self.meta,logger=self.logger)
        
    # Make image ready for DAOPHOT
    def mkdaoim(self):
        phot.mkdaoim(self.filename,self.wtfile,self.maskfile,self.meta,self.daofile,logger=self.logger)

    # DAOPHOT detection
    #----------------------
    def daofind(self):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        cat = phot.daofind(self.daofile,outfile=daobase+".coo",logger=self.logger,bindir=self.bindir)

    # DAOPHOT aperture photometry
    #----------------------------
    def daoaperphot(self):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        imfile=self.daofile
        if self.sexiter==1:
            coofile = daobase+".coo"
            outfile = daobase+".ap"
        else:
            coofile = daobase+str(self.sexiter)+".coo"
            outfile = daobase+str(self.sexiter)+".ap"
        apcat, maglim = phot.daoaperphot(imfile,coofile,outfile=outfile,optfile=daobase+".opt",
                                         logger=self.logger,bindir=self.bindir)
        if self.sexiter==1: self._daomaglim = maglim

    # Pick PSF stars using DAOPHOT
    #-----------------------------
    def daopickpsf(self,maglim=None,nstars=100):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        if maglim is None: maglim=self.maglim
        psfcat = phot.daopickpsf(self.daofile,daobase+".ap",maglim,daobase+".lst",nstars,
                                 logger=self.logger,bindir=self.bindir)

    # Run DAOPHOT PSF
    #-------------------
    def daopsf(self,verbose=False):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        psfcat = phot.daopsf(self.daofile,daobase+".lst",outfile=daobase+".psf",
                             verbose=verbose,logger=self.logger,bindir=self.bindir)

    # Subtract neighbors of PSF stars
    #--------------------------------
    def subpsfnei(self):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        psfcat = phot.subpsfnei(self.daofile,daobase+".lst",daobase+".nei",
                                daobase+"a.fits",logger=self.logger,bindir=self.bindir)

    # Create DAOPHOT PSF
    #-------------------
    def createpsf(self,listfile=None,apfile=None,doiter=True,maxiter=5,minstars=6,subneighbors=True,verbose=False):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        subit = phot.createpsf(daobase+".fits",daobase+".ap",daobase+".lst",meta=self.meta,logger=self.logger)
        self.subiter=subit
        
    # Run ALLSTAR
    #-------------
    def allstar(self,psffile=None,apfile=None,subfile=None):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        imfile = daobase+".fits"
        meta = self.meta
        subfile = daobase+str(self.sexiter)+"s.fits"
        if self.sexiter==1: 
            apfile = daobase+".ap"
            outfile = daobase+".als"
        else: 
#            subfile = daobase+str(self.sexiter)+"s.fits"
            apfile = daobase+str(self.sexiter)+".ap"
            outfile = daobase+str(self.sexiter)+".als"
        alscat = phot.allstar(imfile,daobase+".psf",apfile=apfile,subfile=subfile,
                              outfile=outfile,optfile=daobase+".als.opt",meta=meta,
                              logger=self.logger,bindir=self.bindir) #ktedit:sex2


    # Combine total + new SExtractor & ALLSTAR catalog files #ktedit:sex2; this function is new
    #-----------------------------------------------------
    def combine_cats(self,type):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]

        if type=="sexcat":
            file1 = daobase+".coo"                                      # total SExcat stored in this file
            file2 = daobase+str(self.sexiter)+".coo"                    # new SExcat stored in this file
            # must also combie .fits format of SE cat
            file3 = "flux_sex.cat.fits"
            file4 = "flux_sex"+str(self.sexiter)+".cat.fits"
            sexcat1 = Table.read(file3)
            sexcat2 = Table.read(file4)
            sexcat_total = vstack([sexcat1,sexcat2])
            sexcat_total.write(file3,overwrite=True)

        elif type=="alscat":
            file1 = daobase+".als"                                      # total ALLSTAR cat stored in this file
            file2 = daobase+str(self.sexiter)+".als"                    # new ALLSTAR cat stored in this file

        cat1 = readlines(file1)
        cat2 = readlines(file2)
        combined_cat = cat1+cat2[3:]                                    # combine the catalogs 
        writelines(file1,combined_cat,overwrite=True)

    # Get aperture correction
    #------------------------
    def getapcor(self):
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        apcorr = phot.apcor(daobase+"a.fits",daobase+".lst",daobase+".psf",self.meta,
                            optfile=daobase+'.opt',alsoptfile=daobase+".als.opt",logger=self.logger)
        self.apcorr = apcorr
        self.meta['apcor'] = (apcorr,"Aperture correction in mags")

    # Combine SE and DAOPHOT catalogs
    #--------------------------------
    def finalcat(self,outfile=None,both=True,sexdetect=True):
        # both       Only keep sources that have BOTH SE and ALLSTAR information
        # sexdetect  SE catalog was used for DAOPHOT detection list
        self.logger.info("--  Creating final combined catalog --")

        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        if outfile is None: outfile=self.base+".cat.fits"

        # Check that we have the SE and ALS information
        if (self.sexcat is None) | (os.path.exists(daobase+".als") is None):
            self.logger.warning("SE catalog or ALS catalog NOT found")
            return

        # Load ALS catalog
        als = Table(phot.daoread(daobase+".als")) 
        nals = len(als)
        # Apply aperture correction
        if self.apcorr is None:
            self.logger.error("No aperture correction available")
            return
        als['MAG'] -= self.apcorr

        # Just add columns to the SE catalog
        ncat = len(self.sexcat)
        newcat = self.sexcat.copy()
        alsnames = ['X','Y','MAG','ERR','SKY','ITER','CHI','SHARP']
        newnames = ['XPSF','YPSF','MAGPSF','ERRPSF','SKY','ITER',
                    'CHI','SHARP','RAPSF','DECPSF']
        newtypes = ['float64','float64','float','float','float','float',
                    'float','float','float64','float64']
        nan = float('nan')
        newvals = [nan, nan, nan, nan ,nan, nan, nan, nan, nan, nan]
        # DAOPHOT detection list used, need ALS ID
        if not sexdetect:
            alsnames = ['ID']+alsnames
            newnames = ['ALSID']+newnames
            newtypes = ['int32']+newtypes
            newvals = [-1]+newvals
        newcols = []
        for n,t,v in zip(newnames,newtypes,newvals):
            col = Column(name=n,length=ncat,dtype=t)
            col[:] = v
            newcols.append(col)
        newcat.add_columns(newcols)
        # Match up with IDs if SE list used by DAOPHOT
        if sexdetect:
            mid, ind1, ind2 = np.intersect1d(newcat["NUMBER"],als["ID"],return_indices=True)
            for id1,id2 in zip(newnames,alsnames):
                newcat[id1][ind1] = als[id2][ind2]
            # Only keep sources that have SE+ALLSTAR information
            #  trim out ones that don't have ALS
            if (both is True) & (nals<ncat):
                newcat = newcat[ind1]
            #self.logger.info("newcat has "+str(len(newcat))+" lines") #ktedit:sex2

        # Match up with coordinates, DAOPHOT detection list used
        else:
            print("Need to match up with coordinates")
            # Only keep sources that have SE+ALLSTAR information
            #  trim out ones that don't have ALS
            if (both is True) & (nals<ncat): newcat = newcat[ind1]

        # Add RA, DEC
        r,d = self.wcs.all_pix2world(newcat["XPSF"],newcat["YPSF"],1)
        newcat['RAPSF'] = r
        newcat['DECPSF'] = d        
        #self.logger.info("length of final catalog = "+str(len(newcat))) #ktedit:sex2

        # Write to file
        self.logger.info("Final catalog = "+outfile)
        fits.writeto(outfile,None,self.meta,overwrite=True)  # meta in PDU header
        #  append the table in extension 1
        hdulist = fits.open(outfile)
        hdu = fits.table_to_hdu(newcat)
        hdulist.append(hdu)
        hdulist.writeto(outfile,overwrite=True)
        hdulist.close()
        #newcat.write(outfile,overwrite=True)
        #fits.append(outfile,0,self.meta)  # meta is header of 2nd extension


    # Process a single chip
    #----------------------
    def process(self):

        # Set up SE iteration
        sexiter_endflag = 0
        while (sexiter_endflag==0):

            # For every iteration, run Source Extractor
            self.logger.info("-- SExtractor run "+str(self.sexiter)+" --")

            # determine DETECT_THRESH value from SE iteration
            if self.sexiter==1:
                sex_dt = 1.7
            else:
                sex_dt = 1.1

            self.runsex(dthresh=sex_dt,bindir=self.bindir)

            # Get the info for this iteration's catalog
            nowcat = self.sexcat[self.sexcat['NDET_ITER']==self.sexiter]  # cat for current SE iteration
            nowcat = nowcat[nowcat['REPEAT']==0]                          # select only newly detected sources
            self.logger.info(str(len(nowcat))+" new sources detected")
            nowcat_sn5 = nowcat[1/nowcat['MAGERR_AUTO']>=5]               # select only the entries with SN>=5
            if self.sexiter==1: 
                ogcat = nowcat                                            # to check the current cat against the first one
                ogcat_sn5 = nowcat_sn5

            # Perform aperture photometry and PSF fitting with DAOPHOT
            self.logger.info("-- Getting ready to run DAOPHOT --")

            # for first iteration only, make DAO-ready files
            if self.sexiter==1:
                self.mkopt()
                self.mkdaoim()
            
            # Convert SE cat to DAO format
            #self.daodetect()
            # Create DAOPHOT-style coo file
            # Need to use SE positions
            if self.sexiter==1:
                sdao_ofile = "flux_dao.coo"          # select aperphot output filename
            else:
                sdao_ofile = "flux_dao"+str(self.sexiter)+".coo"
            self.sextodao(outfile=sdao_ofile)

            self.daoaperphot()

            # For first iteration only, fit PSF 
            if self.sexiter==1:
                self.daopickpsf()   
                self.createpsf()

            # Combine SE cats, run ALLSTAR, combine ALLSTAR cats
            if self.sexiter>1: self.combine_cats(type="sexcat")           
            self.allstar()
            if self.sexiter>1: self.combine_cats(type="alscat")                      

            # Check to see if we've run enough SExtractor iterations
            # Requirements to end:
            # - at least 2 iterations, no more than 4
            # - median S/N of latest SE cat <=5
            # - #sources for which S/N>=5 latest cat is less than 25% #sources (also S/N>=5) in first cat
            if (self.sexiter>2 and (self.sexiter==4 or (np.median(1/nowcat['MAGERR_AUTO'])<=5) or
                                    (len(nowcat_sn5)<(.25*len(ogcat_sn5))))):
                sexiter_endflag = 1
            self.sexiter += 1

        # Get aperture correction, create final cat from SE + ALLSTAR cats
        self.getapcor()
        self.finalcat()

        # David's notes:------------------------------------------------------------------------------------

        # Do I need to rerun daoaperphot to get aperture
        # photometry at the FINAL allstar positions??
        
        # Is there a way to reduce the number of iterations needed to create the PSF?
        # what do the ?, * mean anyway?
        # maybe just remove the worse 10% of stars or something

        # Put all of the daophot-running into separate function (maybe separate module)
        # same for sextractor

        # Maybe make my own xmatch function that does one-to-one matching

    # Clean up the files
    #--------------------
    def cleanup(self):
        # Move files we want to keep to temporary "keep" subdirectory
        self.logger.info("Copying final files to 'keep' directory "+self.keepdir)
        base = os.path.basename(self.filename)
        base = os.path.splitext(os.path.splitext(base)[0])[0]
        daobase = os.path.basename(self.daofile)
        daobase = os.path.splitext(os.path.splitext(daobase)[0])[0]
        # Copy the files we want to keep
        # final combined catalog, logs
        outcatfile = os.path.join(self.keepdir,self.bigbase+"_"+str(self.ccdnum)+".fits")
        if os.path.exists(outcatfile): os.remove(outcatfile)
        shutil.copyfile("flux.cat.fits",outcatfile)
        # Copy DAOPHOT opt files
        outoptfile = os.path.join(self.keepdir,self.bigbase+"_"+str(self.ccdnum)+".opt")
        if os.path.exists(outoptfile): os.remove(outoptfile)
        shutil.copyfile(daobase+".opt",outoptfile)
        outalsoptfile = os.path.join(self.keepdir,self.bigbase+"_"+str(self.ccdnum)+".als.opt")
        if os.path.exists(outalsoptfile): os.remove(outalsoptfile)
        shutil.copyfile(daobase+".als.opt",outalsoptfile)
        # Copy DAOPHOT PSF star list
        outlstfile = os.path.join(self.keepdir,self.bigbase+"_"+str(self.ccdnum)+".psf.lst")
        if os.path.exists(outlstfile): os.remove(outlstfile)
        shutil.copyfile(daobase+".lst",outlstfile)
        # Copy DAOPHOT PSF file
        outpsffile = os.path.join(self.keepdir,self.bigbase+"_"+str(self.ccdnum)+".psf")
        if os.path.exists(outpsffile): os.remove(outpsffile)
        shutil.copyfile(daobase+".psf",outpsffile)
        # Copy DAOPHOT .apers file??
        # Copy neighbor-subtracted images to output dir #ktedit
        #for i in range(0,int(self.subiter-1)):
        #    if int(i)==int(self.subiter-2): nsub=""
        #    else:nsub=str(i+1)
        #    outnsubfile=self.keepdir+self.bigbase+"_"+str(self.ccdnum)+"_"+nsub+"a.fits"
        #    nsubfile=daobase+nsub+"a.fits"
        #    if os.path.exists(outnsubfile): os.remove(outnsubfile)
        #    shutil.copyfile(nsubfile,outnsubfile)
        # Copy daophot-ready image to output dir
        #outdimfile = self.keepdir+self.bigbase+"_"+str(self.ccdnum)+"daoim.fits"
        #if os.path.exists(outdimfile): os.remove(outdimfile)
        #shutil.copyfile(daobase+".fits",outdimfile)
        # copy Allstar PSF subtracted files to output dir #ktedit
        #for i in range(1,int(self.sexiter)):
        #    outsubfile = self.keepdir+self.bigbase+"_"+str(self.ccdnum)+"_"+str(i)+"s.fits"
        #    if os.path.exists(outsubfile): os.remove(outsubfile)
        #    shutil.copyfile(daobase+str(i)+"s.fits",outsubfile)
        # Copy SE config file
        outconfigfile = os.path.join(self.keepdir,self.bigbase+"_"+str(self.ccdnum)+".sex.config")
        if os.path.exists(outconfigfile): os.remove(outconfigfile)
        shutil.copyfile("default.config",outconfigfile)
        # Copy SE segmentation files       #ktedit:sex2
        #for i in range(1,int(self.sexiter)):
        #    outsegfile=self.keepdir+self.bigbase+"_"+str(self.ccdnum)+"_"+str(i)+"seg.fits"
        #    if os.path.exists(outsegfile): os.remove(outsegfile)
        #    shutil.copyfile("seg_"+str(i)+".fits",outsegfile)

        # Combine all the log files
        logfiles = glob(base+"*.log")
        loglines = []
        for logfil in logfiles:
            loglines += ["==> "+logfil+" <==\n"]
            f = open(logfil,'r')
            lines = f.readlines()
            f.close()
            loglines += lines
            loglines += ["\n"]
        f = open(base+".logs","w")
        f.writelines("".join(loglines))
        f.close()
        outlogfile =  os.path.join(self.keepdir,self.bigbase+"_"+str(self.ccdnum)+".logs")
        if os.path.exists(outlogfile): os.remove(outlogfile)
        shutil.copyfile(base+".logs",outlogfile)

        # Delete temporary directory/files
        self.logger.info("  Cleaning up")
        files1 = glob("flux*")
        files2 = glob("default*")
        files = files1+files2+["flux.fits","wt.fits","mask.fits","daophot.opt","allstar.opt"]
        for f in files:
            if os.path.exists(f): os.remove(f)

#-------------------------------------------------
# Main command-line program
#-------------------------------------------------
if __name__ == "__main__":

    # Setup
    #------
    # Initiate input arguments
    parser = ArgumentParser(description='Run NSC Instcal Measurement Process on one Exposure.')
    parser.add_argument('--filename',type=str,nargs=1,help='Full path to fluxfile')
    parser.add_argument('--host',type=str,nargs=1,default="None",help='hostname, default "None", other options supported are "cca","tempest_katie","tempest_group","gp09/7"')
    parser.add_argument('-r','--redo', action='store_true', help='Redo exposures that were previously processed')
    args = parser.parse_args()


    # Inputs  
    host = str(args.host[0])                 # hostname of server, default "None"
    if host=="None": host = None
    redo = args.redo                         # if called, redo = True
    print("host = ",host," redo = ",redo)
    
    # Get directories
    basedir, tmpdir = utils.getdirs(host)
    print("Working in basedir,tmpdir = ",basedir,tmpdir)
    # Make sure the directories exist
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # Get file names
    filename = args.filename[0]
    # Check that the files exist
    if os.path.exists(filename) is False:
        print(filename+" file NOT FOUND")
        sys.exit()

    # Start keeping time
    t0 = time.time()
    
    # Create the Exposure object
    exp = Exposure(filename,host=host)
    # Run
    exp.run()

    print("Total time = {:.2f} seconds".format(time.time()-t0))
