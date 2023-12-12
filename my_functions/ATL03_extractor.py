import os, h5py, glob, sys, warnings, tqdm
import pandas as pd
import numpy as np
import geopandas as gp
from pyproj import Transformer
from pyproj import proj
from operator import attrgetter

# the following functions (getCoordRotFwd, get_atl_alongtrack) are taken, unchanged, from https://github.com/UP-RS-ESP/ICESat-2_SVDA/tree/main/python/SVDA_functions.py (Atmani, F.; Bookhagen, B.; Smith, T. Measuring Vegetation Heights and Their Seasonal Changes in the Western Namibian Savanna Using Spaceborne Lidars. Remote Sens. 2022, 14, 2928. https://doi.org/10.3390/rs14122928)

def getCoordRotFwd(xIn,yIn,R_mat,xRotPt,yRotPt,desiredAngle):
    """ The functions below are used to calculate the along-track distance, the functions are the same used by
    PhoREAL (Photon Research and Engineering Analysis Library) https://github.com/icesat-2UT/PhoREAL"""

    # Get shape of input X,Y data
    xInShape = np.shape(xIn)
    yInShape = np.shape(yIn)

    # If shape of arrays are (N,1), then make them (N,)
    xIn = xIn.ravel()
    yIn = yIn.ravel()

    # Suppress warnings that may come from np.polyfit
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # endif

    # If Rmatrix, xRotPt, and yRotPt are empty, then compute them
    if(len(R_mat)==0 and len(xRotPt)==0 and len(yRotPt)==0):

        # Get current angle of linear fit data
        x1 = xIn[0]
        x2 = xIn[-1]
        y1 = yIn[0]
        y2 = yIn[-1]
        # endif
        deltaX = x2 - x1
        deltaY = y2 - y1
        theta = np.arctan2(deltaY,deltaX)

        # Get angle to rotate through
        phi = np.radians(desiredAngle) - theta

        # Get rotation matrix
        R_mat = np.matrix(np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]]))

        # Get X,Y rotation points
        xRotPt = x1
        yRotPt = y1

    else:

        # Get angle to rotate through
        phi = np.arccos(R_mat[0,0])

    # Translate data to X,Y rotation point
    xTranslated = xIn - xRotPt
    yTranslated = yIn - yRotPt

    # Convert np array to np matrix
    xTranslated_mat = np.matrix(xTranslated)
    yTranslated_mat = np.matrix(yTranslated)

    # Get shape of np X,Y matrices
    (xTranslated_matRows,xTranslated_matCols) = xTranslated_mat.shape
    (yTranslated_matRows,yTranslated_matCols) = yTranslated_mat.shape

    # Make X input a row vector
    if(xTranslated_matRows > 1):
        xTranslated_mat = np.transpose(xTranslated_mat)
    #endif

    # Make Y input a row vector
    if(yTranslated_matRows > 1):
        yTranslated_mat = np.transpose(yTranslated_mat)
    #endif

    # Put X,Y data into separate rows of matrix
    xyTranslated_mat = np.concatenate((xTranslated_mat,yTranslated_mat))

    # Compute matrix multiplication to get rotated frame
    measRot_mat = np.matmul(R_mat,xyTranslated_mat)

    # Pull out X,Y rotated data
    xRot_mat = measRot_mat[0,:]
    yRot_mat = measRot_mat[1,:]

    # Convert X,Y matrices back to np arrays for output
    xRot = np.array(xRot_mat)
    yRot = np.array(yRot_mat)

    # Make X,Y rotated output the same shape as X,Y input
    xRot = np.reshape(xRot,xInShape)
    yRot = np.reshape(yRot,yInShape)

    # Reset warnings
    warnings.resetwarnings()

    # Return outputs
    return xRot, yRot, R_mat, xRotPt, yRotPt, phi

class AtlRotationStruct:

    # Define class with designated fields
    def __init__(self, R_mat, xRotPt, yRotPt, desiredAngle, phi):

        self.R_mat = R_mat
        self.xRotPt = xRotPt
        self.yRotPt = yRotPt
        self.desiredAngle = desiredAngle
        self.phi = phi

def get_atl_alongtrack(df):
    """Function to calculate the along-track distance"""
    easting = np.array(df['Easting'])
    northing = np.array(df['Northing'])

    desiredAngle = 90
    crossTrack, alongTrack, R_mat, xRotPt, yRotPt, phi = getCoordRotFwd(easting, northing, [], [], [], desiredAngle)

    df = pd.concat([df,pd.DataFrame(crossTrack, columns=['crosstrack'])],axis=1)
    df = pd.concat([df,pd.DataFrame(alongTrack, columns=['alongtrack'])],axis=1)

    rotation_data = AtlRotationStruct(R_mat, xRotPt, yRotPt, desiredAngle, phi)

    return df, rotation_data

# the following function is taken and modified from https://github.com/UP-RS-ESP/ICESat-2_SVDA/tree/main/python/SVDA_functions.py (Atmani, F.; Bookhagen, B.; Smith, T. Measuring Vegetation Heights and Their Seasonal Changes in the Western Namibian Savanna Using Spaceborne Lidars. Remote Sens. 2022, 14, 2928. https://doi.org/10.3390/rs14122928)

def ATL03_signal_photons(fname, ATL03_output_path, ROI_fname, EPSG_Code, reprocess=False):
    """
    ATL03_signal_photons(fname, ATL03_output_path, ROI_fname, EPSG_Code)

    Takes a ATL03 H5 file, extracts the following attributes for
    each beam (gt1l, gt1r, gt2l, gt2r, gt3l, gt3r):

    heights/lat_ph
    heights/lon_ph
    heights/h_ph
    % heights/dist_ph_along
    heights/signal_conf_ph

    The function extracts along-track distances,
    converts latitude and longitude to local UTM coordinates,
    filters out land values within the geographic area <ROI_fname>,
    usually a shapefile in EPSG:4326 coordinates, and writes these
    to a compressed HDF file in <ATL03_output_path> starting with 'Signal_'
    and the date and time of the beam.

    """
    # read h5 file in (closed h5 format)
    ATL03 = h5py.File(fname,'r')

    # new list "gtr", filled with every key that starts with "gt"
    gtr = [g for g in ATL03.keys() if g.startswith('gt')]

    # Retrieve datasets
    for b in gtr:
        print('Opening %s: %s' % (os.path.basename(fname), b))
        if reprocess==False and os.path.exists(os.path.join(ATL03_output_path,'ATL03_Land_%s_%s.hdf'%('_'.join(os.path.basename(fname).split('_')[1:2]),b))):
            print('File %s already exists and reprocessing is turned off.\n' % os.path.join(ATL03_output_path,'ATL03_Land_%s_%s.hdf'%('_'.join(os.path.basename(fname).split('_')[1:2]),b)))
            continue
        
        # extract photon attributes

        # Latitude of each received photon
        attribute_lat_ph = b + '/heights/lat_ph'
        lat_ph = np.asarray(ATL03[attribute_lat_ph]).tolist()
        # Longitude of each received photon
        attribute_lon_ph = b + '/heights/lon_ph'
        lon_ph = np.asarray(ATL03[attribute_lon_ph]).tolist()
        # Height of each received photon, relative to the WGS­84 ellipsoid including the geophysical corrections
        attribute_h_ph = b + '/heights/h_ph'
        h_ph = np.asarray(ATL03[attribute_h_ph]).tolist()
        
        # Confidence level associated with each photon event selected as signal. 0=noise. 1=added to allow for buffer but algorithm
        # classifies as background; 2=low; 3=med; 4=high). 
        attribute_signal_conf_ph = b + '/heights/signal_conf_ph'
        signal_conf_ph = np.asarray(ATL03[attribute_signal_conf_ph]).tolist()
        
        # set up dataframe
        ATL03_df = pd.DataFrame({'Latitude': lat_ph,            'Longitude': lon_ph, 
                                 #'Along-track_Distance': dist_ph_along,
                                 'Photon_Height': h_ph,         'Signal_Confidence': signal_conf_ph#, 'ATL08_classification': classif
                                 })
        
        # all df column dtypes: float64

        # delete variables not in use anymore
        del lat_ph, lon_ph, h_ph, signal_conf_ph
        del attribute_lat_ph, attribute_lon_ph, attribute_h_ph, attribute_signal_conf_ph

        # signal_conf_ph is a 5 x N array,
        # first row is land surface-type-specific confidence levels associated with each photon event are: 
        # 0 (noise); 1 (added as buffer, but classifed by the algorithm as background); 
        # 2 (low confidence signal); 3 (medium confidence signal); and 4 (high confidence signal)
        ATL03_df.loc[:, 'Land'] = ATL03_df.Signal_Confidence.map(lambda x: x[0])
        ATL03_df = ATL03_df.drop(columns=['Signal_Confidence'])

        # Transform coordinates into UTM
        x, y = np.array(ATL03_df['Longitude']), np.array(ATL03_df['Latitude'])
        transformer = Transformer.from_crs('epsg:4326', EPSG_Code, always_xy=True)
        xx, yy = transformer.transform(x, y)

        # Save the UTM coordinates into the dataframe
        ATL03_df['Easting'] = xx
        ATL03_df['Northing'] = yy

        ATL03_df, rotation_data = get_atl_alongtrack(ATL03_df)

        # turn shp into geopandas geo-df and retrieve bounding box
        ROI = gp.GeoDataFrame.from_file(ROI_fname, crs='EPSG:4326')
        minLon, minLat, maxLon, maxLat = ROI.envelope[0].bounds

        # Subset the dataframe into the study area bounds
        ATL03_df = ATL03_df.where(ATL03_df['Latitude'] > minLat)
        ATL03_df = ATL03_df.where(ATL03_df['Latitude'] < maxLat)
        ATL03_df = ATL03_df.where(ATL03_df['Longitude'] > minLon)
        ATL03_df = ATL03_df.where(ATL03_df['Longitude'] < maxLon)
        ATL03_df = ATL03_df.dropna()

        # calculate alongtrack distance that starts with 0 at the first photon
        
        if not ATL03_df.empty:
            # sort first (so there are no negative values in alongtrack_base)
            ATL03_df.sort_values(by=['alongtrack'], inplace=True)
            ATL03_df['alongtrack_base'] = ATL03_df['alongtrack'] - ATL03_df['alongtrack'].iloc[0] * np.ones(len(ATL03_df))

        # if output dir doesnt exist, create it
        if not os.path.exists(ATL03_output_path):
            os.mkdir(ATL03_output_path)

        # write ATL03 df to hdf
        ATL03_df.to_hdf(os.path.join(ATL03_output_path,'ATL03_Signal_%s_%s.hdf'%('_'.join(os.path.basename(fname).split('_')[1:2]),b)),
                        key='ATL03_%s_%s'%('_'.join(os.path.basename(fname).split('_')[1::])[:-4],b), complevel=7)
        print('saved to %s'%os.path.join(ATL03_output_path,'ATL03_Signal_%s_%s.hdf'%('_'.join(os.path.basename(fname).split('_')[1:2]),b)))
        print()
    ATL03.close()

