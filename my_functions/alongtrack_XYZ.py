import os, h5py, glob, sys, warnings, tqdm
import pandas as pd
import numpy as np
import geopandas as gp
from pyproj import Transformer
from pyproj import proj
from operator import attrgetter

# the following functions are taken and modified from https://github.com/UP-RS-ESP/ICESat-2_SVDA/tree/main/python/SVDA_functions.py (Atmani, F.; Bookhagen, B.; Smith, T. Measuring Vegetation Heights and Their Seasonal Changes in the Western Namibian Savanna Using Spaceborne Lidars. Remote Sens. 2022, 14, 2928. https://doi.org/10.3390/rs14122928)

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

def get_atl_alongtrack_XYZ(df):
    """Function to calculate the along-track distance for data in ESPG:3857 CRS"""
    easting = np.array(df['X'])
    northing = np.array(df['Y'])

    desiredAngle = 90
    crossTrack, alongTrack, R_mat, xRotPt, yRotPt, phi = getCoordRotFwd(easting, northing, [], [], [], desiredAngle)

    df = pd.concat([df,pd.DataFrame(crossTrack, columns=['crosstrack'])],axis=1)
    df = pd.concat([df,pd.DataFrame(alongTrack, columns=['alongtrack_XYZ'])],axis=1)

    rotation_data = AtlRotationStruct(R_mat, xRotPt, yRotPt, desiredAngle, phi)

    return df, rotation_data