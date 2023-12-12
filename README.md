# Thesis_Ground_Classification_ICESat-2

This repository contains the Python code of my thesis "Classiﬁcation of ICESat-2 ATL03 Point Clouds in Alpine Regions" in the form of Jupyter notebooks.

They contain:
- 1_processing_h5: code for extracting a .h5 file for each beam from the .hdf file for the area of interest
- 2_DEM_conversion: converting ICESat-2 to the same height CRS as the reference data and interpolating the DEM
- 3_1: performing ground classification with the extended percentile filtering (EPF)
- 3_2_*: performing ground classification with U-Net, split up in the steps of retrieving ground truth (1), making feature input and label images (2), training U-Net (3) and comparing the prediction to the ground truth (4)
- 4: comparison of EPF and U-Net

The following modules are needed to execute the code:

```
python=3.10
ipython numpy python matplotlib h5py pandas scipy pyproj pip fiona shapely jupyter ipywidgets pykdtree gdal tqdm scikit-learn weightedstats geopandas cartopy plotly rasterio seaborn pytables laspy requests tensorflow
```
