# CytoRings
Repository for segmenting and analyzing cytokinetic ring closure in C. Elegans flourescence microscopy data. 
Requires Python 2 and the requirements listed in requirements.txt 

# Running segmentation via Optimal Net Surface algorithm

```
python2 extract_optnet.py --verbose --idir some_folder_of_tiffs --ofile output_file.hdf5
```

The folder of TIFFs must have files that have a 3-digit identifier in the file name (example: myosin_435.tiff).  This results in a 72-sided polygon for each frame.  That polygon's can be read into NumPy arrays like:

```
import h5py
import numpy as np

ifile = h5py.File('output_file.hdf5', 'r')
keys = list(ifiles.keys())

polygon = np.array(ifile[keys[0]]['xy'])
print(polygon.shape)

# 3D array (frames, index of vertex, x and y coordinate (pixel space))
(120, 72, 2)
```

Or you can convert the database to a folder of CSVs:

```
python2 src/segmentation/convert_to_csv.py --ifile output_file.hdf5 --odir output_csvs/
```
