# CytoRings
Repository for segmenting and analyzing cytokinetic ring closure in C. Elegans flourescence microscopy data. 
Requires Python 2 and the requirements listed in requirements.txt

## External requirements:
```
ffmpeg
hdf5
openmpi (if running some of the HPC routines)
```

## Environment setup:
```
# install Anaconda if nescessary
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
sh Anaconda3-2020.11-Linux-x86_64.sh

conda update -n base -c defaults conda
conda create --name py39 python==3.9
conda activate py39

pip install -r requirements.txt
```

## Running segmentation via Optimal Net Surface algorithm

```
python2 src/segmentation/extract_optnet.py --verbose --idir some_folder_of_tiffs --ofile output_file.hdf5
```

The folder of TIFFs must have files that have a unique 3-digit identifier in the file name (example: myosin_435.tiff).  This results in a 72-sided polygon for each frame in each TIFF.  Those polygons can be read into NumPy arrays like:

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

If you specify:

```
python src/segmentation/extract_optnet.py --verbose --idir some_folder_of_tiffs --ofile output_file.hdf5 --plot --movie_dir movies/
```

Then you will have a directory of visualization MP4s that show the original frames with the polygons plotted. This or some other form of visualization is highly recommended as there's no guaruntee segmentation will work well.  Requires FFMPEG to be installed and in the system path.

Or you can convert the database to a folder of CSVs:

```
python src/segmentation/convert_to_csv.py --ifile output_file.hdf5 --odir output_csvs/
```

## Step 2: Smooth paramaterization via Elliptical Fourier series and estimation of speed

Add speed and smooth paramaterizations with:
```
python src/segmentation/parameterize_shapes.py --ifile output_file.hdf5 --odir viz_dir
```

Then "output_file.hdf5" will have the following keys added:
```
ifile.create_dataset('{0}/{1}/speed'.format(case, rep), data = speed)
ifile.create_dataset('{0}/{1}/speed_phi'.format(case, rep), data = speed_phi)
ifile.create_dataset('{0}/{1}/xy_smooth'.format(case, rep), data = pc.xy)
ifile.create_dataset('{0}/{1}/xy_phi'.format(case, rep), data = pc.xy_phi)
```

Where speed and xy are the smooth versions of the xy data and the estimated l2-norm of the xy derivatives respectively.  The phi keys point to the data with the the phi correction (recommended).

The script will also write visualizations in the form of MP4s in "viz_dir".

The method is outlined in detail the included docs/method.pdf in docs.


