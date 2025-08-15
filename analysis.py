import torch
import cellpose
from cellpose import models, io
from cellpose.io import imread

from skimage.measure import regionprops_table
import skimage.segmentation
import pandas as pd

io.logger_setup()

# ⬇️ pick device (MPS on M4, else CPU)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = models.CellposeModel(model_type='cyto3')

files = ['111023 TA 23-83 laminin-555_0010_txred (1).jpg']

imgs = [imread(f) for f in files]
nimg = len(imgs)

cellpose_output = model.eval(imgs, diameter=None, channels=[1,0])
cellpose_output = cellpose_output[0]

#Get rid of ROIs that are on the border/boundaries
cellpose_output = [skimage.segmentation.clear_border(im) for im in cellpose_output]
cellpose_output = [skimage.segmentation.relabel_sequential(im)[0] for im in cellpose_output]

for im in cellpose_output:
    props = regionprops_table(label_image=im, properties=['feret_diameter_max'])
    data = pd.DataFrame(props)
    print(data)