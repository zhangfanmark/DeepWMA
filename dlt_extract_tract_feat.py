import src.tract_feat as tract_feat

import whitematteranalysis as wma
import numpy as np

import argparse
import os
import h5py

#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
    description="Compute FiberMap of input vtk file.",
    epilog="Written by Fan Zhang, fzhang@bwh.harvard.edu")

parser.add_argument(
    'inputVTK',
    help='input tractography data as vtkPolyData file(s).')
parser.add_argument(
    'outputDir',
    help='The output directory should be a new empty directory. It will be created if needed.')

parser.add_argument(
    '-outPrefix',type=str,
    help='A prefix string of all output files.')

# RAS: Right-Anterior-Superior
parser.add_argument(
    '-feature', action="store", type=str,
    help="Name of feature. Currently support: `RAS`")
parser.add_argument(
    # TODO: RNN to get rid for this.
    '-numPoints', action="store", type=int, default=15,
    help='Number of points per fiber to extract feature.')
parser.add_argument(
    # TODO: RNN to get rid for this.
    '-numRepeats', action="store", type=int, default=15,
    help='Number of repiteation times.')

parser.add_argument(
    '-downsampleStep', action="store", type=int,
    help='Downsample the input')
parser.add_argument(
    '-groundTruthLabel', action="store", type=str,
    help='Path to the ground truth label file. Should be provided when downsample is used.')

args = parser.parse_args()

script_name = '<extract_tract_feat>'

if not os.path.exists(args.inputVTK):
    print(script_name, "Error: Input tractography ", args.inputVTK, "does not exist.")
    exit()

if not os.path.exists(args.outputDir):
    print(script_name, "Output directory", args.outputDir, "does not exist, creating it.")
    os.makedirs(args.outputDir)

print(script_name, 'Reading input tractography:', args.inputVTK)
pd_tract = wma.io.read_polydata(args.inputVTK)

print(script_name, 'Computing feauture:', args.feature)

if args.feature == 'RAS':

    feat_RAS = tract_feat.feat_RAS(pd_tract, number_of_points=args.numPoints)

    # Reshape from 3D (num of fibers, num of points, num of features) to 4D (num of fibers, num of points, num of features, 1)
    # The 4D array considers the input has only one channel (depth = 1)
    feat_shape = np.append(feat_RAS.shape, 1)
    feat = np.reshape(feat_RAS, feat_shape)

if args.feature == 'Orientation-3D':

    feat_orient = tract_feat.feat_orientation_3D(pd_tract, number_of_points=args.numPoints, repeat_time=args.numPoints)

    feat = feat_orient

elif args.feature == 'RAS-3D':

    feat_RAS_3D = tract_feat.feat_RAS_3D(pd_tract, number_of_points=args.numPoints, repeat_time=args.numRepeats)

    feat = feat_RAS_3D

elif args.feature == 'RASF':

    feat_RAS_FS = tract_feat.feat_RASF(pd_tract, number_of_points=args.numPoints)

    # Reshape from 3D (num of fibers, num of points, num of features) to 4D (num of fibers, num of points, num of features, 1)
    # The 4D array considers the input has only one channel (depth = 1)
    feat_shape = np.append(feat_RAS_FS.shape, 1)
    feat = np.reshape(feat_RAS_FS, feat_shape)

elif args.feature == 'RASF-3D':

    feat_RAS_FS = tract_feat.feat_RASF_3D(pd_tract, number_of_points=args.numPoints)

    feat = feat_RAS_FS

elif args.feature == 'RAS-1D':

    feat_RAS_1D = tract_feat.feat_1D(pd_tract, number_of_points=args.numPoints)

    feat_shape = np.append(feat_RAS_1D.shape, 1)
    feat_shape = np.append(feat_shape, 1)

    feat = np.reshape(feat_RAS_1D, feat_shape)

elif args.feature == 'RASCurvTors':
    
    feat_curv_tors = tract_feat.feat_RAS_curv_tors(pd_tract, number_of_points=args.numPoints)

    feat_shape = np.append(feat_curv_tors.shape, 1)
    
    feat = np.reshape(feat_curv_tors, feat_shape)

elif args.feature == 'CurvTors':
    
    feat_curv_tors = tract_feat.feat_curv_tors(pd_tract, number_of_points=args.numPoints)

    feat_shape = np.append(feat_curv_tors.shape, 1)
    
    feat = np.reshape(feat_curv_tors, feat_shape)

print(type(feat))

print(script_name, 'Feature matrix shape:', feat.shape)

if args.groundTruthLabel is not None:
    with h5py.File(args.groundTruthLabel, "r") as f:
        label_array = f['label_array'].value.astype(int)
        label_values = f['label_values'].value
        label_names = f['label_names'].value
        # print script_name, 'Input label_names:'
        # print label_names
else:
    label_array = None
    label_values = None
    label_names = None


## downsampling
if args.downsampleStep is not None:
    print(script_name, 'Downsampling the feature matrix with step size:', args.downsampleStep)

    feat, label_array = tract_feat.downsample(args.downsampleStep, feat, label_array)

    print(script_name, 'Feature matrix shape (downsampled):', feat.shape)
    print(script_name, 'Label array shape (downsampled):', label_array.shape if label_array is not None else label_array)


## Save feat
with h5py.File(os.path.join(args.outputDir, args.outPrefix+'_featMatrix.h5'), "w") as f:
    f.create_dataset('feat', data=feat)

    print(script_name, 'Feature matrix shape:', feat.shape)


## Save label
if args.groundTruthLabel is not None:
    with h5py.File(os.path.join(args.outputDir, args.outPrefix+'_label.h5'), "w") as f:
        f.create_dataset('label_array', data=label_array)
        f.create_dataset('label_values', data=label_values)
        f.create_dataset('label_names', data=label_names)

    print(script_name, 'Ground truth shape:', label_array.shape)
    print(script_name, 'Ground truth label names', label_names)

print(script_name, 'Done! Find results in:', args.outputDir)

