import src.tract_feat as tract_feat
import src.nn_model as nn_model

import whitematteranalysis as wma
import numpy as np

import argparse
import os
import h5py

import keras
from keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

import os

CPU = True
if CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf

    num_cores = 4

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, 
                            allow_soft_placement=True,
                            device_count = {'CPU' : 4})

    session = tf.Session(config=config)

    tf.keras.backend.set_session(session)

    keras.backend.set_session(session)
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
    description="Testing using a CNN model.",
    epilog="Written by Fan Zhang, fzhang@bwh.harvard.edu")

parser.add_argument(
    'inputModel',
    help='Pretrained network model as an h5 file.')
parser.add_argument(
    'inputFeat',
    help='Input tract feature data as an h5 file.')
parser.add_argument(
    'outputDir',
    help='The output directory should be a new empty directory. It will be created if needed.')

parser.add_argument(
    '-modelLabelName',type=str,
    help='Label name in the model as an h5 file.')

parser.add_argument(
    '-inputLabel',type=str,
    help='Input ground truth label as an h5 file.')

parser.add_argument(
    '-outPrefix',type=str,
    help='A prefix string of all output files.')

parser.add_argument(
    '-tractVTKfile',type=str,
    help='Tractography data as a vtkPolyData file. If given, prediction will output tracts')

args = parser.parse_args()

script_name = '<test>'

if not os.path.exists(args.inputModel):
    print(script_name, "Error: Input network model ", args.inputModel, "does not exist.")
    exit()

if not os.path.exists(args.modelLabelName):
    print(script_name, "Error: Input model label name", args.modelLabelName, "does not exist.")
    exit()

if not os.path.exists(args.inputFeat):
    print(script_name, "Error: Input feature ", args.inputFeat, "does not exist.")
    exit()

if args.inputLabel is None:
    print(script_name, "No input label is provided. Will perform prediction only.")
elif not os.path.exists(args.inputLabel):
    print(script_name, "Error: Input label ", args.inputLabel, "does not exist.")
    exit()

if not os.path.exists(args.outputDir):
    print(script_name, "Output directory", args.outputDir, "does not exist, creating it.")
    os.makedirs(args.outputDir)


''' Load data '''

# Load model parameters
print(script_name, 'Load parameters when training the model.')
params = np.load(args.inputModel.replace('_model.h5', '_params.npy'), allow_pickle=True).item(0)

# Load label names in the model
print(script_name, 'Load tracts names along with the model.')
with h5py.File(args.modelLabelName, "r") as f:
    y_names_in_model = f['y_names'].value

# Load test data feature
with h5py.File(args.inputFeat, "r") as f:
    print(script_name, 'Load input feature.')
    x_test = f['feat'].value

# Generate ground truth labels for evaluation
if args.inputLabel is not None:
    
    print(script_name, 'Load input label.')
    with h5py.File(args.inputLabel, "r") as f:
        y_test = f['label_array'].value.astype(int)
        # y_value = f['label_values'].value
        y_names = f['label_names'].value

        # Used for generate ground truth label
        y_test_orig = y_test.copy()
        y_names_orig = y_names.copy()
    
    # Generate final ground truth label
    print(script_name, 'Generate FINAL ground truth label for evaluation.')
    
    print(script_name, ' # Feat Preprocessing - combine subdiviations of some tracts including CBLM, SupT, and Others.')
    
    y_test, y_names, _ = tract_feat.combine_tract_subdiviations_and_merge_outliers(y_test, y_names, verbose=False)

    if params['bilateral_feature']:
        y_test, y_names, _ = tract_feat.bilateralize_feature(y_test, y_names, verbose=False)

    y_test_ground_truth_final = tract_feat.update_y_test_based_on_model_y_names(y_test, y_names, y_names_in_model)

else:
    y_test_ground_truth_final = None


if params['bilateral_feature']:
    print(script_name, 'Make a bilateral copy for each fiber.')
    x_test, _ = tract_feat.bilateral_X_data(x_test)


# Perform predition of multiple tracts

print('')
print('===================================')
print('')
print(script_name, 'Start multi-tract prediction.')

print(script_name, 'x_test shape:', x_test.shape)
print(script_name, 'tracts to predict:', y_names_in_model)
#print script_name, 'tracts in the input data', y_names


output_multi_tract_predition_mask_path = os.path.join(args.outputDir, args.outPrefix+'_multi_tract_specific_prediction_mask.h5')
output_multi_tract_predition_report_path = os.path.join(args.outputDir, args.outPrefix+'_multi_tract_prediction_report.h5')
if not os.path.exists(output_multi_tract_predition_mask_path):

    # Load model
    model = load_model(args.inputModel)

    y_prediction, prediction_report, con_matrix = nn_model.predict(model, x_test, y_data=y_test_ground_truth_final, y_name=y_names_in_model, verbose=True)

    if args.inputLabel is not None:
        if prediction_report is not None:
            with h5py.File(output_multi_tract_predition_report_path, "w") as f:
                f.create_dataset('prediction_report',data=prediction_report)
                f.create_dataset('con_matrix',data=con_matrix)

    with h5py.File(output_multi_tract_predition_mask_path, "w") as f:
        f.create_dataset('y_prediction',data=y_prediction)

    del model

else:
    print(script_name, 'Loading prediction result.')
    with h5py.File(output_multi_tract_predition_mask_path, "r") as f:
        y_prediction = f['y_prediction'].value

if args.tractVTKfile is not None:

    print('')
    print('===================================')
    print('')
    print(script_name, 'Output fiber tracts.')

    tract_prediction_mask = y_prediction

    print(script_name, 'Load vtk:', args.tractVTKfile)
    pd_whole_tract = wma.io.read_polydata(args.tractVTKfile)

    print(script_name, ' # labels in mask:', np.unique(tract_prediction_mask))
    print(script_name, ' # y_names:', y_names_in_model)

    number_of_tracts = np.max(tract_prediction_mask) + 1
    pd_t_list = wma.cluster.mask_all_clusters(pd_whole_tract, tract_prediction_mask, number_of_tracts, preserve_point_data=False, preserve_cell_data=False, verbose=False)

    output_tract_folder = os.path.join(args.outputDir, args.outPrefix+'_prediction_tracts_outlier_removed')
    if not os.path.exists(output_tract_folder):
        os.makedirs(output_tract_folder)

    for t_idx in range(len(pd_t_list)):
        pd_t = pd_t_list[t_idx]

        if y_names_in_model is not None:
            fname_t = os.path.join(output_tract_folder, y_names_in_model[t_idx].decode('UTF-8')+'.vtp')
        else:
            fname_t = os.path.join(output_tract_folder, 'tract_'+str(t_idx)+'.vtp')

        print(script_name, 'output', fname_t)
        wma.io.write_polydata(pd_t, fname_t)

    print(script_name, 'Done! Tracts are in:', output_tract_folder)
