# prerequisite

# 1. Install WMA package (https://github.com/SlicerDMRI/whitematteranalysis).
# 2. Istall 3D Slicer (https://www.slicer.org) and SlicerDMRI (http://dmri.slicer.org). 
# 2. Download `SegModels.zip` (https://github.com/zhangfanmark/DeepWMA/releases) to the current folder, and `tar -xzvf SegModels.zip`
# 3. Download `TestData.zip` (https://github.com/zhangfanmark/DeepWMA/releases) to the current folder, and `tar -xzvf TestData.zip`

BRAINSFitCLI=/Applications/Slicer.app/Contents/lib/Slicer-4.11/cli-modules/BRAINSFit
Slicer=/Applications/Slicer.app/Contents/MacOS/Slicer

atlas_T2=./SegModels/100HCP-population-mean-T2.nii.gz
CNN_model_folder=./SegModels/CNN/

# input data
subject_ID=101410

input_folder=./TestData/${subject_ID}/
output_folder=./TestData/${subject_ID}/DeepWMAOutput
mkdir $output_folder 

subject_b0=${input_folder}/${subject_ID}-dwi_meanb0.nrrd
subject_tract=${input_folder}/${subject_ID}_ukf_l40_f10k.vtp # example whole brain tractography with fiber length over 40 mm. `wm_preprocess_all.py` can be used to remove short fibers for your own data.

# Volume registration
$BRAINSFitCLI --fixedVolume $atlas_T2 --movingVolume $subject_b0 --linearTransform $output_folder/b0_to_atlasT2.tfm --useRigid --useAffine
wm_harden_transform.py ${input_folder} $output_folder $Slicer -t $output_folder/b0_to_atlasT2.tfm -j 1

# FiberMap computation
python ./dlt_extract_tract_feat.py ${output_folder}/${subject_ID}_ukf_l40_f10k.vtp $output_folder -outPrefix ${subject_ID} -feature RAS-3D -numPoints 15

# DeepWMA segmentation
python ./dlt_test.py ${CNN_model_folder}/cnn_model.h5 -modelLabelName ${CNN_model_folder}/cnn_label_names.h5 $output_folder/${subject_ID}_featMatrix.h5 $output_folder -outPrefix ${subject_ID} -tractVTKfile ${subject_tract}

# Clean temp files
rm -r $output_folder/${subject_ID}_featMatrix.h5 ${output_folder}/${subject_ID}_ukf_l40_f10k.vtp