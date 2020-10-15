# DeepWMA (Deep White Matter Analysis)

This code implements a deep learning tractography segmentation method (DeepWMA) that allows fast and consistent white matter fiber tract identification, as described in the following paper:

    Fan Zhang, Suheyla Cetin Karayumak, Nico Hoffmann, Yogesh Rathi, Alexandra J. Golby, and Lauren J. O’Donnell. 
    Deep white matter analysis (DeepWMA): fast and consistent tractography segmentation.
    Medical Image Analysis 65 (2020): 101761

Please download the pre-trained CNN models and testing data: 

    https://github.com/zhangfanmark/DeepWMA/releases

Download `SegModels.zip` and uncompress to the current folder.

Download `TestData.zip` and uncompress to the current folder.

# Example

	sh run_DeepWMA.sh

**Please cite the following papers:**

    Fan Zhang, Suheyla Cetin Karayumak, Nico Hoffmann, Yogesh Rathi, Alexandra J. Golby, and Lauren J. O’Donnell. 
    Deep white matter analysis (DeepWMA): fast and consistent tractography segmentation.
    Medical Image Analysis 65 (2020): 101761

    Fam Zhang, Ye Wu, Isaiah Norton, Yogesh Rathi, Nikos Makris and Lauren J. O’Donnell. 
    An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. 
    NeuroImage, 2018 (179): 429-447

    Fan Zhang, Thomas Noh, Parikshit Juvekar, Sarah F Frisken, Laura Rigolo, Isaiah Norton, Tina Kapur, Sonia Pujol, William Wells III, Alex Yarmarkovich, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Ron Kikinis, Hans J Johnson, Carl-Fredrik Westin, Steve Pieper, Alexandra J Golby, Lauren J O'Donnell. 
    SlicerDMRI: Diffusion MRI and Tractography Research Software for Brain Cancer Surgery Planning and Visualization. 
    JCO Clinical Cancer Informatics 4, e299-309, 2020.
    
    Isaiah Norton, Walid Ibn Essayed, Fan Zhang, Sonia Pujol, Alex Yarmarkovich, Alexandra J. Golby, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Steve Pieper, Ron Kikinis, Hans J. Johnson, Carl-Fredrik Westin and Lauren J. O'Donnell. 
    SlicerDMRI: Open Source Diffusion MRI Software for Brain Cancer Research. Cancer Research 77(21), e101-e103, 2017.
