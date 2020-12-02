# Adapting Faster and Mask R-CNN to Count Objects

This repository contains Jupyter notebooks created for various tasks - from analyzing the dataset to running experiments - to common code implementing PyTorch datasets and models which are used in the notebooks.

Apart from standard libraries such as numpy, pandas and PyTorch, Detectron2 also needs to be installed. Details can be found [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Some additional details may need to be adjusted on a per machine basis, such as paths to directories.

Below are descriptions for each file.

## Notebooks

All notebooks can be found in the ```notebooks/``` folder. Some notebooks will require the download of the dataset [here](https://drive.google.com/file/d/1u-9SPy_biQIVaETdmGaLIopIG7T3dsrW/view?usp=sharing). Due to its size, the ```GRAM``` notebook could not be uploaded to GitHub but can be found [here](https://drive.google.com/file/d/1SYGTuR9f4cBac1W5w20RFZnw6DeAD6L-/view?usp=sharing).

- **Detectron:** contains preliminary experimentation with Detectron2
- **GRAM**: analysis of the M-30 video sequence from the GRAM-RTM, used for the conversion of the provided bounding box annotations in XML to labels for each frame (count of cars) and manual inspection of correctness of the labels. *Warning: around 500MB in size, may take a while to load.*
- **Data Analysis**: analysis of distribution of labels among train, validation and test sets.
- **RCNN**: preliminary tests on adapting RCNN models to count objects
- **Maps**: used to generate and store feature maps from backbones to be used with models under ```rcnn_models/``` for the comparison notebooks
- **Comparisons**: contains tests for performance (subsection 4.1 in the PDF) and usage of different backbones (subsection 4.2 in PDF)
- **Comparisons2**: contains tests for performance comparison while using Adam optimizer for specialized NN, and correlations (subsection 4.3 in PDF)
- **Comparisons3**: contains tests for comparing performance with  RCNN models (subsection 4.4 in PDF)

## Common code

- ```gram.py```: PyTorch dataset for M-30 video sequence from GRAM-RTM and labels of counts of cars
- ```act_maps.py```: PyTorch dataset used to load activation maps generated in ```Maps``` notebook instead of M-30 frames
- ```rcnn.py```: RCNN models from Detectron2 adapted so that the forward pass outputs the count of cars in an image
- ```specialized_nn.py```: specialized NN code adapted from [BlazeIt](https://github.com/stanford-futuredata/blazeit)
- ```rcnn_models/linear_1_conv_1```: Architecture 1 in PDF
- ```rcnn_models/linear_2_conv_1```: Architecture 2 in PDF
- ```rcnn_models/linear_1_conv_2```: Architecture 3 in PDF
- ```rcnn_models/linear_2_conv_2```: Architecture 4 in PDF
- ```rcnn_models/linear_3_conv_3```: Architecture 5 in PDF