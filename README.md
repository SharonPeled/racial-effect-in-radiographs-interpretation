# The racial effect in radiographs interpretation
Neural networks often learn to make predictions that overly rely on spurious
correlations existing in the dataset, which causes the model to be biased. This kind of bias is
often difficult to identify, due to the lake of explainability of such classifiers. 

Our work aims to understand the impact race has on X-ray medical imaging in
deep learning models. To this extent, we’re utilizing two popular large-scale chest X-ray datasets:
CheXpert and MIMIC-CXR. <br>
We analyze how race affects deep learning models’ interpretation of X-ray medical images, and validate our conclusions
using OOD dataset.

## Repo Outline:
* Reproducing the baseline from [1] using Densenet121, achiving an mean AUC score of 0.90. 
* Analyzing patient demographics and measuring performance gaps among protected groups.
* Training a network for predicting race from X-ray images, achiving an average AUC of over 0.90.
* Using transfer learning on different parts of the network to learn race instead of chest X-ray pathologies, examining how race information propogates through the network.

## Data 
<b> CheXpert. </b> 
A chest radiograph dataset from Stanford Hospital that contains 224,316 frontal and
lateral chest radiographs of 65,240 patients [2]. The dataset includes a validation set which contains
an addition 200 studies that has been verified by a board of three certificate radiologists.[Available Here](https://stanfordmlgroup.github.io/competitions/chexpert/)

<b> MIMIC-CXR-JPG. </b> 
A chest radiograph dataset sourced from the Beth Israel Deaconess Medical Center
between 2011 – 2016 [3]. The dataset consists of 371,920 chest X-rays associated with 227,943 imaging
studies from 65,079 patients.[Available Here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)




See poster and report for more information.
