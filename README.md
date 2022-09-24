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
* Reproducing the baseline from [[1]](#1) using Densenet121, achiving an mean AUC score of 0.90. 
* Analyzing patient demographics and measuring performance gaps among protected groups.
* Training a network for predicting race from X-ray images, achiving an average AUC of over 0.90.
* Using transfer learning on different parts of the network to learn race instead of chest X-ray pathologies, examining how race information propogates through the network.
See illustration below.


<p align="center">
  <img src="https://github.com/SharonPeled/racial-effect-in-radiographs-interpretation/blob/master/Docs/densenet121.png" alt="Sublime's custom image"/>
</p>

Figure 1: DenseNet121 architecture. It consists of 4 denseblocks, whereas each block composed of
multiple feature maps. In oppose to ResNets, DenseNets do not sum the output feature maps, but
concatenate them. Each concatenation follows by a pooling layer to lower on computations. If a
denseblock is frozen, the gradients won’t propagate through its layers, leaving it unchanged. Following
the four denseblocks, a fully-connected layer named ‘classifier’ is applied.

## Data 
<b> CheXpert. </b> [(Available Here)](https://stanfordmlgroup.github.io/competitions/chexpert/)
A large-scale chest radiograph dataset from Stanford Hospital that contains 224,316 frontal and
lateral chest radiographs of 65,240 patients [[2]](#2). The dataset includes a validation set which contains
an addition 200 studies that has been verified by a board of three certificate radiologists. 

<b> MIMIC-CXR-JPG. </b> [(Available Here)](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
A chest radiograph dataset sourced from the Beth Israel Deaconess Medical Center
between 2011 – 2016 [[3]](#3). The dataset consists of 371,920 chest X-rays associated with 227,943 imaging
studies from 65,079 patients.

## Other Material
See report and poster in Docs folder.

## References
<a id="1">[1]</a> Judy Gichoya, Imon Banerjee, Ananth Bhimireddy, John Burns, Leo Celi, Li-Ching Chen, Ramon
Correa, Natalie Dullerud, Marzyeh Ghassemi, Shih-Cheng Huang, Po-Chih Kuo, Matthew Lungren, Lyle Palmer, Brandon Price, Saptarshi Purkayastha, Ayis Pyrros, Lauren Oakden-Rayner, Chima Okechukwu, Laleh Seyyed-Kalantari, and Haoran Zhang. Ai recognition of patient race in
medical imaging: a modelling study. The Lancet Digital Health, 4, 05 2022.

<a id="2">[2]</a> Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik
Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David Mong, Safwan
Halabi, Jesse Sandberg, Ricky Jones, David Larson, Curtis Langlotz, Bhavik Patel, Matthew
Lungren, and Andrew Ng. Chexpert: A large chest radiograph dataset with uncertainty labels
and expert comparison. Proceedings of the AAAI Conference on Artificial Intelligence, 33:590–597,
07 2019.

<a id="3">[3]</a> Alistair Johnson, Tom Pollard, Seth Berkowitz, Nathaniel Greenbaum, Matthew Lungren, Chihying Deng, Roger Mark, and Steven Horng. MIMIC-CXR: A large publicly available database of
labeled chest radiographs, 01 2019.
