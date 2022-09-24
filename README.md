# The racial effect in radiographs interpretation
The performance of deep learning models in the field of medical imaging has reached or even
exceeded human-level performance, especially when it comes to diagnosing diseases using chest
X-rays. However, neural networks often learn to make predictions that overly rely on spurious
correlations existing in the dataset, which causes the model to be biased. This kind of bias is
often difficult to identify, due to the lake of explainability of such classifiers. As computer vision
systems are deployed at scale in variety of settings, it becomes increasingly important to be aware
of such drawbacks, especially in the medical domain. Previous studies in medical imaging have
shown disparate abilities of deep learning models to detect a person’s race, yet there is no known
correlation for race on medical imaging that would be obvious to human experts when interpreting
X-ray images. Our work aims to understand the impact race has on X-ray medical imaging in
deep learning models. To this extent, we’re utilizing two popular large-scale chest X-ray datasets:
CheXpert and MIMIC-CXR.
