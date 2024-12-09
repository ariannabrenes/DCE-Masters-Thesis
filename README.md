# README for DCE MRI Pytorch Pipeline
This repository contains the codebase for the master's thesis project: "Direct Reconstruction of Tracer Kinetic (TK) Parameter Maps from DCE MRI K-space Data Using PyTorch." The project explores a novel approach to streamline the estimation of TK parameter maps, leveraging automatic differentiation for modeling Dynamic Contrast-Enhanced (DCE) MRI data. The framework is applied to both brain tumor data (based on a study from USC) and breast lesion data (using the Digital Reference Object toolkit). Link to final thesis document: https://drive.google.com/file/d/15g_fV1uU43iG5IAdodhX5UDAFm5wHBK8/view?usp=sharing

# Repository overview 
## Purpose
The pipeline converts raw K-space DCE MRI data into quantitative TK parameter maps, which provide physiological insights into tissue perfusion and capillary permeability. It is implemented in PyTorch, enabling efficient GPU computation and scalability.
## Key Features
Direct Reconstruction: Directly estimates parameter maps without intermediate image reconstruction.
Automatic Differentiation: Uses PyTorch's autograd capabilities for parameter optimization, contrasting with prior analytical differentiation approaches.
Flexible Framework: Validates the pipeline across both fully sampled and undersampled k-space data, utilizing the Patlak Model (PM).

# Key Components
1. Data Preparation:
 - Brain data from USC study in MATLAB format.
  - Breast data from the DRO toolkit, including lesion masks and coil sensitivity maps.
2. Forward Calculation:
  - Converts TK parameter maps to signal intensities using SPGR and Fourier transforms.
  - Includes options for undersampling radial data and mapping it to Cartesian coordinates.
3. Optimization Loop:
  - Minimizes the Mean Squared Error (MSE) between predicted and known k-space data to refine parameter maps.
4. Analysis:
  - Visualizes results using box and whisker plots for parameter distributions.
  - Evaluates the ability to distinguish benign from malignant lesions.

# Data Sources: 
## Brain data
- Based on Guo et al.'s study (2017) using the Patlak Model in MatLab.
- Simulated with undersampling rates up to 100-fold

## Breast Data
- Generated using the Digital Reference Object toolkit
- Includes simulated malignant and bening lesion masks, coil sensitivity maps, AIFS, and TK parameter ranges. 
