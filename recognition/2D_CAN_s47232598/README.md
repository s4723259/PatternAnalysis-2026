# 2D-CAN MODEL ON BRAIN TISSUE SEGMENTATION

## Model Description 
meow meow meow.

## Feasibility Review  
### 1. User Need, Scope & Acceptance Criteria
The project chosen proposes an annotation assistant that generates brain tissue segmentation contours. The intended user is an imaging specialist, who reviews and adjusts the contours provided by the model. The purpose of implementing this model is to cut down on clinical contouring time. The central engineering dilemma is that high global dice scores can conceal local boundary errors and disconnected anatomical regions. To investigate this, acceptability criteria are a target dice score of >= 0.90 across all tissue labels on the held-out test set, reporting per class dice and intersection over union, inspecting difficult boundaries and measuring peak GPU memory usage and per case inference time. These were chosen as it provides key insight into the model’s ability to produce valid segmentations on edge cases, with respect to the metric of time and computing resources which it is intending to reduce from current system expectations.  
  
### 2. Model Choice & Course Concepts
The proposed dataset configuration is the OASIS Brain 2D dataset. The model architecture chosen is the 2D CAN architecture from the easy difficulty band. This was chosen as the CAN model is specifically designed for dense prediction<sup>[1]</sup>, which is quite applicable to the medical segmentation problem space. It will be compared to a baseline U-Net for analysis of its performance. This is supported by the class content in contact seven, on segmentation in convolutional neural networks. It specifically references the applicability in medical imaging and provides a valuable resource for architecture and features of the model to consider for this problem space.  
  
### 3. Preliminary Feasibility evidence  
An initial dataset audit will verify image-mask alignment, label consistency, file integrity and the number of usable subjects and slices. A small pilot experiment will test data loading, model output dimensions, loss calculation and training stability. GPU memory usage and iteration time will be recorded to estimate training costs. These checks will establish whether the dataset, architecture and hardware are compatible before committing to full training.  
  
### 4. Risks, budget & fallback  
Key technical risks for the brief are fine gyral folds in the data, partial volume effects at boundaries and RF coil intensity inhomogeneity. Managing these challenges in the scope of the computing budget will be a high priority for development. The computing budget available is either the UQ EAIT Rangpur GPU cluster, or a personal home computer with one NVIDIA 3060ti GPU. The next planned experiment is smaller scale testing on the CAN model architecture to ensure validity for the project. The fallback is to the 2D improved U-Net architecture. 

## Dependencies Required
meow.py

## Data Investigation
- **Preprocessing Conducted:** Meow meow
- **Training, Validation and Test Splits:** Mrow meow :3  

## Open Research Dilemma:
### Quantitative Benchmarking
meow meow
### Resource Profiling Table
meow :3
### Qualitative Error Autopsies
meow mrow
### Actionable Engineering Recommendation
meow, mrow mrow  
  
## References  
1. F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” in Proc. ICLR, 2016


## Artificial Intelligence Usage Disclosure
An AI model was used in predict.py, specifically lines 398 to 522. Claude AI was used on 6/11/26 with this prompt:
> Hi Claude, I need help using matplotlib for generating figures of relevant results for a machine learning assignment. Can you please review my code and provide code for plotting figures? I'm specifically looking for ones that plot failure cases, slice positions and global dice vs boundary dice

The code was audited, verified and empirically tested by first line by line inspection, then inspection of the produced results against the known figures

An AI model was used in predict.py, specifically lines to 394. Claude AI was used on 6/11/26 with this prompt:  
> Hi Claude, I need some help with creating a python file that analyses some machine learning models. Here is the code I have written for returning the relevant metrics. Can you please create some code that outputs the results of these to the console with print() in a user friendly way?  
  
The code was audited, verified and empirically tested by line by line inspection, then inspection of the produced results against the known figures (from my first attempt at console output).
 
