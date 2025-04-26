# UrbanEye: Smart Surveillance for Cleaner Cityscapes

# [Deep Learning Project] UrbanEye: Smart Surveillance for Cleaner Cityscapes

## Introduction
Learning and computer vision technologies presents a promising avenue for automating detection and classification tasks in urban environments, enabling more efficient and scalable solutions.

This project aims to address these practical needs by developing a robust object detection model tailored for urban scene analysis. The task involves significant challenges such that variations in lighting conditions, image quality inconsistencies, object occlusion, and a diverse range of issues—from potholes to graffiti—demanding a sophisticated system capable of handling real-world complexities. Leveraging the YOLOv11 architecture, we investigate its ability to create a multi-class detection pipeline that accurately identifies and classifies various urban street issues.


## Project Metadata
### Authors
- **Team:** HUSSAIN ALSHABAAN, ABDULLAH M AL-AWLAQI, RAYAN ALSUBHI
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Visual Pollution Prediction Framework Based on a Deep Active Learning Approach Using Public Road Images](https://www.mdpi.com/2227-7390/11/1/186)

### Reference Dataset
- [SDAIA Smartathon](https://drive.google.com/file/d/1ULqYtd9yomeGz53WBhgRdPRFB37ppeDU/view)


## Project Technicalities

### Terminologies
- **YOLOv11:** A CNN-based single-stage object detection model optimized for real-time inference with high accuracy.
- **Synthetic Data Generation:** The creation of artificial images to balance datasets and improve minority class representation.
- **Class-Weighted Loss:** A modified loss function that assigns different weights to each class based on the dataset distribution.
- **Data Augmentation:** Techniques used to artificially expand the dataset by applying transformations such as rotation, scaling, and flipping.
- **Bounding Box:** A rectangular region used to localize and classify objects in an image.
- **IoU (Intersection over Union):** A metric that measures the overlap between the predicted and ground truth bounding boxes.
- **Early Stopping:** A training strategy where model training is halted once performance on the validation set no longer improves.
- **Recall:** A metric measuring the ability of a model to find all relevant instances in the dataset.
- **Precision:** A metric measuring how many of the model’s positive predictions were actually correct.
- **Dynamic Augmentation:** Adaptive augmentation strategies that vary depending on the dataset's class distribution and complexity.

### Problem Statements
- **Problem 1:** Severe class imbalance in visual pollution datasets leads to biased model training and poor detection of minority classes.
- **Problem 2:** Traditional augmentation and training strategies fail to generalize across varying environmental conditions (e.g., lighting, weather).
- **Problem 3:** Conventional loss functions treat all classes equally, causing underrepresentation of rare or critical pollution types during optimization.

### Loopholes or Research Areas
- **Data Diversity:** Limited diversity in public road images impacts the model’s ability to generalize to unseen environments.
- **Class Sensitivity:** Inadequate attention to minority class detection reduces model reliability for comprehensive pollution monitoring.
- **Real-World Robustness:** Models trained on uniform conditions may perform poorly under real-world variations like nighttime or rainy scenes.

### Problem vs. Ideation:
1. **Dynamic Synthetic Data Generation:** Generate synthetic images per class based on dataset statistics to balance minority classes and enrich the dataset.
2. **Customized Class-Weighted Loss Function:** Modify the loss function to dynamically adjust the contribution of each class during training, improving minority class sensitivity.
3. **Enhanced Data Augmentation Pipeline:** Apply a diverse set of augmentation techniques tailored to simulate real-world environmental variances (e.g., brightness changes, occlusion).


### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced YOLOv11-based visual pollution detection framework using PyTorch. The solution includes:

- **Synthetic Data Generation Module:** Dynamically generates additional training samples to balance class distributions.
- **Customized Loss Function:** Integrates a class-weighted Binary Cross-Entropy (BCE) loss to emphasize minority class learning during training.
- **Enhanced Data Augmentation Pipeline:** Applies advanced augmentation techniques, including random brightness shifts, rotations, and weather-based effects.
- **Optimized Training Strategy:** Incorporates early stopping and learning rate scheduling to maximize training efficiency and prevent overfitting.


### Key Components
- **`preprocessing_dataset.ipynb`**: Handles preprocessing of the UrbanEye dataset.
- **`training_validation_testing.ipynb`**: Contains experiment runs and post-training outputs.
- **`ultralytics/cfg`**: Contains yaml based configuration files for YOLOv11 setup and dataset configuration.
- **`DL504`**: Contains generated models and previous training runs.

## Model Workflow
The workflow of the enhanced YOLOv11-based framework is designed to efficiently detect and localize various types of visual pollution from public road images through a structured data-centric and training-centric process:

1. **Input:**
   - **Image Input:** The model receives road or urban environment images as input for visual pollution detection.
   - **Data Augmentation:** During training, input images undergo dynamic augmentation techniques such as random scaling, brightness adjustment, and rotation to improve model robustness.
   - **Synthetic Data Generation:** Additional synthetic images are introduced to balance underrepresented classes based on dataset statistics.

2. **Training Process:**
   - **Feature Extraction and Detection:** The input images are processed through the YOLOv11 architecture, where convolutional layers extract hierarchical features, and detection heads predict object bounding boxes and class probabilities.
   - **Customized Loss Computation:** A class-weighted Binary Cross-Entropy (BCE) loss function is applied, dynamically adjusting the loss contributions according to class frequencies to handle class imbalance.
   - **Model Optimization:** Early stopping and learning rate scheduling are utilized to optimize training efficiency and prevent overfitting, ensuring the model generalizes well across different visual pollution types.

3. **Output:**
   - **Bounding Box Predictions:** The model outputs bounding boxes with associated class labels (e.g., graffiti, potholes, garbage) and confidence scores.
   - **Detection Results:** The final output consists of accurately localized and classified instances of visual pollution, ready for further use in urban monitoring or environmental management systems.


## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/Smart-Surveillance-for-Cleaner-Cityscapes.git
    cd Smart-Surveillance-for-Cleaner-Cityscapes
    ```

2. **Set Up the Environment:**
    It is advised to use VScode and connect to a virtual server as the training process is both CPU and GPU-intensive. Also, install the dependecies (pyproject.toml)
    ``` 
    pip install .
    ```
   
3. **Run Preprocessing pipline:**
    ```bash
    jupyter preprocessing_dataset.ipynb
    ```
4. **Train the Model:**
    ```bash
    jupyter raining_validation_testing.ipynb
    ```

## Acknowledgments
This project is built on top of [YOLOv11](https://github.com/ultralytics/ultralytics), developed by [Ultralytics](https://ultralytics.com/)
- **Open-Source Communities:** Thanks to the contributors of Python, YOLO, PyTorch, and other libraries for their great work.
- **Individuals:** Special thanks to Dr. Behzad for invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to Visa & StcPay for allowing us to rent resources from ``vast.ai`` using our own money.


