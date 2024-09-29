# Hematology Analyzer

## Project Overview
The *Hematology Analyzer* is a deep learning-based system designed to classify normal and abnormal blood cells from microscopic blood smear images of Cholistani cattle. It can identify specific types of abnormalities, helping diagnose various blood disorders in cattle.

### Key Features:
- *Blood Cell Segmentation: The system uses the **Random Walker method* to isolate individual cells from larger microscopic images.
- *Principal Component Analysis (PCA): After extracting cells, **PCA* is performed on the cell data to reduce dimensionality and visualize feature variance across cells.
- *Blood Cell Classification*: Each cell is classified as either normal or abnormal, with further classification based on the type of abnormality.
- *Interactive Dashboards: **Dash* is used to create visualizations for the PCA of all cells, providing interactive exploration of the relationships between blood cell features.

### Cell Types:
The system is trained to recognize the following cell types:
- *Normal RBCs*: Healthy, biconcave-shaped red blood cells.
- *Schistocytes*: Abnormally shaped cells often associated with cystic disorders.
- *Echinocytes*: "Burr" cells with abnormal surface projections.
- *Teardrop Cells*: Cells with a teardrop shape, often linked to bone marrow issues.

## Workflow Overview

1. *Microscopic Image Input*: 
   - The user inputs a full microscopic blood smear image containing multiple blood cells.
  
2. *Cell Segmentation*:
   - The system applies the *Random Walker* method to segment individual cells from the larger image.

3. *Principal Component Analysis (PCA)*:
   - *PCA* is applied to reduce dimensionality for all segmented cells, enabling visual analysis of feature variance. The PCA results are visualized using *Dash*, which provides an interactive interface.

4. *Classification*:
   - Each segmented cell is passed through a convolutional neural network (CNN) that classifies it as normal or abnormal, with further classification of abnormal cells (Cystocyte, Echinocyte, Teardrop Cell).

5. *Results & Visualization*:
   - The classification results and PCA visualizations are displayed in an interactive *Dash* dashboard for easy analysis.

## Installation

### Installation
   Use the provided requirements.txt file to install all dependencies:
   bash
   pip install -r requirements.txt
   

  *Run the Application*:
   bash
   python hema_main.py
   


## License
This project is licensed under the MIT License.