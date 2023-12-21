# This file contain all informations about the project:

# Description of the github structure:

- data : folder containing csv of jets events from 2011 to 2016, csv of non jet events from HEK database and the whole downloaded dataset of image sequencies.

- data collection : folder containing a jupyter notebook that shows how the dataset was created and a python file that contain helper functions for this task.

- model training : folder containing a jupyter notebook that shows how the model has been implemented and trained. Also a python file with helper functions for this task.

- model evaluation : folder that contains:
  - model_analysis.ipynb : jupyter notebook that shows the result of the model
  - helper_analysis.py : python file that contain helper functions for this task
  - results_cv_final.json : json file that contain the results of the cross validation (18 models)
  - 2 Best resulting models : Trained_RCNN.pth and Trained_RCNN_2.pth
  - f1_vs_threshold.eps : result of the plot for determine threshold parameter
  - figures : folder that contains every missclassified events as gif

- try_model_with_your_data.ipynb : jupyter notebook for the ESA that will try new datas with our model to benefit from automatic classification and so, detection.

- animation.gif : exemple of the event 29 containing a jet and used in our paper.


# Libraries used in the project:
The following is a summary of the libraries and their specific modules used in this project, organized by their functional category:

- **Core Libraries**
  - NumPy
  - Pandas

- **Solar Physics**
  - Sunpy
  - Astropy

- **Operating System Interaction**
  - OS
  - System

- **Machine Learning Frameworks**
  - PyTorch
  - Torchvision

- **Cross-Validation and Metrics**
  - Scikit-Learn
  - SciPy

- **Visualization and Display**
  - Matplotlib
  - IPython Display Utilities
  - Seaborn

- **Miscellaneous**
  - Random


# Run the code
Take care to not run cells in red (These are explicitly noted in the notebooks) because the computationnal requirement is huge.

## Steps : 
- Make sure to have all used libraries installed.
- Start with data collection 
- Then with model training 
- Finally with the model evaluation. 

# Timeline of the Project
- **6-12 November**: familiarization with the project and exploration of solar physics Python libraries
- **13-30 November**: development of the algorithm to download the data and save it in the correct format
- **1-17 December**: design of machine learning architecture and initiation of report writing.
- **18-21 December**: completion of results analysis and finalization of the report.
