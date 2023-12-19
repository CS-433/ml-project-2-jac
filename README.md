# This file contain all informations about the project:

# Description of the github structure:

- data : folder containing csv of jets events from 2011 to 2016 and csv of non jet events from HEK database

- data0/data0_test/data0_val/data1/data1_test/data1_val : folders containing each set of the data used for the deep learning model. 0 mean no jet and 1, a jet event.

- Download_Data.ipynb : Jupyter notebook that contain the steps of creating the dataset **(group both dataset, positive/negative together)**

- functions_AIA.py : python file where every funcions used to download the data are stored.

- RCNN.ipynb : Jupyter notebook where our model is defined, trained and visualization of results. 
**--> Questions ? everything in the same notebook or separate model, training and results ?, I think train/validate/test functions should be on another pyton file**

- result_cv_final.json : Json file containing the results of 18 different model used as cross-validation, for more details see the report.  

- Trained_RCNN.pth : Our trained model with cosine optimizer

- Trained_RCNN_2.pth : Our trained model with exponential optimizer

**We must choose one of the two model for final one**

# Run the code
As we don't need to recreate the dataset and will take too much time (approx 1.5 days), only run the RCNN file. The part where we train the model will also not be runned as it take around 1 hour with gpu. 
## Steps : 
- Make sure to have all used libraries installed : 
        - numpy
        - matplotlib.pyplot
        - torch (including torchvision)
        - sklearn (metrics and special)
        - pandas
        - astropy
        - sunpy
        - seaborn

- Run the RCNN.ipynb file 

