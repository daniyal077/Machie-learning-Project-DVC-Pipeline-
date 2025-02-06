Water Potability Prediction
==============================

A machine learning pipeline for predicting water potability using DVC for data version control and pipeline management.
Project Organization
------------

    ├── Makefile           
    ├── README.md          
    ├── data
    │   ├── raw            
    │   ├── processed      
    │
    ├── models                         
    │   ├── rf_model.pkl      
    |
    ├── notebooks         <- Jupyter notebook          
    │   ├── research.ipynb                              
    │                             │
    ├── results            
    │   └── result_metrics.json        
    │
    ├── requirements.txt  
    │                     
    ├── src                
    │   │
    │   ├── data          
    │   │   └── data_collection.py
    │   │   └── data_preprocessing.py
    │   │
    │   ├── models         
    │   │   │                 predictions
    │   │   ├── model_building.py
    │   │   └── model_evaluation.py
    │   │    │
    └── tox.ini            

