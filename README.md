Cyclone Casualty and Property Loss Prediction Model for India
Overview
This repository contains the code and documentation for a Cyclone Casualty and Property Loss Prediction model developed for India. The model utilizes machine learning techniques and integrates various datasets to predict the impact of cyclones on casualties and property losses.

Datasets
Population Data: Population statistics for all states in India.
GDP Data: GDP data for all states in India.
Historical Cyclone Dataset: A dataset spanning 50 years, including casualty and property loss information, and affected regions of the states due to cyclones.
Model Development
Integrated Population, GDP, and Historical Cyclone datasets to create a comprehensive dataset for model development.
Employed various machine learning models including Linear Regression, Random Forest, XGBoost, and Light GBM.
Addressed scarcity of historical data by generating synthetic dataset based on limited real data, enhancing its resemblance to real data.
Comparative analysis revealed Random Forest model consistently exhibited highest accuracy across both real and synthetic datasets.
Feature Analysis
Conducted thorough feature analysis to identify key relationships with target variables.
Significant positive correlations found between specific features and casualty or property loss predictions.
For casualty prediction, features such as 'wind speed', 'Total Population', 'Males Population', 'Females Population', and 'number of Households' demonstrated highest accuracy.
For property loss prediction, features including 'wind speed', 'GDP value', 'number of Inhabited villages', and 'number of Towns' exhibited strong predictive capabilities.
Graphical User Interface (GUI)
Developed a python-based GUI seamlessly connected to a MongoDB database.
The database stores all necessary information of each state of India, essential for future prediction.
GUI facilitates user interaction, providing a user-friendly interface for data input and output.
Inputs include the state name and corresponding wind speed of the cyclone, predicting total casualty and property loss, and affected areas of the state.
Usage
Clone the repository: git clone https://github.com/your-username/cyclone-prediction.git
Install dependencies: pip install -r requirements.txt
Run the GUI application: python gui.py

