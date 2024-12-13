<h1 align="center">ImmoEliza: Deployment</h1> <br>
<p align="center">
  <a href="https://becode.org/" target="_blank">BeCode</a> learning project.
</p>
<p align="center">AI & Data Science Bootcamp</p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
    <li> <a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributors">Contributors</a></li>
    <li><a href="#timeline">Timeline</a></li>
  </ol>
</details>

<details>
<summary>Project Directory Structure</summary>

```
IMMO_ELIZA_DEVELOPMENT/
├── .streamlit/
│   ├── config.toml
├── model/
│   ├── data/
│   │   ├── codes-ins-nis-postaux-belgique.csv
│   │   ├── fisc2022_C_NL.xls
│   │   ├── immoweb_database.csv
│   │   ├── postal-codes-belgium.csv
│   ├── features_data.csv
│   ├── main.py
│   ├── model.py
│   ├── preprocess.py
│   ├── province_encoder.pkl
│   └── random_forest_regressor_model.pkl
├── predict/
│   └── prediction.py
├── preprocessing/
│   └── cleaning_data.py
├── app.py
├── README.md
└── requirements.txt
```

#### Project Directory Structure

##### **IMMO_ELIZA_DEVELOPMENT/**
The root directory for your project. It contains configuration, Python scripts, and other resources required for the machine learning model and deployment.

---

##### **.streamlit/**
A folder used by Streamlit for deploying the model in a web application. 

- **config.toml**
This file contains the configuration settings for the Streamlit application. It controls the appearance of the app, such as theme, layout, and other customization options.

---

##### **model/**
This directory contains all files related to the machine learning model, including data, Python scripts, and trained model files.

- **data/**
  This folder contains various data files used in the project, including the original dataset, external data sources, and additional resources.
  - **`codes-ins-nis-postaux-belgique.csv`**  
    CSV file containing nis codes for Belgium.
  - **`fisc2022_C_NL.xls`**  
    An Excel file related to fiscal data for Belgium, such as prosperity index per municipality. 
  - **`immoweb_database.csv`**  
    The main dataset collected through web scraping from the website Immoweb, containing property features used for training the model.
  - **`postal-codes-belgium.csv`**  
    A CSV file listing postal codes for Belgian regions and municpalities.

- **features_data.csv**  
  CSV file containing features to be used for mapping during the prediction.

- **main.py**  
  The main entry point for the model, orchestrating the workflow of data preprocessing, model loading, and prediction.

- **model.py**  
  Python script that contains the core machine learning model code, including the definition of the Random Forest Regressor model, training, and evaluation. 

- **preprocess.py**  
  This script handles data preprocessing tasks, such as cleaning, feature engineering, and transformations applied to the dataset before feeding it to the model.

- **province_encoder.pkl**  
  A pickle file containing the encoder used for encoding the 'province' feature. It is loaded when the model needs to process input data to ensure consistency in encoding.

- **random_forest_regressor_model.pkl**  
  A pickle file containing the trained Random Forest regression model. It is loaded during prediction to make real-time predictions on user input.

---

##### **predict/**
This folder contains Python scripts related to making predictions using the trained model.

- **prediction.py**  
  A Python script responsible for making predictions with the trained model. It processes input data, applies the trained model, and returns the predicted price.

---

##### **preprocessing/**
This folder contains scripts for data preprocessing tasks.

- **cleaning_data.py**  
  Python script that cleans the user input data and prepared it for the prediction.

---

##### **app.py**
The main entry point for the Streamlit application. This script builds the user interface for the app and integrates the prediction model.

---

##### **README.md**
A markdown file with project documentation. It includes an overview of the project, setup instructions, and usage guidelines.

---

##### **requirements.txt**
A text file listing all the Python packages and dependencies required to run the project.

</details>

## **About The Project**

**Deployment Stage Description**
In this stage, the focus is on deploying the machine learning model that predicts real estate prices in Belgium using a Random Forest Regression Model. The model was trained using a dataset of 37,021 entries with 20 different features, which was initially collected through a [scrapping challenge](https://github.com/MaximSchuermans/immo-eliza/blob/main/data/cleaned_data.csv) from the Belgian real estate website [Immoweb](https://www.immoweb.be/). The dataset was expanded with additional data from a second round of scraping.

The model is being integrated into a web application built with Streamlit, an open-source Python library for creating interactive web apps.The deployment allows users to input property characteristics and locations to get price predictions. 

**Summary of the database after preprocess**
- Shape: 21887, 5
- Target: 'price'
- Features: 'province', 'living_area', 'prosperity_index', 'extra_investment'. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Model
 > **Random Forest Regressor parameters** 
 n_estimators=100
 min_samples_split=5
 min_samples_leaf=17
 max_leaf_nodes=100
 max_depth=80

- **Data Split**: The data was split into training and testing sets using an 80/20 ratio (80% for training and 20% for testing).

- **SQRT Transformation on Target ('price')**: A square root transformation was applied to the target variable ('price') to reduce right skewness in the data, which compresses higher values and spreads out lower values. After making predictions, both the predicted and actual values were transformed back to their original scale by squaring them before evaluating the model's performance.

**Metrics:**
- MAE(train): 66790.418
- MAE(test): 69297.892
- RMSE(train): 89373.949
- RMSE(test): 93937.044
- R²(train): 0.596
- R²(test): 0.538
- MAPE (train): 0.230
- MAPE (test): 0.260
- sMAPE (train): 21.025
- sMAPE (test): 21.712

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## **Installation**

1. Clone this repository
2. Install the required libraries by running 
```shell
pip freeze > requirements.txt
```

## **Usage**
- The model directory contains all necessary data files and scripts for cleaning, preprocessing the database, and training and evaluating the model. To execute this, run:
  ```bash
  python main.py
- The app.py file utilizes the trained model and additional resources to create a Streamlit web application that allows users to input property details and receive predicted property prices.  

## **Contributor**
Project completed by [Jess Rojas-Alvarado](https://github.com/jessrojasal)

## **Project Timeline**

1. **Data Scraping**: 
  **12 Nov 2024 - 15 Nov 2024**  
   - Collected real estate data through web scraping from [Immoweb](https://www.immoweb.be/).  
   - Data cleaned and prepared for analysis.

2. **Data Analysis**:  
   **18 Nov 2024 - 22 Nov 2024**  
   - Analyzed the scraped data to identify trends and relationships.  
   - Performed exploratory data analysis (EDA) to understand feature importance and data distribution.

3. **Model Development**:  
   **2 Dec 2024 - 9 Dec 2024**  
   - Developed and trained the Random Forest Regression model for price prediction.  
   - Evaluated the model's performance using various regression metrics.  

4. **Deployment**:  
   **9 Dec 2024 - 13 Dec 2024**  
   - Integrated the machine learning model into a web application using Streamlit.  
   - Deployed the application to provide real-time price predictions and insights for users.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
