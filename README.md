# Agriculture-Market-Prediction


Introduction
The goal of this project is to predict the market rate of agricultural products using machine learning models. The dataset includes features such as district, year, month, season, rainfall, temperature, crop type, and successful crops. Various regression models are trained and evaluated to find the best-performing model.

Installation
To set up the environment and install the necessary libraries, run the following pip commands:

bash
Copy
pip install numpy pandas matplotlib seaborn joblib scikit-learn xgboost
These libraries are essential for data manipulation, visualization, and model training.

Usage
Clone the repository:

bash
Copy
git clone https://github.com/SK-111961/agriculture-market-prediction.git
cd agriculture-market-prediction
Prepare the dataset:

Ensure that the dataset is in the correct format and placed in the specified path. The dataset should be a CSV file with columns representing the features and target variable.

Run the model training script:

Execute the Jupyter notebook models.ipynb to train the models and evaluate their performance.

bash
Copy
jupyter notebook models.ipynb
Dataset
The dataset used for training the models contains the following columns:

dst: District

yr: Year

mnth: Month

ssn: Season

rf: Rainfall (in mm)

temp: Temperature (in Celsius)

crp: Crops

scrp: Successful Crops

mr: Market Rate (per kg in Rupees)

The dataset is preprocessed to encode categorical variables and split into training and testing sets.

Model Training
The following machine learning models are trained and evaluated:

Random Forest Regressor

Gradient Boosting Regressor

Extra Trees Regressor

XGBoost Regressor

The models are trained using cross-validation, and their performance is evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score, and Explained Variance Score.

Evaluation
The performance of the models is visualized using box plots to compare the accuracy of different models. The best-performing model is saved using joblib for future use.

Dependencies
The project relies on the following Python libraries:

numpy: For numerical operations.

pandas: For data manipulation and analysis.

matplotlib: For data visualization.

seaborn: For statistical data visualization.

joblib: For saving and loading models.

scikit-learn: For machine learning model training and evaluation.

xgboost: For implementing the XGBoost algorithm.
