## Introduction
This project demonstrates how to build a machine learning system capable of predicting the amount of calories burned during exercise. This is a highly relevant machine learning use case, as many individuals are interested in understanding the energy expenditure associated with their physical activities. This model aims to provide insights into how various exercise parameters contribute to calorie burn.

## Problem Statement
The core problem is to predict the number of calories burned during different types of exercise. The human body undergoes several metabolic activities during physical exertion. When a person exercises, the energy from consumed food (e.g., carbohydrates broken down into glucose) is utilized. Glucose is broken down using oxygen, which is supplied by increased blood flow (higher heart rate). A portion of this energy is converted into heat, leading to an increase in body temperature and subsequent sweating to cool down.

Therefore, key parameters like the duration of exercise, average heart rate, and body temperature are indirect indicators of exercise intensity and, consequently, calorie burn. The model will leverage these and other personal attributes (height, weight, age, gender) to predict calories.

## Dataset
The project utilizes two CSV files:
1.  `calories.csv`: Contains `User_ID` and `Calories` burned.
2.  `exercise.csv`: Contains `User_ID`, `Gender`, `Age`, `Height`, `Weight`, `Duration` (minutes), `Heart_Rate` (beats per minute), and `Body_Temp` (Celsius).

These two datasets are merged based on the `User_ID` to create a comprehensive dataset for the model. The combined dataset contains 15,000 data points (rows) and 9 columns (features + target).

## Workflow
The project follows a standard machine learning workflow:

1.  **Data Collection:**
    * Data was collected from Kaggle, specifically the "Calories Burnt Prediction" dataset.
    * Two CSV files, `calories.csv` and `exercise.csv`, were downloaded and loaded into pandas DataFrames.

2.  **Data Pre-processing:**
    * The `calories.csv` and `exercise.csv` DataFrames were concatenated (joined column-wise) based on `User_ID` to form a single `calories_data` DataFrame.
    * The dataset was inspected for missing values using `isnull().sum()`. Fortunately, no missing values were found, eliminating the need for imputation.
    * The 'Gender' column, originally in text format ('male', 'female'), was converted into numerical representation (0 for male, 1 for female) using one-hot encoding for machine learning compatibility.

3.  **Data Analysis:**
    * **Statistical Measures:** `calories_data.describe()` was used to obtain descriptive statistics (mean, std, min, max, quartiles) for numerical columns, providing insights into data distribution and ranges.
    * **Data Visualization:**
        * A count plot was generated for the 'Gender' column to visualize the distribution of males and females, showing an almost equal distribution.
        * Distribution plots (`distplot`) were created for 'Age', 'Height', and 'Weight' columns to understand their value distributions. For example, 'Age' showed a peak in the 20-30 range, suggesting more data points for younger individuals.
    * **Correlation Heatmap:** A heatmap was constructed to visualize the correlation between all numerical features. Key insights from the heatmap included:
        * 'Height' and 'Weight' are positively correlated.
        * 'Duration', 'Heart_Rate', and 'Body_Temp' are highly positively correlated with 'Calories' burned, indicating that as these parameters increase, calorie burn also increases. This highlights their importance as indirect measures of exercise intensity.

4.  **Separating Features and Target:**
    * The `User_ID` and `Calories` columns were dropped from the main DataFrame to form the feature set (`X`).
    * The `Calories` column was isolated as the target variable (`Y`).

5.  **Splitting Data:**
    * The dataset (`X` and `Y`) was split into training and testing sets using `train_test_split` with a `test_size` of 0.2 (20% for testing) and `random_state=2` for reproducibility. This resulted in 12,000 training data points and 3,000 test data points.

6.  **Model Training:**
    * An XGBoost Regressor model (`XGBRegressor`) was initialized and loaded.
    * The model was trained using the training data (`X_train`, `Y_train`).

7.  **Model Evaluation:**
    * Predictions were made on the unseen test data (`X_test`) using the trained model (`model.predict(X_test)`).
    * The model's performance was evaluated using the Mean Absolute Error (MAE), which measures the average magnitude of the errors in a set of predictions, without considering their direction.
    * A Mean Absolute Error of **1.48** was achieved, which is considered a very good score given the magnitude of the 'Calories' values (typically in hundreds). This indicates a high level of accuracy in the model's predictions.

## Key Learnings/Insights
* The intensity of exercise (implicitly captured by duration, heart rate, and body temperature) is a strong predictor of calories burned.
* Data preprocessing, including handling categorical variables, is crucial for preparing data for machine learning models.
* Visualization techniques like count plots, distribution plots, and heatmaps are effective for understanding data distributions and relationships.

## Model Performance
The XGBoost Regressor model achieved a Mean Absolute Error (MAE) of **1.48**. This indicates that, on average, the model's predictions for calories burned are very close to the actual values.

## How to Run the Project
1.  **Clone the Repository:** (Assumes this will be in a repo)
2.  **Download Datasets:**
    * Download `calories.csv` and `exercise.csv` from the Kaggle dataset: [https://www.kaggle.com/datasets/fmendes/fmendescalories-burnt-prediction](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos)
3.  **Set up Environment:**
    * Use Google Colaboratory (Google Colab) for running the notebook.
    * Alternatively, ensure you have Python installed with the following libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`.
4.  **Upload Data to Colab:**
    * In Google Colab, click on the 'Files' icon on the left sidebar.
    * Click on 'Upload to session storage' (or right-click and 'Upload') and upload both `calories.csv` and `exercise.csv`.
5.  **Run the Notebook:**
    * Open `Calorie_Prediction.ipynb` in Google Colab.
    * Run all the cells sequentially. The notebook provides detailed explanations for each step.
