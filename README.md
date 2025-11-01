# InternPe_AI_ML_Project_3
IPL WINNING TEAM PREDICTION


## Project 3: IPL WINNING TEAM PREDICTION
<br>

**Dataset** : https://www.kaggle.com/datasets/yuvrajdagur/ipl-dataset-season-2008-to-2017
<br>

This project is part of my internship at InternPe, where i implemented various machine learning models to predict the winning team in an IPL based on key features :
<br>

:) mid: Unique match id.
<br>

:) date: Date on which the match was played.
<br>

:) venue: Stadium where match was played.
<br>

:) batting_team: Batting team name.
<br>

:) bowling_team: Bowling team name.
<br>

:) batsman: Batsman who faced that particular ball.
<br>

:) bowler: Bowler who bowled that particular ball.
<br>

:) runs: Runs scored by team till that point of instance.
<br>

:) wickets: Number of Wickets fallen of the team till that point of instance.
<br>

:) overs: Number of Overs bowled till that point of instance.
<br>

:) runs_last_5: Runs scored in previous 5 overs.
<br>

:) wickets_last_5: Number of Wickets that fell in previous 5 overs.
<br>

:) striker: max(runs scored by striker, runs scored by non-striker).
<br>

:) non-striker: min(runs scored by striker, runs scored by non-striker).
<br>

:) total: Total runs scored by batting team at the end of first innings.
<br>

**Project Overview** :
<br>
This project focuses on predicting the winning team in an IPL (Indian Premier League) match using machine learning models. The dataset used contains historical IPL match data from 2008 to 2017, which has been cleaned, preprocessed, and analyzed to train various classification models for prediction.
<br>


**Tools Used** :
<br>
**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**Libraries Used** :
<br>
**pandas (pd)** – For data manipulation and analysis
<br>
**numpy (np)** – For numerical computations
<br>
**sklearn.model_selection** – For splitting the dataset (train_test_split)
<br>
**sklearn.preprocessing** – For encoding categorical data (LabelEncoder) and feature scaling (StandardScaler)
<br>
**sklearn.linear_model** – For implementing linear regression (LinearRegression)
<br>
**sklearn.metrics** – For evaluating model performance (mean_absolute_error, mean_squared_error, r2_score)
<br>

**Implementation Steps** :
<br>
(1) **Data Preprocessing**:
<br>
:) Removed irrelevant columns
<br>
:) Filtered only consistent teams (teams that played across seasons)
<br>
:) Excluded the first 5 overs of every match (to focus on impactful overs)
<br>
:) Handled missing values and performed feature engineering
<br>

(2) **Feature Encoding** :
<br>
:) Label Encoding for categorical variables
<br>
:) One-Hot Encoding & Column Transformation for better model performance
<br>

(3) **Model Implementation**:
<br>
Trained multiple machine learning models for prediction:
<br>
:) Decision Tree Regressor
<br>
:) Linear Regression
<br>
:) Random Forest Regression
<br>
:) Lasso Regression
<br>
:) Support Vector Machine
<br>
:) Neural Networks
<br>

(4) **Evaluation**:
<br>
:) Used metrics like Mean absolute error , Mean squared error , Root mean squared error to evaluate model performance.

**Results**:
<br>

**Decision Tree Regressor**
<br>
:) Train Score : 99.99%
<br>
:) Test Score : 85.83%
<br>
:) Mean Absolute Error (MAE): 3.980615806532037
<br>
:) Mean Squared Error (MSE): 125.66389304412864
<br>
:) Root Mean Squared Error (RMSE): 11.209990769136638
 <br>

**Linear Regression**
<br>
:) Train Score : 65.99%
<br>
:) Test Score : 65.62%
<br>
:) Mean Absolute Error (MAE): 13.112410435179777
<br>
:) Mean Squared Error (MSE): 305.0563257623772
<br>
:) Root Mean Squared Error (RMSE): 17.465861724013998
<br>

**Random Forest Regression**
<br>
:) Train Score : 99.07%
<br>
:) Test Score : 93.57%
<br>
:) Mean Absolute Error (MAE): 4.398791178425996
<br>
:) Mean Squared Error (MSE): 56.99923941594296
<br>
:) Root Mean Squared Error (RMSE): 7.549784064193026
<br>

**Lasso Regression**
<br>
:) Train Score : 64.95%
<br>
:) Test Score : 64.93%
<br>
:) Mean Absolute Error (MAE): 13.093065567932516
<br>
:) Mean Squared Error (MSE): 311.1276056545018
<br>
:) Root Mean Squared Error (RMSE): 17.638809643921604
 <br>

**Support Vector Machine**
<br>
:) Train Score : 57.41%
<br>
:) Test Score : 57.39%
<br>
:) Mean Absolute Error (MAE): 14.635093385154384
<br>
:) Mean Squared Error (MSE): 378.08184639617673
<br>
:) Root Mean Squared Error (RMSE): 19.444326843482568
<br>

**Neural Networks**
<br>
:) Train Score : 85.45%
<br>
:) Test Score : 85.99%
<br>
:) Mean Absolute Error (MAE): 7.978030236608694
<br>
:) Mean Squared Error (MSE): 124.29835286278254
<br>
:) Root Mean Squared Error (RMSE): 11.148917116150004
<br>

**From above results, we can see that Random Forest performed the best, closely followed by Decision Tree and Neural Networks. So we will be choosing Random Forest for the final model**
<br>



