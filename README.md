<h1 align="center">Algerian Forest Fire Analysis & Prediction</h1>

<p align="center">
  <strong>Exploratory Data Analysis & Machine Learning Model</strong><br>
  A data-driven approach to understanding and predicting forest fires in Algeria.
</p>

## 📌 Project Overview

In this project, we delve into the  dynamics of forest fires in Algeria, using a data-driven approach to both understand and predict these natural phenomena. The aim is to utilize the power of exploratory data analysis (EDA) and machine learning to extract valuable insights from historical data and to develop predictive models that can forecast forest fire occurrences based on meteorological conditions.

The dataset used in this project is sourced from UCI Machine Learning Repository, providing a rich set of variables including temperature, humidity, wind speed (Ws), rain, and other environmental factors recorded over specific periods. 
## 🚀 Technologies Used

- Python 🐍
- Pandas, NumPy 📊
- Matplotlib, Seaborn 📈
- Scikit-learn 🤖
- Jupyter Notebook 📓

## 📊 Data Analysis & Modeling
**Algerian_Forest_Fires_EDA.ipynb**
-  Perform feature engineering to create new variable (Classes) to help with classification prediction
-  Visualize data patterns using various plots to understand the relationship between meteorological conditions and fire events.
-  Carry out correlation analysis to identify which features have the strongest association with fire outbreaks.
-  Use these insights to guide our feature selection for the predictive model.

<table>
  <thead>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Temperature</td>
      <td>243.0</td>
      <td>32.152263</td>
      <td>3.628039</td>
      <td>22.0</td>
      <td>30.00</td>
      <td>32.0</td>
      <td>35.00</td>
      <td>42.0</td>
    </tr>
    <tr>
      <td>RH</td>
      <td>243.0</td>
      <td>62.041152</td>
      <td>14.828160</td>
      <td>21.0</td>
      <td>52.50</td>
      <td>63.0</td>
      <td>73.50</td>
      <td>90.0</td>
    </tr>
    <tr>
      <td>Ws</td>
      <td>243.0</td>
      <td>15.493827</td>
      <td>2.811385</td>
      <td>6.0</td>
      <td>14.00</td>
      <td>15.0</td>
      <td>17.00</td>
      <td>29.0</td>
    </tr>
    <tr>
      <td>Rain</td>
      <td>243.0</td>
      <td>0.762963</td>
      <td>2.003207</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>16.8</td>
    </tr>
    <tr>
      <td>FFMC</td>
      <td>243.0</td>
      <td>77.842387</td>
      <td>14.349641</td>
      <td>28.6</td>
      <td>71.85</td>
      <td>83.3</td>
      <td>88.30</td>
      <td>96.0</td>
    </tr>
    <tr>
      <td>DMC</td>
      <td>243.0</td>
      <td>14.680658</td>
      <td>12.393040</td>
      <td>0.7</td>
      <td>5.80</td>
      <td>11.3</td>
      <td>20.80</td>
      <td>65.9</td>
    </tr>
    <tr>
      <td>DC</td>
      <td>243.0</td>
      <td>49.430864</td>
      <td>47.665606</td>
      <td>6.9</td>
      <td>12.35</td>
      <td>33.1</td>
      <td>69.10</td>
      <td>220.4</td>
    </tr>
    <tr>
      <td>ISI</td>
      <td>243.0</td>
      <td>4.742387</td>
      <td>4.154234</td>
      <td>0.0</td>
      <td>1.40</td>
      <td>3.5</td>
      <td>7.25</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>BUI</td>
      <td>243.0</td>
      <td>16.690535</td>
      <td>14.228421</td>
      <td>1.1</td>
      <td>6.00</td>
      <td>12.4</td>
      <td>22.65</td>
      <td>68.0</td>
    </tr>
    <tr>
      <td>FWI</td>
      <td>243.0</td>
      <td>7.035391</td>
      <td>7.440568</td>
      <td>0.0</td>
      <td>0.70</td>
      <td>4.2</td>
      <td>11.45</td>
      <td>31.1</td>
    </tr>
  </tbody>
</table>

![bejaia_fires](https://github.com/user-attachments/assets/a6c1ff2f-4a15-4922-9203-5f593a11ec1d)

![sidi_bel_fires](https://github.com/user-attachments/assets/160c6a7a-8975-4b53-bd99-ca4c63853472)


**Algerian_Forest_Fire_Model.ipynb**
- Selection of appropriate algorithms based on the EDA outcomes.
- Model training with cross-validation to ensure robustness.
- Hyperparameter tuning to enhance model performance.
- Evaluation of the model using R2 Score and Mean Absolute Error
- Achieved a R2 Score of 97.3 with the tuned Random Forest Regressor
### Regression
```python
best_random_grid.fit(Xtrain_new_scaled, y_train)
bestrf_pred = best_random_grid.predict(Xtest_new_scaled)
mae = MAE(y_test, bestrf_pred)
r2 = r2_score(y_test, bestrf_pred)

print('Random Forest Tuned + Refined Data')
print('R2 Score Value {:.4f}'.format(r2))
print('MAE value: {:4f}'.format(mae))

```
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>R2 Score</td>
      <td>0.9785</td>
    </tr>
    <tr>
      <td>MAE</td>
      <td>0.615166</td>
    </tr>
  </tbody>
</table>

### Classification
XGB Boost Classifier
```python
# Import KNeighborsClassifier to Train from SKlearn
xgb = XGBClassifier()
xgb.fit(X_train_scaled,y_train)
xgb_pred = xgb.predict(X_test_scaled)
score = accuracy_score(y_test, xgb_pred)
cr = classification_report(y_test, xgb_pred)
```
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>R2 Score</td>
      <td>0.9726</td>
    </tr>
    <tr>
      <td>Weighted AVG F1 Score</td>
      <td>0.97</td>
    </tr>
  </tbody>
</table>

```python
xgb_cm = ConfusionMatrixDisplay.from_estimator(xgb, X_test_scaled, y_test)
```

![xgb_confusion](https://github.com/user-attachments/assets/acbd669a-5a6e-44ec-9f31-a16996b4e89b)


![top_features_fires](https://github.com/user-attachments/assets/78e18520-b8de-478f-b0d9-b8dd478f0142)



## 📢 Results & Insights
- Analyzed Distibutions of all features via Histogramns
- Identified top features when modeling forest fires predictions
- Developed a Decision Tree Regressor Model achieving a R2 Score of .9785 and an MAE Score of .6151 
- Developed a XGB Boost Classification Model achieving a R2 Score of 0.9726 and a weighted avg f1 score of .96
