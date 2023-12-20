import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FitFailedWarning)

# Load the CSV file
file_path = "./geological_data.csv"
data = pd.read_csv(file_path)
data = data.drop_duplicates()
data.head()
print(data.describe())
print(data.isnull().sum())

selected_columns = ['gold_grade', 'X_Coordinate', 'Y_Coordinate', 'Z_Coordinate']
subset_data = data[selected_columns]
descriptive_stats = subset_data.describe()
print(descriptive_stats)

# Pearson and Spearman's Rank Correlation Coefficients
pearson_corr = data[['gold_grade', 'X_Coordinate', 'Y_Coordinate', 'Z_Coordinate']].corr(method='pearson')
spearman_corr = data[['gold_grade', 'X_Coordinate', 'Y_Coordinate', 'Z_Coordinate']].corr(method='spearman')
print(" Pearson Correlation:\n",pearson_corr,"\n")
print(" Spearman Correlation:\n",spearman_corr)

x = data['collar.x']
y = data['collar.y']
z = data['collar.z']

# Creating the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=data['gold_grade'], cmap='viridis')

# Adding color bar to represent the gold grade
colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label('Gold Grade')

# Setting labels
ax.set_xlabel('Collar X Coordinate')
ax.set_ylabel('Collar Y Coordinate')
ax.set_zlabel('Collar Z Coordinate (Elevation)')

# Adding title
ax.set_title('3D Plot of Drill Hole Collar Coordinates with Gold Grade Coloring')
plt.show()

# Creating a 3D plot including more feature columns
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Using 'X_Coordinate', 'Y_Coordinate', and 'Z_Coordinate' for the 3D space
x = data['X_Coordinate']
y = data['Y_Coordinate']
z = data['Z_Coordinate']

# Using 'Sample depth' as the size of the points. We will normalize the size for better visualization.
sizes = (data['Sample depth'] - data['Sample depth'].min()) / (data['Sample depth'].max() - data['Sample depth'].min()) * 100

# Color by 'gold' grade
colors = data['gold_grade']

# Scatter plot
scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=sizes)

# Adding color bar to represent the gold grade
colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label('Gold Grade')

# Setting labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate (Elevation)')

# Adding title
ax.set_title('3D Plot of Sample Coordinates with Gold Grade and Sample Depth Size Encoding')
plt.show()

# Prepare the features and target for modeling
selected_features = ['depth_from', 'depth_to', 'collar.x', 'collar.y', 'collar.z', 'Sample depth', 'Dip', 'Azimuth', 'X_Coordinate', 'Y_Coordinate', 'Z_Coordinate']

features = data[selected_features]
target = data['gold_grade']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Splitting the data into training, validation, and testing sets.
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) 

# RandomForest Hyperparameter Grid
rf_param_grid = {'n_estimators': [100, 500, 1000],
                 'max_depth': [5, 10, 20],
                 'min_samples_split': [5, 10], 
                 'max_features': ['auto', 'log2']}

# XGBoost Hyperparameter Grid
xgb_param_grid = {'n_estimators': [100, 500, 1000],
                  'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'subsample': [0.7, 0.8, 0.9]}

# Randomized search for RF
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=10, cv=5, random_state=42)
rf_search.fit(X_train, y_train)

# Randomized search for XGBoost
xgboost = xgb.XGBRegressor(n_jobs=-1, random_state=42)
xgb_search = RandomizedSearchCV(xgboost, xgb_param_grid, n_iter=10, cv=5, random_state=42)
xgb_search.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)

# Ensemble model
ensemble = VotingRegressor([('rf', rf_search.best_estimator_), ('xgb', xgb_search.best_estimator_)])

# Fit ensemble on the training data
ensemble.fit(X_train, y_train)

print("\n")
print("Random Forest and XGBoost Ensemble Model Metrics:")
# Evaluate ensemble on test data
y_pred = ensemble.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print("RMSE:", rmse)
print("MAE:", mae)
print("\n")
try:
    data = pd.read_csv(file_path)
    # Saving the trained ensemble model
    joblib.dump(ensemble, 'ensemble_model.joblib')

    # Cross-validation
    scores = cross_val_score(ensemble, X_train, y_train, cv=5)
    print("Cross-Validation Scores:", scores)
    print("\n")
    
except Exception as e:
    print("An error occurred:", e)
 
print("AdaBoost Model R2 Score Results for Comaparision With RF ensemble Model:")
# Building and evaluating the AdaBoost model.
ada = AdaBoostRegressor(random_state=42)
ada.fit(X_train, y_train)
ada_predictions = ada.predict(X_test)
ada_r2 = r2_score(y_test, ada_predictions)
ada_mse = mean_squared_error(y_test, ada_predictions)
print(f"AdaBoost - R2 Score: {ada_r2}, MSE: {ada_mse}")
print("\n")
print("Bagging Model R2 Score Results for Comaparision With RF ensemble Model:")
# Building and evaluating the Bagging model.
bagging = BaggingRegressor(random_state=42)
bagging.fit(X_train, y_train)
bagging_predictions = bagging.predict(X_test)
bagging_r2 = r2_score(y_test, bagging_predictions)
bagging_mse = mean_squared_error(y_test, bagging_predictions)
print(f"Bagging - R2 Score: {bagging_r2}, MSE: {bagging_mse}")