import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load the data
url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nMissing values before filling:")
print(df.isnull().sum())

# Fill missing values with zeros
df = df.fillna(0)

print("\nMissing values after filling:")
print(df.isnull().sum())

# Prepare features and target
y = df['fuel_efficiency_mpg']
X = df.drop('fuel_efficiency_mpg', axis=1)

# Train/validation/test split (60%/20%/20%)
# First split: 60% train, 40% temp
X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Second split: split the 40% into 20% validation and 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=1
)

print(f"\nTrain set size: {len(X_train_full)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Convert to dictionaries and vectorize
train_dicts = X_train_full.to_dict(orient='records')
val_dicts = X_val.to_dict(orient='records')
test_dicts = X_test.to_dict(orient='records')

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

print(f"\nFeature names: {dv.get_feature_names_out()}")

# ============================================================
# Question 1: Decision Tree with max_depth=1
# ============================================================
print("\n" + "="*60)
print("QUESTION 1: Decision Tree with max_depth=1")
print("="*60)

dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train_full)

# Get the feature used for splitting
feature_idx = dt.tree_.feature[0]
feature_name = dv.get_feature_names_out()[feature_idx]

print(f"\nFeature used for splitting: {feature_name}")

# ============================================================
# Question 2: Random Forest with n_estimators=10
# ============================================================
print("\n" + "="*60)
print("QUESTION 2: Random Forest RMSE")
print("="*60)

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train_full)

y_pred_val = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"\nRMSE on validation data: {rmse:.4f}")

# ============================================================
# Question 3: n_estimators experimentation
# ============================================================
print("\n" + "="*60)
print("QUESTION 3: n_estimators experimentation")
print("="*60)

n_estimators_list = range(10, 201, 10)
rmse_scores = []

for n_est in n_estimators_list:
    rf = RandomForestRegressor(n_estimators=n_est, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train_full)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    print(f"n_estimators={n_est:3d}, RMSE={rmse:.6f}")

# Find when RMSE stops improving (to 3 decimal places)
rmse_rounded = [round(score, 3) for score in rmse_scores]
best_rmse = min(rmse_rounded)
best_idx = rmse_rounded.index(best_rmse)
best_n_estimators = list(n_estimators_list)[best_idx]

print(f"\nBest RMSE: {best_rmse:.3f}")
print(f"Achieved at n_estimators: {best_n_estimators}")

# Find where it stops improving
for i in range(len(rmse_rounded) - 1):
    if rmse_rounded[i+1] >= rmse_rounded[i]:
        print(f"\nRMSE stops improving after n_estimators={list(n_estimators_list)[i]}")
        break
else:
    print(f"\nRMSE keeps improving, best at n_estimators={list(n_estimators_list)[-1]}")

# ============================================================
# Question 4: Best max_depth using mean RMSE
# ============================================================
print("\n" + "="*60)
print("QUESTION 4: Best max_depth using mean RMSE")
print("="*60)

max_depth_list = [10, 15, 20, 25]
n_estimators_range = range(10, 201, 10)

results = {}

for max_depth in max_depth_list:
    rmse_list = []
    for n_est in n_estimators_range:
        rf = RandomForestRegressor(
            n_estimators=n_est, 
            max_depth=max_depth, 
            random_state=1, 
            n_jobs=-1
        )
        rf.fit(X_train, y_train_full)
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_list.append(rmse)
    
    mean_rmse = np.mean(rmse_list)
    results[max_depth] = mean_rmse
    print(f"max_depth={max_depth}, mean_RMSE={mean_rmse:.6f}")

best_max_depth = min(results, key=results.get)
print(f"\nBest max_depth: {best_max_depth}")

# ============================================================
# Question 5: Feature Importance
# ============================================================
print("\n" + "="*60)
print("QUESTION 5: Feature Importance")
print("="*60)

rf = RandomForestRegressor(
    n_estimators=10, 
    max_depth=20, 
    random_state=1, 
    n_jobs=-1
)
rf.fit(X_train, y_train_full)

# Get feature importances
feature_names = dv.get_feature_names_out()
importances = rf.feature_importances_

# Create a dictionary of feature importances
feature_importance_dict = dict(zip(feature_names, importances))

# Filter for the four features in question
target_features = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
print("\nFeature importances for target features:")
for feat in target_features:
    if feat in feature_importance_dict:
        print(f"{feat}: {feature_importance_dict[feat]:.6f}")

# Find the most important among these
target_importances = {feat: feature_importance_dict[feat] 
                      for feat in target_features 
                      if feat in feature_importance_dict}
most_important = max(target_importances, key=target_importances.get)
print(f"\nMost important feature: {most_important}")

# ============================================================
# Question 6: XGBoost with different eta values
# ============================================================
print("\n" + "="*60)
print("QUESTION 6: XGBoost with different eta values")
print("="*60)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train_full)
dval = xgb.DMatrix(X_val, label=y_val)

watchlist = [(dtrain, 'train'), (dval, 'val')]

# Train with eta=0.3
print("\nTraining with eta=0.3:")
xgb_params_03 = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model_03 = xgb.train(
    xgb_params_03, 
    dtrain, 
    num_boost_round=100,
    evals=watchlist,
    verbose_eval=False
)

y_pred_03 = model_03.predict(dval)
rmse_03 = np.sqrt(mean_squared_error(y_val, y_pred_03))
print(f"RMSE with eta=0.3: {rmse_03:.6f}")

# Train with eta=0.1
print("\nTraining with eta=0.1:")
xgb_params_01 = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model_01 = xgb.train(
    xgb_params_01, 
    dtrain, 
    num_boost_round=100,
    evals=watchlist,
    verbose_eval=False
)

y_pred_01 = model_01.predict(dval)
rmse_01 = np.sqrt(mean_squared_error(y_val, y_pred_01))
print(f"RMSE with eta=0.1: {rmse_01:.6f}")

print("\n" + "="*60)
print("SUMMARY OF ANSWERS")
print("="*60)
if rmse_03 < rmse_01:
    best_eta = "0.3"
elif rmse_01 < rmse_03:
    best_eta = "0.1"
else:
    best_eta = "Both give equal value"

print(f"\nBest eta: {best_eta}")

print("\nAll answers:")
print(f"Q1: Feature used for splitting: {feature_name}")
print(f"Q2: RMSE with n_estimators=10: {np.sqrt(mean_squared_error(y_val, y_pred_val)):.2f}")
print(f"Q3: n_estimators stops improving at: {best_n_estimators}")
print(f"Q4: Best max_depth: {best_max_depth}")
print(f"Q5: Most important feature: {most_important}")
print(f"Q6: Best eta: {best_eta}")
