# *Homework-6*

I'll help you solve this homework step by step. Let me start by loading the data and preparing it according to the instructions.Now I'll create a Python script to solve all the questions. Since this is a machine learning homework that requires scikit-learn and XGBoost, I'll provide you with a complete Python script:I've created a complete Python script that solves all 6 questions in your homework. Here's what the script does:

```python
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
```

## Data Preparation

- Loads the dataset from the URL
- Fills missing values with zeros
- Performs 60%/20%/20% train/validation/test split with `random_state=1`
- Uses `DictVectorizer(sparse=True)` to convert dataframes to matrices

## Solutions

**Question 1**: Trains a Decision Tree with `max_depth=1` and identifies which feature is used for splitting

**Question 2**: Trains a Random Forest with `n_estimators=10` and calculates RMSE on validation data

**Question 3**: Tests different values of `n_estimators` from 10 to 200 (step 10) to find when RMSE stops improving (rounded to 3 decimal places)

**Question 4**: Tries different `max_depth` values [10, 15, 20, 25] with varying `n_estimators` and finds the best `max_depth` based on mean RMSE

**Question 5**: Trains a model with specific parameters and extracts feature importances to find the most important feature among the four specified

**Question 6**: Trains XGBoost models with `eta=0.3` and `eta=0.1` and compares their RMSE scores

To run this script:

1. Make sure you have the required libraries installed:

   ```bash
   pip install pandas numpy scikit-learn xgboost
   ```

2. Run the Python script
3. The script will print all intermediate results and a summary of answers at the end

- The script is well-commented and prints detailed output for each question, making it easy to verify the results and understand the process

---

Great question! Let's break down this line in detail:

```python
most_important = max(target_importances, key=target_importances.get)
```

## What Does `max()` Do?

The `max()` function finds the **maximum** value, but it can work in two ways:

### **Without `key` parameter:**

```python
numbers = [3, 7, 2, 9, 1]
biggest = max(numbers)  # Returns 9
```

Simple - it just finds the biggest number.

### **With `key` parameter:**

When you have a dictionary, `max()` by default looks at the **keys** (not the values):

```python
scores = {'Alice': 95, 'Bob': 87, 'Charlie': 92}
result = max(scores)  # Returns 'Charlie' (last alphabetically)
# This is NOT what we want!
```

## The Problem We're Solving

We have a dictionary like this:

```python
target_importances = {
    'vehicle_weight': 0.32,
    'horsepower': 0.18,
    'acceleration': 0.095,
    'engine_displacement': 0.125
}
```

We want to find which feature has the **highest importance score** (the values), not which feature name comes last alphabetically (the keys).

## How `key=target_importances.get` Works

The `key` parameter tells `max()` **how to compare** items. Let's break it down:

### **Step 1: What does `.get` do?**

```python
target_importances.get('vehicle_weight')  # Returns 0.32
target_importances.get('horsepower')      # Returns 0.18
```

`.get()` is a dictionary method that retrieves the **value** for a given key.

### **Step 2: How `max()` uses the `key` function**

When you write `key=target_importances.get`, here's what happens internally:

```python
# max() iterates through each key and compares their values
# 
# For 'vehicle_weight': target_importances.get('vehicle_weight') → 0.32
# For 'horsepower': target_importances.get('horsepower') → 0.18
# For 'acceleration': target_importances.get('acceleration') → 0.095
# For 'engine_displacement': target_importances.get('engine_displacement') → 0.125
#
# max() compares: 0.32 vs 0.18 vs 0.095 vs 0.125
# Winner: 0.32
# Returns: 'vehicle_weight' (the key associated with 0.32)
```

## Visual Comparison

```python
# WITHOUT key (WRONG - compares keys alphabetically):
max(target_importances)
# Compares: 'acceleration' vs 'engine_displacement' vs 'horsepower' vs 'vehicle_weight'
# Returns: 'vehicle_weight' (just because 'v' comes last alphabetically!)

# WITH key=target_importances.get (CORRECT - compares values):
max(target_importances, key=target_importances.get)
# Compares: 0.095 vs 0.125 vs 0.18 vs 0.32
# Returns: 'vehicle_weight' (because 0.32 is the highest score)
```

## Alternative Ways to Write This

Here are equivalent ways to achieve the same result:

### **Option 1: Using a lambda function**

```python
most_important = max(target_importances, key=lambda x: target_importances[x])
```

This is more explicit but means the same thing.

### **Option 2: Using `items()` and unpacking**

```python
most_important = max(target_importances.items(), key=lambda x: x[1])[0]
# x[1] is the value, [0] gets the key from the (key, value) tuple
```

### **Option 3: Sorting (less efficient but more readable)**

```python
sorted_features = sorted(target_importances.items(), 
                        key=lambda x: x[1], 
                        reverse=True)
most_important = sorted_features[0][0]  # Get the first key
```

## Summary

```python
most_important = max(target_importances, key=target_importances.get)
```

- **`max(target_importances, ...)`**: Look through all the keys in the dictionary
- **`key=target_importances.get`**: But compare them by their **values** (importance scores)
- **Returns**: The **key** (feature name) that has the maximum value

So if `vehicle_weight` has importance 0.32 (the highest), this returns the string `'vehicle_weight'`.

Does that make sense now? 😊
