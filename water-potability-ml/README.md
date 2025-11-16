# *Predicting Water Potability for Safe Drinking Access*

## *Project Overview*

Problem Statement: Over 2 billion people worldwide lack access to safe drinking water. This project builds an ML classification system to predict whether water is safe for human consumption based on chemical properties and quality measurements.

**Real-World Impact: Such systems can help:**

Resource-limited communities prioritize water testing
NGOs and governments identify high-risk water sources
Reduce waterborne diseases through early detection

## **ML Task: Binary Classification (Potable vs Non-Potable)**

## **Dataset**

Primary Dataset: [Water Quality/Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

- Size: 3,276 rows × 10 columns (~200 KB)
- Features: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity
- Target: Potability (0 = Not Safe, 1 = Safe)
- Challenge: Imbalanced classes (~39% potable), missing values

## **Key Metrics & Why They Matter**

- Accuracy: Overall correctness (baseline metric)
- Precision: Of water labeled "safe," how many truly are? (Critical - false positives risk health!)
- Recall: Of all safe water, how much did we catch? (Important for coverage)
- F1-Score: Balance between precision/recall
- AUC-ROC: Model's ability to distinguish classes across thresholds
- Confusion Matrix: Visualize false positives (dangerous!) vs false negatives

Priority: Minimize false positives (predicting unsafe water as safe) - this is a safety-critical application.
