# diabetes-prediction

# Diabetes Prediction with Logistic Regression

Predicting whether a patient has diabetes from diagnostic measurements, using the Pima Indians Diabetes dataset (768 patients, 8 features).

## Problem
Binary classification: given measurements like glucose, BMI, age and insulin, predict `Outcome` (1 = diabetes, 0 = no diabetes).

## Approach
1. **Data cleaning**
   - Found that Glucose, BloodPressure, SkinThickness, Insulin and BMI contain zeros that are biologically impossible, so I treated them as missing values.
   - Filled missing values with the median (Insulin was filled per outcome group).
   - Checked outliers with the IQR method and clipped extreme values in Insulin and SkinThickness.
2. **Train/test split:** 80/20, stratified on the target to keep the class balance.
3. **Scaling:** `StandardScaler` fitted on the training set only, then applied to the test set.
4. **Model:** Logistic Regression from scikit-learn.
5. **Evaluation:** classification report, confusion matrix and ROC-AUC.

## Results
| Metric | Value |
|---|---|
| Accuracy | [your value] |
| Recall (diabetes class) | [your value] |
| ROC-AUC | [your value] |

**Key observations**
- [Which features had the largest coefficients, e.g. Glucose and BMI]
- [One line on what the confusion matrix shows, e.g. how many diabetic cases were missed]

## Limitations and next steps
- Missing-value imputation and outlier clipping were done before the train/test split, which can make the test score slightly optimistic. Next step: move all preprocessing into a scikit-learn `Pipeline` so it is fitted on training data only.
- Try L1 (Lasso) regularization to see which features get dropped.
- Compare with other models (e.g. Random Forest) and tune the decision threshold to improve recall.

## Tech
Python, pandas, NumPy, scikit-learn, Google Colab

## Dataset
[Pima Indians Diabetes dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
