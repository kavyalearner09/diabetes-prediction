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
Evaluated on 154 held-out test patients.

| Metric | Value |
|---|---|
| Accuracy | 0.75 |
| Precision (diabetes class) | 0.64 |
| Recall (diabetes class) | 0.65 |
| ROC-AUC | 0.83 |

Confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 80 | 20 |
| **Actual 1** | 19 | 35 |

## Key observations
- The model separates diabetic from non-diabetic patients reasonably well (ROC-AUC 0.83). It does better on non-diabetic cases (recall 0.80) than on diabetic ones (0.65).
- It correctly identified 35 of 54 diabetic patients and missed 19. In a medical setting a missed diagnosis is the costlier error, so lowering the decision threshold to improve recall would be a sensible next step.
- Feature coefficients (on scaled features): Insulin (0.91), Glucose (0.82) and BMI (0.54) had the largest positive effect on predicting diabetes. Pregnancies (0.33) and Age (0.18) contributed moderately, while BloodPressure and SkinThickness were close to zero.

## Limitations and next steps
- **Possible target leakage:** Insulin's missing values were filled using the median of each `Outcome` group, so the label leaks into that feature. This likely inflates Insulin's coefficient (it ranked above Glucose, which is unusual for this dataset) and the test scores. Next step: refill with a label-free median and compare results.
- Missing-value imputation and outlier clipping were done before the train/test split. Next step: move all preprocessing into a scikit-learn `Pipeline` so it is fitted on training data only.
- Try L1 (Lasso) regularization to see which features get dropped.
- Tune the decision threshold to trade some precision for higher recall on the diabetic class.
- Compare with other models (e.g. Random Forest).

## Tech
Python, pandas, NumPy, scikit-learn, Google Colab

## Dataset
[Pima Indians Diabetes dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
