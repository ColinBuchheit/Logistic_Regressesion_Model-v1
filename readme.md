# Online Shopping Purchase Prediction - Logistic Regression Classifier

## Overview

This project implements a logistic regression classifier to predict whether a user will make a purchase during an online shopping session. The prediction is based on user session data such as page interactions, activity duration, browser type, traffic type, and whether the session occurred on a weekend. The model has been developed using custom feature scaling methods and evaluates the impact of regularization on model performance.

## Dataset

The dataset contains approximately 5,000 user sessions with the following key columns:
- **Administrative, Administrative_Duration**: Number of administrative pages visited and the time spent.
- **Informational, Informational_Duration**: Number of informational pages visited and the time spent.
- **ProductRelated, ProductRelated_Duration**: Number of product-related pages visited and the time spent.
- **BounceRates, ExitRates**: User interaction metrics.
- **PageValues**: Values assigned to page visits from Google Analytics.
- **SpecialDay**: Indicator of proximity to a special day (e.g., Valentine's Day).
- **Month**: The month of the shopping session.
- **VisitorType**: Distinguishes between Returning Visitor, New Visitor, or Other.
- **Weekend**: Binary indicator (TRUE/FALSE) if the session occurred on a weekend.
- **Revenue**: The target label, indicating whether a purchase was made (TRUE/FALSE).

## Feature Transformation

- **Weekend** and **Revenue** columns are mapped to numerical values: `TRUE` to `1`, and `FALSE` to `0`.
- **Month** is converted into numerical values (e.g., `Jan` = 1, `Feb` = 2, ...).
- **VisitorType** is mapped to `1` for Returning Visitors, `2` for New Visitors, and `3` for Others.

## Scaling Methods

Three custom scaling methods have been implemented to normalize features:
- **MinMax Scaling**: Rescales the data to fit within the range `[0,1]`.
- **Mean Normalization**: Centers the data around zero by subtracting the mean and dividing by the range.
- **Z-Score Normalization**: Standardizes the data by subtracting the mean and dividing by the standard deviation.

## Model Training

- The logistic regression classifier uses gradient descent for optimization.
- The model was trained using 80% of the data, with 20% used for testing.
- Early stopping was implemented during gradient descent to prevent overfitting.

## Regularization

- Regularization was applied to prevent overfitting by penalizing large coefficients in the model.
- Two versions of the model were tested: with and without regularization (λ=100).

## Results

| Scaling Method         | Accuracy | Precision | Recall  | F1 Score |
|------------------------|----------|-----------|---------|----------|
| No Scaling              | 0.6986   | 0.6362    | 0.9242  | 0.7536   |
| MinMax Scaling          | 0.6803   | 0.6312    | 0.8637  | 0.7294   |
| Mean Normalization      | 0.6726   | 0.6495    | 0.7464  | 0.6946   |
| Z-Score Normalization   | 0.7837   | 0.8474    | 0.6908  | 0.7611   |

### Regularization Impact

- **Without Regularization**:  
  Accuracy = 0.6738, Precision = 0.6308, Recall = 0.8341, F1 Score = 0.7184
- **With Regularization (λ=100)**:  
  Accuracy = 0.6803, Precision = 0.6312, Recall = 0.8637, F1 Score = 0.7294

### Cost Function Graph

A graph illustrating the cost function for both regularized and non-regularized models was generated. Regularization shows smoother convergence, reducing overfitting in high-dimensional data.

## Conclusion

- **Best Scaling Method**: Z-Score Normalization yielded the best results with the highest accuracy of 0.7837.
- **Regularization**: Improved recall and overall performance by reducing overfitting.
- **Future Work**: Precision can be improved by experimenting with different regularization parameters or testing more complex models like decision trees or ensemble methods.


## Challenges Faced and Solutions

### 1. **Class Imbalance in the Dataset**
   **Issue**: The dataset had a class imbalance, with significantly more non-purchase sessions (labeled `0` in the `Revenue` column) than purchase sessions (labeled `1`). This imbalance can lead to biased model predictions, favoring the majority class.
   
   **Solution**: To address this issue, I applied **upsampling** on the minority class (purchase sessions) to balance the dataset. By resampling the minority class and creating a balanced dataset, I ensured the model received a fair representation of both classes during training.

### 2. **Feature Scaling Implementation**
   **Issue**: Implementing feature scaling methods such as MinMax Scaling, Mean Normalization, and Z-Score Normalization without using external libraries presented a challenge in terms of correct computation and consistency.

   **Solution**: I wrote custom functions for each scaling method to rescale the features. I tested these functions to ensure they transformed the data correctly before passing the data into the model. This ensured that the model performed fairly across different scales.

### 3. **Regularization Effect**
   **Issue**: Initially, the performance of the model did not significantly change between the regularized and non-regularized versions. Both models showed similar accuracy and performance metrics, which was unexpected.

   **Solution**: After investigating the issue, I realized that the regularization parameter (lambda) needed fine-tuning. By increasing the value of `lambda` (regularization strength), I observed improved recall and overall performance for the regularized model. This helped the model avoid overfitting while achieving better generalization.

### 4. **Early Stopping in Gradient Descent**
   **Issue**: The gradient descent process for training the model required fine-tuning of the iteration limit and tolerance for early stopping. Without proper early stopping, the model was prone to either underfitting or overfitting.

   **Solution**: I introduced an **early stopping mechanism** based on the change in cost between iterations. If the cost reduction was below a certain tolerance threshold, the training would stop. This allowed the model to converge efficiently while preventing unnecessary iterations.

### 5. **Graphing Cost Function Behavior**
   **Issue**: When visualizing the cost function over time, I encountered erratic behavior in the cost values for the non-regularized model. The cost fluctuated dramatically, making the graph difficult to interpret.

   **Solution**: After analyzing the cause, I realized that the lack of regularization was causing the model to overfit, leading to unstable cost values. By introducing regularization, I was able to smooth out the cost function graph and reduce the fluctuations, showing more predictable convergence.

### 6. **Data Preprocessing: String to Numerical Conversion**
   **Issue**: The dataset contained categorical data (e.g., `Month`, `VisitorType`, `Weekend`) that needed to be converted into numerical values for model training. Initially, this caused errors during training.

   **Solution**: I created mappings to convert categorical string values into numerical representations. For instance, I mapped months (`Jan`, `Feb`, etc.) to numbers (`1`, `2`, etc.), and visitor types (`Returning_Visitor`, `New_Visitor`) to integers (`1`, `2`). This allowed the model to handle the categorical data correctly.
   


## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/[your-username]/shopping-prediction.git
    ```
2. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook project1_logistic_regression.ipynb
    ```


