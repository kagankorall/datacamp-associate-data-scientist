# Machine Learning with Tree-Based Models in Python

## Course Overview
Comprehensive course covering tree-based machine learning algorithms including decision trees, ensemble methods, and hyperparameter tuning for both classification and regression tasks.

## Key Topics Covered

### 1. Classification and Regression Trees (CARTs)
- Decision tree fundamentals
- Information Gain and splitting criteria
- Decision tree classification
- Decision tree regression

### 2. The Bias-Variance Tradeoff
- Generalization error decomposition
- Overfitting and underfitting
- Cross-validation techniques
- Diagnosing and remedying bias/variance problems

### 3. Bagging and Random Forests
- Bootstrap aggregation (Bagging)
- Out-of-Bag (OOB) evaluation
- Random Forest algorithm
- Feature importance analysis

### 4. Boosting
- AdaBoost (Adaptive Boosting)
- Gradient Boosting (GB)
- Stochastic Gradient Boosting (SGB)
- Sequential error correction

### 5. Model Tuning
- Hyperparameter optimization
- Grid Search Cross-Validation
- Model selection strategies
- Performance optimization

## Key Concepts

### Decision Trees
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

### Random Forests
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=0.12,
    random_state=1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingRegressor

gbt = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=1,
    random_state=1
)
gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_test)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

params_rf = {
    'n_estimators': [300, 400, 500],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [0.1, 0.2],
    'max_features': ['log2', 'sqrt']
}

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=params_rf,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)
best_model = grid_rf.best_estimator_
```

## Course Notes

# Classification and Regression Trees (CARTs)

## Decision Tree for Classification

### Classification tree

A sequence of if-else questions about individual features. The objective is to infer class labels. Able to capture non-linear relationships between features and labels. Tree-based models do not require feature scaling (for example, standardization).

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

dt = DecisionTreeClassifier(max_depth=2, random_state=1)

[dt.fit](http://dt.fit)(X_train, y_train)

y_pred = dt.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
```

Decision region: A region in the feature space where all instances are assigned to one class label.

Decision boundary: A surface separating different decision regions.

## Classification Tree Learning

A decision tree is a structure consisting of a hierarchy of nodes.

A branch is a connection that links nodes.

A node is a question or a prediction. There are three kinds of nodes:

- Root: No parent node. A question that gives rise to two child nodes.
- Internal node: One parent node. A question that gives rise to two child nodes.
- Leaf: One parent node and no child nodes, which produces a prediction.

### Information Gain (IG)

Information Gain is a way to measure how useful a feature is for splitting data in a decision tree.

Learning steps in a tree:

1. Nodes are grown recursively.
2. At each node, split the data based on feature f and split point sp to maximize IG(node).
3. If IG(node) = 0, declare the node a leaf.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

dt = DecisionTreeClassifier(criterion='gini', random_state=1)

[dt.fit](http://dt.fit)(X_train, y_train)

y_pred = dt.predict(X_test)

print('Accuracy (gini):', accuracy_score(y_test, y_pred))
```

## Decision Tree for Regression

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

dt = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=0.10,
    random_state=3
)

[dt.fit](http://dt.fit)(X_train, y_train)

y_pred = dt.predict(X_test)

mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt ** 0.5
print('RMSE (DT):', rmse_dt)
```

# The Bias-Variance Tradeoff

## Generalization Error

The goal of supervised learning is to find the best model, f_hat, that best approximates f, which means f_hat ≈ f.

Discard noise as much as possible.

End goal: f_hat should achieve low predictive error on unseen datasets.

Difficulties in approximating f:

1. Overfitting: f_hat(x) fits the training set noise.
2. Underfitting: f_hat is not flexible enough to approximate f.

### Overfitting

In overfitting, the model achieves low training set error but high test set error.

### Underfitting

In underfitting, the training set error is roughly equal to the test set error, and both errors are relatively high.

### Generalization Error

Generalization error of f_hat: Does f_hat generalize well to unseen data?

It can be decomposed as follows:

Generalization error of f_hat = bias^2 + variance + irreducible error

Bias: An error term that tells you, on average, how much f_hat differs from f. High-bias models lead to underfitting.

Variance: Tells you how much f_hat is inconsistent over different training sets. High-variance models lead to overfitting.

Model complexity sets the flexibility of f_hat. For example, increasing the maximum tree depth in decision tree models increases model complexity. As model complexity increases, bias decreases.

## Diagnosing Bias and Variance Problems

### Estimating the Generalization Error

Solution:

- Split the data into training and test sets.
- Fit f_hat to the training set.
- Evaluate the error of f_hat on the unseen test set.
- The generalization error of f_hat is approximately the test set error of f_hat.

### Better Model Evaluation with Cross-Validation

The test set should not be touched until you are confident about f_hat’s performance.

Evaluating f_hat on the training set gives a biased estimate because f_hat has already seen all training points. The solution is cross-validation (K-fold CV or hold-out CV).

For example, using K-fold CV with k = 10:

CV error = (E1 + … + E10) / 10

### Diagnose Variance Problems

If f_hat suffers from high variance, the CV error of f_hat is greater than the training set error of f_hat. In this case, f_hat overfits the training set.

To remedy overfitting:

- Decrease model complexity, for example decrease max depth or increase min samples per leaf.
- Gather more data.

### Diagnose Bias Problems

If f_hat suffers from high bias, the CV error of f_hat is nearly equal to the training set error of f_hat and both are much larger than the desired error. In this case, f_hat underfits the training set.

To remedy underfitting:

- Increase model complexity, for example increase max depth.
- Gather more relevant features.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 123

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

dt = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=0.14,
    random_state=SEED
)

MSE_CV = -cross_val_score(
    dt, X_train, y_train, cv=10,
    scoring='neg_mean_squared_error', n_jobs=-1
)

[dt.fit](http://dt.fit)(X_train, y_train)

y_pred_train = dt.predict(X_train)

y_pred_test = dt.predict(X_test)

# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))

# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_pred_train)))

# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_pred_test)))
```

If the training set error is smaller than the cross-validation error, we can deduce that dt overfits the training set and suffers from high variance.

## Ensemble Learning

CARTs are simple to understand, simple to interpret, easy to use, and flexible. They can describe non-linear dependencies and do not need standardized or normalized features.

They also have some limitations. In classification, they produce orthogonal decision boundaries. They are very sensitive to small variations in the training set. Because of high variance, unconstrained CARTs may overfit the training set. A solution is ensemble learning.

### Ensemble Learning

1. Train different models on the same dataset.
2. Let each model make its predictions.
3. Meta-model: Aggregate the predictions of individual models.
4. Final prediction: More robust and less prone to errors.
5. Best results: Models are skillful in different ways.

Ensemble learning in practice: Voting Classifier

Binary classification task.

N classifiers make predictions: P1, P2, …, PN with Pi = 0 or 1.

Meta-model prediction: hard voting.

**Hard voting** is a method where each trained model makes discrete predictions, such as 0 or 1. For example, if a Decision Tree predicts 1, Logistic Regression predicts 0, and KNN predicts 1, the Voting Classifier counts the votes and selects the class with the highest number of votes as the final decision. In this case, the result would be 1, since two models predicted 1 and only one predicted 0.

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)

classifiers = [
    ('Logistic Regression', lr),
    ('K Nearest Neighbours', knn),
    ('Classification Tree', dt)
]

for clf_name, clf in classifiers:
    # Fit clf to the training set
    [clf.fit](http://clf.fit)(X_train, y_train)
    
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))

# VotingClassifier
vc = VotingClassifier(estimators=classifiers)
[vc.fit](http://vc.fit)(X_train, y_train)
y_pred = vc.predict(X_test)
print('Voting Classifier: {:.3f}'.format(accuracy_score(y_test, y_pred)))
```

# Bagging and Random Forests

## Bagging

In a Voting Classifier, different algorithms are trained on the same training set. In contrast, in bagging, a single algorithm is trained on different subsets of the training set.

Bagging stands for Bootstrap Aggregation. It uses a technique known as the bootstrap and reduces the variance of individual models in the ensemble.

In classification, bagging aggregates predictions by majority voting (`BaggingClassifier`). In regression, bagging aggregates predictions through averaging (`BaggingRegressor`).

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=0.16,
    random_state=SEED
)

bc = BaggingClassifier(
    estimator=dt, n_estimators=300, n_jobs=1
)

[bc.fit](http://bc.fit)(X_train, y_train)

y_pred = bc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))
```

## Out-of-Bag (OOB) Evaluation

Some instances may be sampled several times for one model, while other instances may not be sampled at all.

On average, for each model, 63% of the training instances are sampled. The remaining 37% constitute the OOB instances.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=0.16,
    random_state=SEED
)

bc = BaggingClassifier(
    estimator=dt, n_estimators=300, oob_score=True, n_jobs=1
)

[bc.fit](http://bc.fit)(X_train, y_train)

y_pred = bc.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', test_accuracy)
print('OOB accuracy:', bc.oob_score_)
```

## Random Forests (RF)

The base estimator in a Random Forest is a decision tree. Each estimator is trained on a different bootstrap sample having the same size as the training set. Random Forests introduce further randomization in the training of individual trees.

At each node, d features are sampled without replacement (d < total number of features).

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

rf = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=0.12,
    random_state=SEED
)

[rf.fit](http://rf.fit)(X_train, y_train)

y_pred = rf.predict(X_test)

rmse_test = MSE(y_test, y_pred) ** 0.5
print('RMSE (RF):', rmse_test)
```

### Feature Importance

Tree-based methods enable measuring the importance of each feature in prediction. In scikit-learn, feature importance reflects how much the tree nodes use a particular feature (weighted average) to reduce impurity.

```python
import pandas as pd
import matplotlib.pyplot as plt

importances_rf = pd.Series(rf.feature_importances_, index=X.columns)

sorted_importances_rf = importances_rf.sort_values()

sorted_importances_rf.plot(kind='barh', color='lightgreen')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
[plt.show](http://plt.show)()
```

# Boosting

## AdaBoost (Adaptive Boosting)

Boosting: Ensemble method combining several weak learners to from a strong learner. A weak learner means a model doing slightly better than random guessing.

In boosting, train an ensemble of predictors sequentially. Each predictor tries to correct errors made by its predecessor.

In AdaBoost, each predictor pays more attention to the instances wrongly predicted by its predecessor. Achieved by changing the weights of training instances. Each predictor is assigned a coefficient alpha. Alpha depends on the predictor’s training error.

Learning rate (eta) must be between 0 and 1.

In classification → Weighted majority voting

In regression → Weighted average

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

dt = DecisionTreeClassifier(max_depth = 1, random_state = SEED)

adb_clf = AdaBoostClassifier(base_estimator = dt, n_estimators = 100)

[adb_clf.fit](http://ada.fit)(X_train, y_train)

y_pred_proba = adb_clf.predict_proba(X_test)[:,1]

adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
```

## Gradient Boosting (GB)

Sequential correction of predecessor’s errors. Does not tweak the weights of training instances. Fit each predictor is trained using its predecessor’s residual errors as labels.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

gbt = GradientBoostingRegressor(n_estimators=300, max_depth = 1, random_state = SEED)

gbt.fit(X_train, y_train)

y_pred = gbt.predict(X_test)

rmse_test = MSE(y_test, y_pred)**(1/2)
```

## Stochastic Gradient Boosting (SGB)

GB involves an exhaustive search procedure. Each CART is trained to find the best split points and features. This procedure may lead to CARTs using the same split points and maybe the same features.

In Stochastic Gradient Boosting, each tree is trained on a random subset of rows of the training data. The sampled instances are sampled without replacement. Features are sampled (without replacement) when choosing split points.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

sgbt = GradientBoostingRegressor(max_depth=1,
																 subsample=0.8,
																 max_features=0.2,
																 n_estimators=300,
																 random_state=SEED)
sgbt.fit(X_train, y_train)

y_pred = sgbt.predict(X_test)

rmse_test = MSE(y_test, y_pred)**(1/2)
```

# Model Tuning

## Tuning a CARTs Hyperparameters

### Hyperparameters

In machine learning model, parameters are learned from the data. For example, in CARTs split-point of a node or split-feature of a node are parameters.

Hyperparameters are not learned from data, set prior to training. For example, in CARTs max_depth, min_samples_lead, splittin criterion are hyperparameters.

Hyperparameter tuning is search for a set of optimal hyperparameters for a learning algorithm. Solution is to find a set of optimal hyperparameters that results in an optimal model. Optimal model yields an optimal score. Score is defaults to accuracy (classification) and R^2 (regression) in sklearn.

### Grid Search CV

Manually set a grid of discrete hyperparameter values. Set a metric for scoring model performance. Search exhaustively through the grid. For each set of hyperparameters, evaluate each model’s CV score. The optimal hyperparameter are those of the model achieving the best CV score.

```python
from sklearn.tree import DecisionTreeClassifier

SEED = 1

dt = DecisionTreeClassifier(random_state = SEED)

print(dt.get_params())

from sklearn.model_selection import GridSearchCV

params_df = {'max_depth':[3,4,5,6],
						 'min_samples_leaf': [0.04,0.06,0.08]
						 'max_features':[0.2, 0.4, 0.6, 0.8]}

grid_dt = GridSearchCV(estimator = dt,
											 param_grid = params_dt,
											 scoring = 'accuracy',
											 cv = 10,
											 n_jobs = -1)
											 
grid_dt.fit(X_train, y_train)

best_hyperparams = grid_dt.best_params_

best_CV_score = grid_dt.best_score_

best_model = grid_dt.best_estimator_

test_acc = best_model.score(X_test, y_test)
```

## Tuning an RF’s Hyperparameters

Hyperparameter tuning is computationally expensive, sometimes leads to very slight improvement.

```python
from sklearn.ensemble import RandomForestRegressor

SEED = 1

rf = RandomForestRegressor(random_state = SEED)

rf.get_params()

from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

params_rf = {'n_estimators':[300,400,500],
						 'max_depth': [4,6,8]
						 'min_samples_leaf': [0.1,0.2]
						 'max_features':['log2', 'sqrt']}

grid_rf = GridSearchCV(estimator = rf,
											 param_grid = params_rf,
											 cv=3
											 scoring = 'neg_mean_squared_error',
											 verbose = 1,
											 n_jobs = -1)
											 
grid_rf.fit(X_train, y_train)

best_hyperparams = grid_rf.best_params_

best_model = grid_rf.best_estimator_

y_pred = best_model.predict(X_test)

rmse_test = MSE(y_test, y_pred)**(1/2)
```

## Key Algorithms Covered

### Ensemble Methods
1. **Voting Classifier**: Combines predictions from multiple models using hard voting
2. **Bagging**: Bootstrap aggregation with same algorithm on different subsets
3. **Random Forests**: Bagging with decision trees and feature randomization
4. **AdaBoost**: Sequential boosting with weighted instances
5. **Gradient Boosting**: Sequential correction using residual errors
6. **Stochastic Gradient Boosting**: GB with random sampling for efficiency

## Skills Demonstrated

### Model Development
- Building classification and regression trees
- Implementing ensemble methods
- Feature importance analysis
- Model evaluation and validation

### Performance Optimization
- Bias-variance tradeoff understanding
- Cross-validation implementation
- Hyperparameter tuning with GridSearchCV
- Model selection strategies

### Advanced Techniques
- Bootstrap sampling and aggregation
- Out-of-bag evaluation
- Sequential boosting algorithms
- Stochastic optimization methods

## Important Concepts

### Decision Boundaries
- **Decision Region**: Feature space area where instances share a class label
- **Decision Boundary**: Surface separating different decision regions
- **Orthogonal Boundaries**: Classification boundaries perpendicular to axes

### Model Complexity
- **Underfitting**: High bias, low model flexibility
- **Overfitting**: High variance, excessive model complexity
- **Optimal Complexity**: Balance between bias and variance

### Ensemble Advantages
- Reduced variance through aggregation
- Improved generalization performance
- Robustness to outliers and noise
- Handling of non-linear relationships

## Key Takeaways

- **Tree-based models** don't require feature scaling
- **Decision trees** can capture non-linear relationships
- **Ensemble methods** combine multiple models for better performance
- **Random Forests** reduce overfitting through randomization
- **Boosting** corrects predecessor errors sequentially
- **Hyperparameter tuning** is essential for optimal performance
- **Cross-validation** provides unbiased performance estimates
- **Feature importance** helps understand model decisions