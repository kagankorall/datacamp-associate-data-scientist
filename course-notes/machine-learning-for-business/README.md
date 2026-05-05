# Machine Learning for Business

## Course Overview
A business-focused course exploring how machine learning creates value in organizations. It covers machine learning use cases across the data needs pyramid, the differences between inference and prediction models, supervised and unsupervised learning, scoping business requirements, evaluating model performance, and best practices for managing machine learning projects from design to production.

## Key Topics Covered

### 1. Machine Learning and Data Use Cases
- Machine learning and the data pyramid
- Terminology clarification (AI, ML, deep learning, data science)
- Ordering data pyramid needs
- Matching tasks within the data pyramid
- Machine learning principles
- Modeling types
- Identifying supervised and unsupervised cases
- Job roles, tools and technologies
- Job role responsibilities
- Matching data projects with job roles
- Team structure types

### 2. Machine Learning Types
- Prediction vs. inference dilemma
- Inference and prediction differences
- Identifying inference vs. prediction use cases
- Inference (causal) models
- Experiments and causal models
- Identifying non-actionable variables
- Prediction models (supervised learning)
- Supervised modeling principles
- Identifying classification and regression models
- Prediction models (unsupervised learning)
- Unsupervised modeling use cases
- Classification, regression or unsupervised models

### 3. Business Requirements and Model Design
- Business requirements
- Identifying situation, opportunity and action
- Identifying successful experiments
- Model training
- Model training process
- Training, validation and test sets
- Model performance measurement
- Poor performance examples
- Identifying performance metrics
- Machine learning risks
- Fixing non-performing models
- Non-actionable models
- Identifying actionable recommendations

### 4. Managing Machine Learning Projects
- Common machine learning mistakes
- Identifying machine learning mistakes
- Data needs pyramid
- Matching ML mistakes by their types
- Communication management
- Business communication focus
- Market testing
- Machine learning in production
- Production systems
- Production systems ML use cases
- ML in production launch
- Wrap-up

## Course Notes

# Machine Learning and Data Use Cases

## Machine Learning and the Data Pyramid

Machine learning is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without explicit instructions, relying on patterns and inference instead.

The **data needs pyramid** describes the layers of work required before machine learning can deliver value:

1. **Collection** — gathering raw data from sources (transactions, sensors, logs).
2. **Storage** — reliable databases and data warehouses.
3. **Preparation** — cleaning, transforming and organizing data (ETL).
4. **Analysis / BI** — dashboards, reports, descriptive analytics.
5. **Experimentation** — A/B tests and causal experiments.
6. **Machine learning / AI** — predictive and prescriptive models.

Each layer is a prerequisite for the next. Jumping to ML without a strong foundation in collection, storage and preparation is a common source of failure.

## Terminology Clarification

- **Artificial Intelligence (AI):** broad field of building systems that simulate human intelligence.
- **Machine Learning (ML):** subset of AI focused on algorithms that learn from data.
- **Deep Learning:** subset of ML using multi-layer neural networks for complex tasks (vision, NLP).
- **Data Science:** discipline combining statistics, programming and domain knowledge to extract insights from data.

## Machine Learning Principles

- ML models learn patterns from historical data and apply them to new, unseen data.
- The quality of the model depends heavily on the quality and quantity of training data.
- Models do not understand causation by default — they capture correlations.
- A model is only as good as the business question it is designed to answer.

## Modeling Types

There are three core types of machine learning:

- **Supervised learning** — labeled data; the model learns the mapping from inputs to a known target.
- **Unsupervised learning** — unlabeled data; the model uncovers structure or groupings.
- **Reinforcement learning** — an agent learns by interacting with an environment and receiving rewards.

## Job Roles, Tools and Technologies

Typical roles in a data organization:

- **Data Analyst / BI Analyst** — descriptive analytics, dashboards, SQL, Tableau, Power BI.
- **Data Engineer** — pipelines, storage, ETL, Spark, Airflow, cloud platforms.
- **Data Scientist** — modeling, statistics, experimentation, Python/R, scikit-learn.
- **Machine Learning Engineer** — productionizing models, MLOps, deployment, monitoring.

## Team Structure Types

Three common structures for organizing data teams:

- **Centralized** — a single data team serves the whole company. Strong standards, but risks being detached from business units.
- **Decentralized** — data professionals are embedded inside business units. Strong domain alignment, but risks duplication and inconsistency.
- **Hybrid (center of excellence)** — central team sets standards and tooling while embedded members work directly with business units. Balances consistency with proximity to business needs.

# Machine Learning Types

## Prediction vs. Inference Dilemma

Two fundamental purposes for building a model:

- **Prediction (supervised ML):** focus on predicting an outcome accurately on new data. The model is judged by predictive performance, not interpretability.
- **Inference (causal models):** focus on understanding how features affect the outcome. The model is judged by the validity and interpretability of its coefficients.

| Aspect | Prediction | Inference |
| --- | --- | --- |
| Goal | Estimate Y for new X | Understand effect of X on Y |
| Priority | Accuracy | Interpretability, causality |
| Typical methods | Random forests, boosting, neural nets | Linear/logistic regression, A/B tests |
| Output | Predicted values | Coefficients, p-values |

## Inference (Causal) Models

Causal inference asks **"what would happen if we changed X?"**.

- Run controlled **experiments (A/B tests)** whenever possible — random assignment isolates the causal effect.
- When experiments are not possible, use observational techniques (regression with controls, instrumental variables, difference-in-differences).
- **Non-actionable variables** (such as customer age or gender) cannot be changed by the business and should not drive intervention decisions, even when correlated with outcomes.

## Prediction Models (Supervised Learning)

Supervised learning is used when historical data has labeled outcomes.

- **Classification** — the target is categorical (e.g., churn vs. no-churn, fraud vs. legitimate).
- **Regression** — the target is continuous (e.g., revenue, demand, price).

### Supervised Modeling Principles

- Split the data into **training, validation and test** sets.
- Train on the training set, tune on the validation set, and report final performance on the held-out test set.
- Beware of data leakage — features that are not available at prediction time must be excluded.

## Prediction Models (Unsupervised Learning)

Unsupervised learning is used when there are no labels — the goal is to discover structure.

Common business use cases:

- **Customer segmentation** with clustering (k-means, hierarchical).
- **Anomaly detection** for fraud or system failures.
- **Recommendation systems** based on similarity.
- **Dimensionality reduction** (PCA) to simplify high-dimensional data.

# Business Requirements and Model Design

## Business Requirements

Every machine learning project should start by clearly defining the business problem. A useful framework is **Situation – Opportunity – Action**:

- **Situation:** what is currently happening? (e.g., customer churn is rising)
- **Opportunity:** what could change if the problem is solved? (retain X% more customers)
- **Action:** what decision will the model enable? (which customers to target with retention offers)

If the model output cannot be tied to a concrete business action, the project will not deliver value.

## Identifying Successful Experiments

A successful ML experiment typically:

- Targets a clearly measurable business KPI.
- Has a control group for comparison.
- Runs long enough to reach statistical significance.
- Produces an actionable recommendation, not just a number.

## Model Training Process

The standard workflow:

1. **Data preparation** — cleaning, feature engineering, splitting.
2. **Training** — fit the model on the training set.
3. **Validation** — tune hyperparameters using the validation set.
4. **Testing** — evaluate the final model on the held-out test set.
5. **Deployment** — release the model to production.

### Training, Validation and Test Sets

- **Training set** (~60–70%) — used to fit model parameters.
- **Validation set** (~15–20%) — used to compare models and tune hyperparameters.
- **Test set** (~15–20%) — used **only once** for the final, unbiased estimate of performance.

## Model Performance Measurement

Choosing the right metric depends on the problem:

### Classification metrics
- **Accuracy** — fraction of correct predictions. Misleading on imbalanced classes.
- **Precision** — of those predicted positive, how many actually are. (Minimize false positives.)
- **Recall** — of all actual positives, how many were caught. (Minimize false negatives.)
- **F1 score** — harmonic mean of precision and recall.
- **ROC AUC** — overall ranking quality across thresholds.

### Regression metrics
- **MAE** (mean absolute error) — average absolute deviation; robust to outliers.
- **MSE / RMSE** — penalizes larger errors more heavily.
- **R²** — proportion of variance explained.

### Poor Performance Examples

- Reporting accuracy on a 99/1 imbalanced fraud problem (a model predicting "not fraud" always scores 99%).
- Optimizing recall for a costly intervention without considering precision.
- Comparing models trained on different time periods or feature sets.

## Machine Learning Risks

Common risks that hurt ML projects:

- **Overfitting** — model memorizes training data and fails on new data.
- **Underfitting** — model is too simple to capture real patterns.
- **Concept drift** — the relationship between features and target changes over time.
- **Bias in training data** — model inherits and amplifies historical biases.
- **Non-actionable predictions** — accurate output that no team can act on.

### Fixing Non-Performing Models

- Gather **more or better data** before tuning algorithms.
- Add or engineer **more relevant features**.
- Try **different model families** (linear, tree-based, neural).
- **Tune hyperparameters** systematically (grid or random search).
- Re-examine the **business problem** — sometimes the wrong question is being asked.

# Managing Machine Learning Projects

## Common Machine Learning Mistakes

Mistakes can be grouped by where in the data needs pyramid they occur:

- **Data collection mistakes** — missing critical signals, biased sampling.
- **Data storage / engineering mistakes** — broken pipelines, untrustworthy data.
- **Analysis mistakes** — drawing conclusions from incomplete or unrepresentative data.
- **Modeling mistakes** — choosing the wrong target, leaking labels, ignoring class imbalance.
- **Production mistakes** — deploying models without monitoring, no retraining strategy.

## Communication Management

Successful ML projects depend on tight communication between business and technical teams.

### Business Communication Focus

When communicating with business stakeholders:

- Lead with the **business impact**, not the algorithm.
- Translate model metrics into **business KPIs** (revenue, retention, cost saved).
- Be explicit about **assumptions, limitations and risks**.
- Propose **concrete actions** the business can take based on the output.

Avoid technical jargon (precision/recall, AUC, regularization) without context — frame them in terms of decisions and trade-offs.

## Market Testing

Before scaling a model across the entire business:

- Run a **pilot / A/B test** on a limited segment.
- Compare against a control group with the existing process.
- Measure **business KPIs**, not just model metrics.
- Iterate on the model and the surrounding process before full rollout.

## Machine Learning in Production

Going to production is where most ML value is realized — and where most projects fail.

### Production Systems

A production ML system requires:

- **Reliable data pipelines** that feed the model in real time or batch.
- **Model serving infrastructure** (APIs, batch jobs, embedded inference).
- **Monitoring** for input drift, prediction drift and performance degradation.
- **Retraining pipelines** to refresh the model as new data arrives.
- **Rollback strategy** in case of failures.

### Production Systems ML Use Cases

- **Real-time scoring** — fraud detection, recommendation, ad targeting.
- **Batch scoring** — churn lists, demand forecasts, lead scoring.
- **Embedded models** — on-device inference (mobile, IoT).

### ML in Production Launch

A safe launch typically follows these stages:

1. **Shadow mode** — model runs in parallel with the existing process, predictions are logged but not used.
2. **Canary release** — model serves a small fraction of traffic.
3. **Gradual rollout** — traffic is increased as confidence grows.
4. **Full deployment** — model fully replaces or augments the previous process.
5. **Continuous monitoring** — performance and drift are tracked indefinitely.

## Skills Demonstrated

### Strategic Thinking
- Translating business problems into ML problems
- Identifying when ML is (and isn't) the right tool
- Mapping projects to the data needs pyramid

### Project Management
- Scoping ML initiatives with the Situation–Opportunity–Action framework
- Coordinating data, engineering and business stakeholders
- Designing pilots and market tests

### Model Evaluation
- Choosing appropriate performance metrics
- Diagnosing overfitting, underfitting and drift
- Distinguishing actionable from non-actionable predictions

### Production Awareness
- Understanding deployment patterns (shadow, canary, full rollout)
- Planning monitoring and retraining strategies
- Managing risks across the ML lifecycle

## Key Takeaways

- **ML is built on a foundation** — collection, storage and preparation must be solid before modeling adds value.
- **Inference and prediction answer different questions** — choose the modeling approach to match the business goal.
- **Supervised vs. unsupervised** depends on whether labeled outcomes are available.
- **Business requirements come first** — always tie a model to a concrete decision and KPI.
- **Pick metrics that match the problem** — accuracy is rarely enough on its own.
- **Communication is half the job** — frame results in business language, not algorithms.
- **Most ML projects fail in production**, not in training — design for monitoring, drift and retraining from day one.
- **Non-actionable predictions create no value**, regardless of how accurate the model is.
