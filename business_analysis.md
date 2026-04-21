###Business Case Analysis: Promotion Effectiveness
***B1. Problem Formulation***
**(a) Machine Learning Problem Formulation**
**Target Variable:** items_sold (Continuous numerical value).

**Candidate Input Features:** store_id, store_size, location_type, promotion_type, is_weekend, is_festival, competition_density, and engineered date features (month, year, is_month_end).

**ML Problem Type:** Supervised Learning — Specifically, Regression.

**Justification:** We are trying to predict a continuous numerical output (sales volume) based on historical labeled data (past transactions and their corresponding sales numbers).

**(b) Items Sold vs. Total Sales Revenue**
Using items sold (volume) is a much more reliable target variable because total sales revenue is directly confounded by the promotions themselves. If we offer a "Flat Discount," the price per item drops. Even if the promotion successfully drives massive customer traffic and moves huge inventory, the total revenue might look flat or even lower due to the discounted price.

**Broader Principle:** This illustrates the principle of avoiding target variable confounding/endogeneity. The target variable must represent the true, unskewed objective of the business (demand/volume) rather than a metric directly manipulated by the features we are testing (price drops).

**(c) Alternative Modeling Strategy**
A single global model assumes that a promotion behaves the exact same way universally, ignoring critical interaction effects.
**Alternative Strategy:** Combine Unsupervised and Supervised learning. First, use a clustering algorithm (like K-Means) to segment the 50 stores into 3-4 groups based on attributes like size, location type, and average footfall. Then, train separate regression models for each cluster.
**Justification:** A "Buy-One-Get-One" offer might cause a stampede in a highly competitive urban mall but go unnoticed in a rural standalone store. Clustered models allow us to capture these localized interaction effects without creating a complex, over-parameterized global model.

***B2. Data and EDA Strategy***
**(a) Data Joining and Grain**
**Joining Strategy:** We will start with a base Calendar table. We will use Left Joins to bring in the Store Attributes (using store_id), Promotion Details (using date and store_id), and Transactions.

**Aggregations:** The raw transaction data is likely at the individual receipt/customer level. We must aggregate this by summing the items sold per store, per month.

**Final Grain:** The grain of the final modeling dataset will be Store-Month (one row = the performance of a specific store during a specific month).

**(b) EDA Strategy Prior to Modeling**
Distribution of Target Variable (Histogram of items_sold): To check for extreme positive skewness or outliers. If highly skewed, we may need to apply a log transformation to the target variable.

**Correlation Heatmap:** To check for multicollinearity among numerical features (e.g., checking if store_size perfectly correlates with competition_density). We would drop redundant features to simplify the model.

**Boxplots of Items Sold by Promotion Type:** To visually establish the baseline historical effectiveness of each promotion. This helps validate if the model's future predictions make logical sense.

**Time Series Line Chart of Sales over Time:** To visually identify seasonality (e.g., huge spikes in December). This confirms the necessity of including temporal features like is_festival or month.

**(c) Addressing Class Imbalance (80% No Promotion)**
If 80% of the data features no promotion, the model will become heavily biased toward predicting the "baseline" sales volume and will struggle to learn the nuanced differences between the 5 specific promotion types.
**Steps to address:** We can downsample the "No Promotion" periods to artificially balance the dataset during training, or assign higher sample weights to the rows where promotions were active so the model's loss function penalizes errors on promotional days more heavily.

***B3. Model Evaluation and Deployment***
**(a) Train-Test Split and Metrics**
Split Setup: Sort the 3 years of data chronologically. Use the first 28 months for training and the most recent 8 months for testing (a temporal split).

**Why random is inappropriate:** A random split shuffles time, meaning the model would be trained on "future" data to predict "past" events, causing data leakage and completely ruining its real-world forecasting validity.

**Evaluation Metrics: * MAE (Mean Absolute Error):** Highly interpretable for the business (e.g., "Our forecast is typically off by exactly 150 items").

**RMSE (Root Mean Squared Error):** Heavier penalty for massive misses. Important if under-stocking items by a large margin causes severe logistical failures.

**(b) Investigating Different Monthly Recommendations**
To explain why Store 12 gets Loyalty Points in December and Flat Discounts in March, we would look at the model's Feature Importances and interaction effects. December is likely flagged by the month or is_festival feature. We would communicate to marketing that the model has learned seasonal consumer behavior: during high-spend holiday seasons, customers value racking up loyalty points for future use. Conversely, in the slower month of March, direct Flat Discounts are required to artificially stimulate demand.

**(c) End-to-End Deployment Process**
Saving the Model: Serialize the entire Scikit-Learn Pipeline (which includes the ColumnTransformer for scaling/encoding AND the trained Random Forest model) into a file using the pickle library.

**Feeding New Data:** At the start of the month, the new store/calendar parameters are formatted into a dataframe and passed through the loaded 'pickle' pipeline using the '.predict()' method. No retraining is required.

**Monitoring:** We establish a feedback loop to monitor "Concept Drift." We track the MAE of our predictions against the actual items sold at the end of each month. If the MAE exceeds a predefined business threshold (e.g., error rate jumps by 15%), it triggers an alert to data scientists that consumer behavior has shifted and the model must be retrained on fresh data.
