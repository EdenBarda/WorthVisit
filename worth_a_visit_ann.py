import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
import time

start_time = time.time()

# Load the dataset
data = pd.read_csv("perth/all_perth_310121.csv")

# remove rows with NULL values
data = data.dropna(subset=['GARAGE', 'BUILD_YEAR'])

# add unique num to each suburb
suburb_nums = pd.factorize(data['SUBURB'])[0]
data.insert(data.columns.get_loc('SUBURB') + 1, 'SUBURB_NUM', suburb_nums)

# split 'DATE_SOLD' column
data[['MONTH_SOLD', 'YEAR_SOLD']] = data['DATE_SOLD'].str.split('-', expand=True)

# make all columns names lowercase
data.columns = data.columns.str.lower()

# Select the features and target variable
selected_features = ["bedrooms", "bathrooms", "garage", "land_area", "floor_area", "build_year",
                     "cbd_dist", "nearest_stn_dist", "month_sold", "year_sold", "latitude", "longitude",
                     "nearest_sch_dist"]
data['in_budget'] = [0 if 500000 <= x <= 600000 else 1 for x in data['price']]
X = data[selected_features]
y = data["in_budget"]

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42, test_size=0.01)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Create and train the ANN model
ann_model = MLPRegressor(max_iter=500, random_state=42)

"""best for 6: (62, 38, 13, 12, 6, 10)
best for 5: (42, 35, 13, 12, 6), 0.83748
best for 4: (62, 38, 13, 12), 0.83225"""

# hidden_layer = [(42, 35, 13, 12, i) for i in range(2, 12)]
grid = {
    "hidden_layer_sizes": [(42, 35, 13, 12, 6)],
    "batch_size": [600],
    "solver": ["adam"],
    "learning_rate": ["adaptive"],
    "learning_rate_init": [0.02],
    "activation": ["relu"],
    "early_stopping": [False],
}

grid_search = GridSearchCV(ann_model, grid, scoring="roc_auc")
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Predict if the house prices is in the budget
y_predict = best_model.predict(X_test_scaled)

# Calculate evaluation metrics
roc = roc_auc_score(y_test, y_predict)

training_time = time.time() - start_time

# prints information
print(f"Training Time: {training_time:.2f} seconds")
print(f"Test roc: {roc:.5f}")
print(best_model)

# Plot loss curve (training curve)
plt.plot(best_model.loss_curve_)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.title("Convergence Curve")
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()
