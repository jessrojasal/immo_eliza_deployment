from preprocess import (
    DataAugmentation,
    DataImputation,
    FeatureCombination,
    RemoveOutliers,
    FilterAndDrop,
)

from model import SplitData, ModelRandomForestRegressor

# Preprocess
# Data Augmentation: Add new features
augmenter = DataAugmentation(base_path="data")
data_augmented = augmenter.add_new_features()

# Handle missing values with imputation
imputer = DataImputation(data_augmented)
data_no_missing_values = imputer.impute_missing_values()

# Create new columns by performing feature combination
feature_combiner = FeatureCombination(data_no_missing_values)
combined_data = feature_combiner.combine_features()

# Remove outliers
data_cleaner = RemoveOutliers(combined_data)
cleaned_data = data_cleaner.remove_outliers()

# Filter observations and drop columns
filter_obj = FilterAndDrop(cleaned_data)
data = filter_obj.filter_drop()

# Model
# Split database
splitter = SplitData(data, target_column="price", categorical_columns=["province"])
X_train, X_test, y_train, y_test = splitter.split()
splitter.save_encoders()

# Train Random Forest Regressor model
rf_regressor = ModelRandomForestRegressor(
    random_state=100,
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=17,
    max_leaf_nodes=100,
    max_depth=80,
)
rf_regressor.train(X_train, y_train)

# Save the trained model
rf_regressor.save_model(filename="random_forest_regressor_model.pkl")

# Evaluate the model
rf_metrics = rf_regressor.evaluate(X_train, X_test, y_train, y_test)
