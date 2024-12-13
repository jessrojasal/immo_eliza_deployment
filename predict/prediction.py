import pickle
import pandas as pd

def predict(living_area, province, prosperity_index, extra_investment):
    """
    Predicts the price of a property based on the given features using a trained Random Forest Regression model.

    :param living_area: The living area of the property in square meters.
    :param province: The encoded province of the property.
    :param prosperity_index: The prosperity index of the municipality.
    :param extra_investment: The calculated extra investment based on the state and kitchen features.
    :return: A string representing the predicted price of the property in euros, formatted as '€xxx,xxx'.
    """

    # Load the trained model
    with open("model/random_forest_regressor_model.pkl", "rb") as pickle_in:
        model = pickle.load(pickle_in)

    # DataFrame with column names matching the features used during training
    features = pd.DataFrame(
        [[living_area, province, prosperity_index, extra_investment]],
        columns=["living_area", "province", "prosperity_index", "extra_investment"],
    )

    # Predict price
    price_transformed = model.predict(features)[0]

    # Reverse the square root transformation applied on target
    price = price_transformed**2

    # Round price to integer
    price = round(price)

    # Format price
    formatted_price = f"€{price:,.0f}"

    return formatted_price
