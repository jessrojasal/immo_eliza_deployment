import pandas as pd
import pickle

def preprocess(user_input):
    """
    Preprocesses the user input data by encoding categorical variables and
    calculating additional features required for prediction.

    :param user_input: A dictionary containing the user input with the following keys:
       - "living_area" (str): The living area of the property (m²).
       - "province" (str): The province of the property.
       - "municipality" (str): The municipality of the property.
       - "state" (str): The condition of the property.
       - "kitchen" (str): Whether the property has a fully equipped kitchen ("Yes"/"No").

    :return: A dictionary containing the processed features for prediction:
       - "living_area" (int): The living area of the property (m²).
       - "province" (int): The encoded province value.
       - "prosperity_index" (Optional[int]): The prosperity index of the municipality, or None if not found.
       - "extra_investment" (int): The sum of encoded values for the state and kitchen.
    """

    # Encode province
    encoder_path = "model/province_encoder.pkl"
    with open(encoder_path, "rb") as file:
        province_encoder = pickle.load(file)

    province_encoded = province_encoder.transform([user_input["province"]])[0]

    # Encode 'state' and 'kitchen' and calculate extra_investment
    state_mapping = {
        "Just renovated": 3,
        "As new": 4,
        "Good": 3,
        "To renovate": 2,
        "To be done up": 2,
        "To restore": 1,
    }
    state_encoded = state_mapping.get(user_input["state"], 0)

    kitchen_encoded = 1 if user_input["kitchen"] == "Yes" else 0

    extra_investment = state_encoded + kitchen_encoded

    # Extract prosperity_index from the 'municipality' using the features_data.csv file
    df = pd.read_csv("model/features_data.csv")
    municipality_data = df[df["municipality"] == user_input["municipality"]]

    if not municipality_data.empty:
        prosperity_index = municipality_data["prosperity_index"].values[0]
    else:
        prosperity_index = None

    # Return dictionary with processed features
    processed_data = {
        "living_area": user_input["living_area"],
        "province": province_encoded,
        "prosperity_index": prosperity_index,
        "extra_investment": extra_investment,
    }

    return processed_data
