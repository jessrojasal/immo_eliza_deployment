import streamlit as st
import pandas as pd
from predict.prediction import predict
from preprocessing.cleaning_data import preprocess

# Load the DataFrame from the file in the 'model' folder
df = pd.read_csv("model/features_data.csv")

# Page setup
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# Main page header
st.markdown(
    """
    <h1 style="font-size: 2.5rem; text-align: center;">Real Estate Price Predictor</h1>
    <p style="text-align: center;">Predict Belgian property prices based on locality and building characteristics</p>
    """,
    unsafe_allow_html=True,
)

# CSS for select box color
st.markdown(
    """
    <style>
    /* Apply custom text color to all sidebar elements */
    section[data-testid="stSidebar"] .st-expander, 
    section[data-testid="stSidebar"] .st-expander-content, 
    section[data-testid="stSidebar"] {
        color: #e7e1d2 !important;
    }

    /* Change the color of the collapse button when the sidebar is expanded */
    section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] {
        color: #e7e1d2 !important;
    }

    /* Change the color of the collapse button when the sidebar is collapsed */
    section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] {
        color: #000000 !important;
    }

    /* Change text color when the option is selected */
    .st-bm.st-ak {
        color: #e7e1d2 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar content with sections
with st.sidebar:
    # About this App section
    with st.expander("About this App"):
        st.markdown(
            """
        This app is designed to help users predict Belgian property prices by providing building characteristics and location information. 
        It uses a Random Forest Regression model trained on data collected from Immoweb, Belgium's largest real estate platform.
        """
        )

    # Instructions section
    with st.expander("Instructions"):
        st.markdown(
            """
            1. **Select the locality and property features.**  
            Start by choosing the province and municipality where the property is located.

            2. **Provide property details.**  
            Specify the living area (m²), the state of the building, and whether the kitchen is fully equipped.

            3. **Click 'Predict Property Price.'**  
            The app will process your input and predict the property's price.
            """
        )

# Header 2:
st.markdown(
    """
    <h3 style="font-size: 1.8rem;">Select the desired property characteristics and locality</h3>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

# Column 1
with col1:
    st.text("Locality")

    # Unique provinces from the DataFrame
    provinces = df["province"].unique()
    province = st.selectbox("Select the province:", provinces)

    # Filter municipalities based on the selected province
    municipalities = df[df["province"] == province]["municipality"].unique()
    municipality = st.selectbox("Select the municipality:", municipalities)

# Column 2
with col2:
    st.text("Property characteristics")
    livable_space = st.slider("Livable space (m²):", 10, 500, 20)
    state = st.selectbox(
        "State of the building:",
        [
            "Just renovated",
            "As new",
            "Good",
            "To renovate",
            "To be done up",
            "To restore",
        ],
    )
    kitchen = st.selectbox("Fully equipped kitchen:", ["Yes", "No"])

if st.button("Predict Property Price"):
    # Prepare user input for preprocessing
    user_input = {
        "living_area": livable_space,
        "province": province,
        "municipality": municipality,
        "state": state,
        "kitchen": kitchen,
    }

    # Apply the preprocessing function
    preprocessed_input = preprocess(user_input)

    # Extract preprocessed features for the prediction function
    living_area = preprocessed_input["living_area"]
    province = preprocessed_input["province"]
    prosperity_index = preprocessed_input.get("prosperity_index", 0)
    extra_investment = preprocessed_input.get("extra_investment", 0)

    # Predict
    result = predict(living_area, province, prosperity_index, extra_investment)
    # Result message
    st.markdown(
        f"""
        <div style="background-color:#ded1ff; padding:5px; border-radius:10px; margin-top:20px;">
            <h4 style="color:#431f79; text-align:center;">Predicted property price: {result}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    """
    **App developed by: Jess Rojas-Alvarado**\n
    Connect with me: [GitHub](https://github.com/jessrojasal)
    
    [BeCode](https://becode.org/) learning project\n
    AI & Data Science Bootcamp  
    """
)
