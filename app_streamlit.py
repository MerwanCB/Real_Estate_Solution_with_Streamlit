import pickle
import pandas as pd
import streamlit as st
import os

# --- Configuration ---
MODEL_PATH = "models/RE_Model.pkl"
TREE_IMAGE_PATH = "reports/figures/decision_tree.png"


# --- Helper Function to Load Model ---
@st.cache_resource  # Cache the model loading for efficiency
def load_model(path):
    """Loads the pickled model."""
    if not os.path.exists(path):
        st.error(f"Error: Model file not found at {path}")
        st.stop()
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Page Setup ---
# st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")
st.title("🏡 Real Estate Price Predictor")
st.write(
    """
This app predicts the estimated price of a property based on its features.
Enter the details of the property below to get a prediction.
The prediction is based on a Decision Tree model trained on historical data.
"""
)

# --- Feature Order (CRITICAL - Must match training) ---
# These are the features the model was trained on, in order.
EXPECTED_FEATURE_ORDER = [
    "year_sold",
    "property_tax",
    "insurance",
    "beds",
    "baths",
    "sqft",
    "year_built",
    "lot_size",
    "basement",
    "popular",
    "recession",
    "property_age",
    "property_type_Bunglow",
    "property_type_Condo",
]


# --- Input Form ---
with st.form("property_inputs"):
    st.subheader("Property Details")

    # Use columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        year_built = st.number_input(
            "Year Built", min_value=1800, max_value=2025, value=2000, step=1
        )
        sqft = st.number_input(
            "Square Footage (Living Area)", min_value=100, value=1500, step=50
        )
        beds = st.slider("Number of Bedrooms", min_value=0, max_value=10, value=3)
        baths = st.slider(
            "Number of Bathrooms", min_value=0.0, max_value=8.0, value=2.0, step=0.5
        )
        lot_size = st.number_input(
            "Lot Size (Square Feet)", min_value=0, value=5000, step=100
        )

    with col2:
        year_sold = st.number_input(
            "Year Sold", min_value=1990, max_value=2025, value=2015, step=1
        )
        property_tax = st.number_input(
            "Annual Property Tax ($)", min_value=0, value=300, step=10
        )
        insurance = st.number_input(
            "Annual Insurance Cost ($)", min_value=0, value=100, step=5
        )
        basement = st.selectbox("Has Basement?", options=["No", "Yes"])
        property_type = st.selectbox(
            "Property Type", options=["Condo", "Bunglow", "Other"]
        )  # Match model features

    with col3:
        popular = st.selectbox("Is in Popular Area?", options=["No", "Yes"])
        recession = st.selectbox("Was Sold During Recession?", options=["No", "Yes"])
        # property_age is calculated, not input

    # Submit button for the form
    submitted = st.form_submit_button("Predict Price")


# --- Prediction Logic ---
if submitted:
    # 1. Process Inputs into the required numerical format
    basement_numeric = 1 if basement == "Yes" else 0
    popular_numeric = 1 if popular == "Yes" else 0
    recession_numeric = 1 if recession == "Yes" else 0

    # Property Type Dummies (Only Condo and Bunglow are features)
    prop_bungalow = 1 if property_type == "Bunglow" else 0
    prop_condo = 1 if property_type == "Condo" else 0

    # Calculate derived feature: property_age
    property_age = year_sold - year_built
    if property_age < 0:
        st.warning("Warning: Year Sold is before Year Built. Property Age set to 0.")
        property_age = 0

    # 2. Create a dictionary matching EXPECTED_FEATURE_ORDER keys
    input_data = {
        "year_sold": year_sold,
        "property_tax": property_tax,
        "insurance": insurance,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "year_built": year_built,
        "lot_size": lot_size,
        "basement": basement_numeric,
        "popular": popular_numeric,
        "recession": recession_numeric,
        "property_age": property_age,
        "property_type_Bunglow": prop_bungalow,
        "property_type_Condo": prop_condo,
    }

    # 3. Create Pandas DataFrame in the correct order for prediction
    try:
        # Create DataFrame from the single input dictionary
        input_df = pd.DataFrame([input_data])
        # Reorder DataFrame columns to match the training order exactly
        input_df = input_df[EXPECTED_FEATURE_ORDER]
    except KeyError as e:
        st.error(
            f"Error preparing input data: Missing expected feature {e}. Check EXPECTED_FEATURE_ORDER."
        )
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while preparing data for prediction: {e}")
        st.stop()

    # 4. Make Prediction
    try:
        prediction = model.predict(input_df)
        predicted_price = prediction[0]  # Get the single prediction value
        print(predicted_price)
        # 5. Display Result
        st.subheader("Prediction Result:")
        st.metric(label="Estimated Property Price", value=f"${predicted_price:,.2f}")
        st.info(
            "Note: This is an estimated price based on the provided features and the trained model. Market conditions can vary."
        )

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- Additional Information / Visualization ---
st.write("---")  # Separator
st.subheader("About the Model")
st.write(
    """
The prediction uses a Decision Tree Regressor model.
Decision trees work by splitting the data based on feature values to arrive at a price estimate.
Below is a visualization of the trained decision tree structure (may be large).
"""
)

# Display the saved decision tree image
if os.path.exists(TREE_IMAGE_PATH):
    st.image(TREE_IMAGE_PATH, caption="Trained Decision Tree Structure")
else:
    st.warning(f"Could not find the decision tree image at: {TREE_IMAGE_PATH}")
