
# Real Estate Price Predictor

This project builds a simple web application that estimates the price of a house based on its features, like size, number of rooms, and location. It uses machine learning to make these predictions.

[Visit the live app here](https://realestatesolutionwithapp-26hxyr5qgdtbingszfmwgc.streamlit.app/)

This application provides an estimated market price for a property based on its characteristics. It utilizes a machine learning model trained on historical real estate data and is presented through a user-friendly Streamlit interface.

## Features
- User-friendly web interface powered by Streamlit.
- Input form to enter property details such as size, age, number of rooms, location type, etc.
- Real-time prediction of the estimated property price using a trained Decision Tree model.
- Visualization of the underlying Decision Tree model structure used for prediction.

## Dataset
The application is trained on a dataset containing historical property sales data (`final.csv`). Key features used for prediction include:
- Year Built
- Square Footage (Living Area)
- Number of Bedrooms
- Number of Bathrooms
- Garage Size (Number of Cars)
- Lot Size
- Presence of a Pool
- Property Type (e.g., Townhouse, Condo, Bunglow)
- Location Type (e.g., Urban, Suburban, Rural)

## Technologies Used
- **Streamlit**: For building and serving the interactive web application.
- **Scikit-learn**: For model training (Linear Regression, Decision Tree, Random Forest), evaluation (MAE), and data splitting.
- **Pandas**: For data loading, manipulation, and preparation.
- **NumPy**: For numerical operations.
- **Matplotlib**: Used indirectly via Scikit-learn for plotting the decision tree visualization.
- **Pickle**: For saving and loading the trained machine learning model.

## Model
The predictive model currently implemented in the Streamlit app is a **Decision Tree Regressor**. During development (see `main.py`), Linear Regression and Random Forest models were also trained and evaluated using Mean Absolute Error (MAE). The Decision Tree was chosen for this version of the application and saved for deployment. Preprocessing involved separating features and target variables.


## Future Enhancements
*   Implement feature scaling and hyperparameter tuning for potentially improved model accuracy.
*   Explore and potentially deploy more advanced regression models (e.g., Random Forest, Gradient Boosting).
*   Add a feature importance plot to the Streamlit app to show which factors most influence the price.
*   Allow users to upload a CSV file for batch predictions.
*   Incorporate more detailed model evaluation metrics within the app interface.
*   Integrate explainability tools (e.g., SHAP) to provide insights into individual predictions.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MerwanCB/Real_Estate_Solution_with_Streamlit.git
    cd Real_Estate_Solution_with_Streamlit
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows use: venv\Scripts\activate
    # On macOS/Linux use: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure Data and Model are Present:**
    *   Place your dataset `final.csv` inside the `data/raw/` directory.
    *   Run the main pipeline (`python main.py`) to train the model and generate the necessary `models/RE_Model.pkl` file and the `reports/figures/decision_tree.png` image. *(If these files are already included in the repository, this step might be optional for just running the app).*

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app_streamlit.py
    ```

---
#### Thank you for checking out the Real Estate Price Predictor! Feel free to contribute or share feedback via the [GitHub repository](https://github.com/MerwanCB/Real_Estate_Solution_with_Streamlit.git).