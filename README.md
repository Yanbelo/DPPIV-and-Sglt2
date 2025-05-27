# DPPIV-and-Sglt2
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import joblib
import matplotlib.pyplot as plt
from rdkit import RDLogger
from rdkit.Chem import Descriptors
# Corrected import statement
from rdkit.DataStructs import TanimotoSimilarity, DiceSimilarity # Import the specific functions needed
from sklearn.metrics.pairwise import cosine_similarity

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Load the trained stacked models
try:
    dppiv_stacked_model = joblib.load("DPPIV_classifier.pkl")
    sglt2_stacked_model = joblib.load("sglt2_stacked_model.pkl")
    model_loaded = True
except FileNotFoundError:
    st.error("Error: Model files 'DPPIV_classifier.pkl' or 'sglt2_stacked_model.pkl' not found.")
    st.warning("Please ensure the models are trained and saved before running the Streamlit app.")
    model_loaded = False

# Function to standardize SMILES (ensure this matches your training code)
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        Chem.rdmolops.RemoveStereochemistry(mol)
        mol = rdMolStandardize.ChargeParent(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    return None

# Function to predict activity
def predict_activity(smiles, dppiv_model, sglt2_model):
    """
    Predicts the activity (DPP-IV and SGLT2 inhibition) for a given SMILES string
    using pre-trained stacked models.

    Args:
      smiles (str): The SMILES string of the molecule.
      dppiv_model: The loaded trained DPP-IV stacked model.
      sglt2_model: The loaded trained SGLT2 stacked model.

    Returns:
      dict: A dictionary containing the predicted labels and probabilities for
            DPP-IV and SGLT2, or None if processing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"SMILES": smiles, "Error": "Could not parse SMILES"}

    # Standardize SMILES
    standardized_smiles = standardize_smiles(smiles)
    # Fix typo in variable name
    if standardized_smiles is None:
        return {"SMILES": smiles, "Error": "Could not standardize SMILES"}

    # Generate Morgan fingerprint
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(standardized_smiles), 2, nBits=2048)
        X_unknown = np.array([np.array(fp)], dtype=np.int32)
    except Exception as e:
        return {"SMILES": smiles, "Error": f"Could not generate fingerprint: {e}"}

    # Predict using the DPP-IV model
    try:
        dppiv_prediction = dppiv_model.predict(X_unknown)[0]
        dppiv_probability = dppiv_model.predict_proba(X_unknown)[0][1]
    except Exception as e:
         dppiv_prediction = -1 # Indicate error
         dppiv_probability = -1.0
         st.warning(f"Error predicting with DPP-IV model for SMILES {smiles}: {e}")


    # Predict using the SGLT2 model
    try:
        sglt2_prediction = sglt2_model.predict(X_unknown)[0]
        sglt2_probability = sglt2_model.predict_proba(X_unknown)[0][1]
    except Exception as e:
        sglt2_prediction = -1 # Indicate error
        sglt2_probability = -1.0
        st.warning(f"Error predicting with SGLT2 model for SMILES {smiles}: {e}")


    return {
        'SMILES': smiles,
        'Standardized_SMILES': standardized_smiles,
        'DPPIV_Prediction': int(dppiv_prediction) if dppiv_prediction != -1 else -1,
        'DPPIV_Probability': float(dppiv_probability) if dppiv_probability != -1.0 else -1.0,
        'SGLT2_Prediction': int(sglt2_prediction) if sglt2_prediction != -1 else -1,
        'SGLT2_Probability': float(sglt2_probability) if sglt2_probability != -1.0 else -1.0
    }

# Streamlit UI
st.set_page_config(page_title="Drug Candidate Prediction")

st.title("DPP-IV and SGLT2 Inhibitor Prediction")

st.write("This application predicts the potential activity of a given molecule (represented by its SMILES string) against DPP-IV and SGLT2 using pre-trained stacked machine learning models.")

# Input Method
input_method = st.radio("Select Input Method:", ("Enter SMILES Manually", "Upload CSV/Excel File"))

predicted_results_df = pd.DataFrame() # Initialize an empty DataFrame

if input_method == "Enter SMILES Manually":
    st.subheader("Enter SMILES String")
    smiles_input = st.text_area("Enter one SMILES string per line:", height=150)
    smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]

    if st.button("Predict"):
        if not model_loaded:
            st.error("Models are not loaded. Please check the notebook execution or file paths.")
        elif not smiles_list:
            st.warning("Please enter at least one SMILES string.")
        else:
            st.subheader("Prediction Results")
            with st.spinner("Predicting activity..."):
                prediction_results_list = []
                for smiles in smiles_list:
                    result = predict_activity(smiles, dppiv_stacked_model, sglt2_stacked_model)
                    if result:
                        prediction_results_list.append(result)

                predicted_results_df = pd.DataFrame(prediction_results_list)

                if not predicted_results_df.empty:
                    st.dataframe(predicted_results_df)

                    st.subheader("Predicted Dual Inhibitors (Active for both DPP-IV and SGLT2)")
                    dual_active_df = predicted_results_df[
                        (predicted_results_df['DPPIV_Prediction'] == 1) &
                        (predicted_results_df['SGLT2_Prediction'] == 1)
                    ]
                    if not dual_active_df.empty:
                        st.dataframe(dual_active_df)
                        # Optional: Display structures of dual inhibitors (requires RDKit drawing)
                        # from stmol import showmol # You might need to install stmol
                        # for i, row in dual_active_df.iterrows():
                        #     mol = Chem.MolFromSmiles(row['SMILES'])
                        #     if mol:
                        #         st.write(f"**SMILES:** {row['SMILES']}")
                        #         showmol(mol) # Or use RDKit's drawing function if stmol is not used
                    else:
                        st.info("No compounds in the input list were predicted to be active for both targets.")

                else:
                    st.warning("No valid prediction results were generated.")

elif input_method == "Upload CSV/Excel File":
    st.subheader("Upload File (.csv or .xlsx)")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Try reading with different separators
                try:
                    input_df = pd.read_csv(uploaded_file)
                except Exception:
                     uploaded_file.seek(0) # Reset file pointer
                     input_df = pd.read_csv(uploaded_file, sep=';') # Try semicolon

            elif uploaded_file.name.endswith('.xlsx'):
                input_df = pd.read_excel(uploaded_file)

            st.write("File uploaded successfully.")
            st.write("Columns in the uploaded file:", input_df.columns.tolist())

            smiles_column = st.selectbox("Select the column containing SMILES strings:", input_df.columns.tolist())

            if st.button("Predict from File"):
                 if not model_loaded:
                    st.error("Models are not loaded. Please check the notebook execution or file paths.")
                 elif smiles_column not in input_df.columns:
                    st.error(f"Selected column '{smiles_column}' not found in the file.")
                 else:
                    smiles_list_from_file = input_df[smiles_column].dropna().astype(str).tolist()

                    if not smiles_list_from_file:
                        st.warning(f"No valid SMILES found in the '{smiles_column}' column.")
                    else:
                        st.subheader("Prediction Results from File")
                        with st.spinner(f"Predicting activity for {len(smiles_list_from_file)} molecules..."):
                            prediction_results_list = []
                            progress_bar = st.progress(0)
                            for i, smiles in enumerate(smiles_list_from_file):
                                result = predict_activity(smiles, dppiv_stacked_model, sglt2_stacked_model)
                                if result:
                                    prediction_results_list.append(result)
                                progress_bar.progress((i + 1) / len(smiles_list_from_file))

                            predicted_results_df = pd.DataFrame(prediction_results_list)

                            if not predicted_results_df.empty:
                                st.dataframe(predicted_results_df)

                                st.subheader("Predicted Dual Inhibitors (Active for both DPP-IV and SGLT2)")
                                dual_active_df = predicted_results_df[
                                    (predicted_results_df['DPPIV_Prediction'] == 1) &
                                    (predicted_results_df['SGLT2_Prediction'] == 1)
                                ]
                                if not dual_active_df.empty:
                                    st.dataframe(dual_active_df)
                                else:
                                    st.info("No compounds in the uploaded file were predicted to be active for both targets.")

                                # Option to download results
                                csv_data = predicted_results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Prediction Results as CSV",
                                    data=csv_data,
                                    file_name="prediction_results.csv",
                                    mime="text/csv"
                                )

                            else:
                                st.warning("No valid prediction results were generated from the file.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Optional: Add information about the models or methodology
st.sidebar.header("About")
st.sidebar.info("""
This app uses stacked machine learning models to predict if a molecule is an inhibitor for DPP-IV and SGLT2.
The models were trained on data from ChEMBL.
Predictions are based on Morgan fingerprints of the molecules.
""")
