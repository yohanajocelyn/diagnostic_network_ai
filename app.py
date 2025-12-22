import streamlit as st
import pandas as pd
from src.network_structure import load_and_train_model

# Page setup
st.set_page_config(page_title="AI Diagnostic Assistant", page_icon="🩺")
st.title("🩺 Bayesian Diagnostic Assistant")
st.markdown("Select your symptoms in the sidebar to calculate probabilities.")

# Load Model (Cached to prevent retraining on every click)
# Since training on 40,000 rows might take 1–5 minutes depending on your CPU, you must use Streamlit Caching effectively so the user doesn't wait every time they move their mouse.
@st.cache_resource
def initialize_model():
    # Relative path to your data folder
    return load_and_train_model('data/dataset.csv')

try:
    model, infer = initialize_model()

    # Sidebar UI
    st.sidebar.header("Symptom Checklist")
    user_evidence = {}
    
    # Extract symptoms from model nodes
    symptoms = sorted([node for node in model.nodes() if node != 'TYPE'])

    for s in symptoms:
        label = s.replace("_", " ").title()
        user_evidence[s] = 1 if st.sidebar.checkbox(label) else 0

    # Analysis Button
    if st.button("Run Diagnostic Analysis"):
        if sum(user_evidence.values()) == 0:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing data patterns..."):
                # Perform Inference
                result = infer.query(variables=['TYPE'], evidence=user_evidence)
                
                # Format Results into a DataFrame for Visualization
                prob_values = result.values
                states = result.state_names['TYPE']
                prob_df = pd.DataFrame({'Condition': states, 'Probability': prob_values})
                prob_df = prob_df.sort_values(by='Probability', ascending=False)

                # Show Bar Chart
                st.subheader("Probability Distribution")
                st.bar_chart(prob_df.set_index('Condition'))

                # Show Top Result
                top_condition = prob_df.iloc[0]
                st.success(f"Primary match: **{top_condition['Condition']}** ({top_condition['Probability']:.2%} confidence)")
                
                # Show full table
                with st.expander("View detailed probability table"):
                    st.table(prob_df)

except Exception as e:
    st.error(f"Error loading system: {e}")