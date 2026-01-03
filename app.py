import streamlit as st
import pandas as pd
from src.network_structure import load_and_train_model
from src.triage_feature import TriageAgent

# --- Page Config ---
st.set_page_config(page_title="AI Diagnostic Assistant", page_icon="🩺", layout="wide")

# --- 1. Load Model (Cached) ---
# This ensures we load the heavy AI model only ONCE, no matter how many times you switch menus.
@st.cache_resource
def get_ai_system():
    # Load the DAG and Inference Engine
    # Ensure your data path is correct relative to app.py
    model, infer = load_and_train_model('data/dataset.csv')
    
    # Initialize our Smart Triage Logic
    agent = TriageAgent(model, infer)
    return model, infer, agent

# --- 2. Helper Functions ---
def display_results(probs):
    """Reusable function to display probability charts"""
    if not probs:
        st.warning("No sufficient data to form a diagnosis yet.")
        return

    # Convert dict to DataFrame
    df_probs = pd.DataFrame({
        'Condition': list(probs.keys()),
        'Probability': list(probs.values())
    })
    
    # Sort for better visualization
    df_probs = df_probs.sort_values(by='Probability', ascending=False)
    
    # Top Prediction
    top_cond = df_probs.iloc[0]['Condition']
    top_prob = df_probs.iloc[0]['Probability']
    
    # Visuals
    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.metric(label="Primary Diagnosis", value=top_cond, delta=f"{top_prob:.1%}")
    with col_b:
        st.bar_chart(df_probs.set_index('Condition'))
    
    with st.expander("See detailed probabilities"):
        st.table(df_probs)

# --- 3. Main Application Flow ---
try:
    model, infer, agent = get_ai_system()
    
    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode:", ["Manual Checklist", "Smart Triage (AI)"])
    st.sidebar.markdown("---")

    # =========================================================
    # MODE 1: MANUAL CHECKLIST
    # =========================================================
    if app_mode == "Manual Checklist":
        st.title("📋 Symptom Checklist")
        st.markdown("Select all symptoms that apply to see an immediate diagnosis.")
        st.divider()

        # Get all symptoms from the model (excluding the Target 'TYPE')
        # We sort them alphabetically for easier reading
        all_nodes = sorted([n for n in model.nodes() if n != 'TYPE'])
        
        # Create a form so the page doesn't reload on every single checkbox click
        with st.form("checklist_form"):
            col1, col2, col3 = st.columns(3)
            user_evidence = {}
            
            # Distribute checkboxes across 3 columns
            for i, symptom in enumerate(all_nodes):
                label = symptom.replace("_", " ").title()
                
                # Logic to place in columns
                if i % 3 == 0:
                    with col1:
                        is_checked = st.checkbox(label)
                elif i % 3 == 1:
                    with col2:
                        is_checked = st.checkbox(label)
                else:
                    with col3:
                        is_checked = st.checkbox(label)
                
                user_evidence[symptom] = 1 if is_checked else 0
            
            submitted = st.form_submit_button("Run Analysis", type="primary")

        if submitted:
            if sum(user_evidence.values()) == 0:
                st.warning("Please select at least one symptom.")
            else:
                with st.spinner("Calculating Bayesian probabilities..."):
                    # Get prediction using the agent helper (reusing logic)
                    probs = agent.get_current_prediction(user_evidence)
                    display_results(probs)

    # =========================================================
    # MODE 2: SMART TRIAGE (Conversational)
    # =========================================================
    elif app_mode == "Smart Triage (AI)":
        st.title("🩺 Smart Triage Assistant")
        st.markdown("I will ask specific questions to narrow down the diagnosis efficiently.")
        st.divider()

        # Initialize Session State for Triage
        if 'triage_evidence' not in st.session_state:
            st.session_state['triage_evidence'] = {}
        if 'triage_history' not in st.session_state:
            st.session_state['triage_history'] = []
        if 'triage_finished' not in st.session_state:
            st.session_state['triage_finished'] = False

        # --- 1. LIVE ANALYSIS (MOVED TO TOP) ---
        # We use a container to keep this section distinct and wide
        with st.container():
            st.subheader("Live Analysis")
            current_probs = agent.get_current_prediction(st.session_state['triage_evidence'])
            
            if current_probs:
                # Create a top layout: Metric on Left, Chart on Right (Wider)
                metric_col, chart_col = st.columns([1, 2])
                
                # Helper to prepare data
                df_probs = pd.DataFrame({
                    'Condition': list(current_probs.keys()),
                    'Probability': list(current_probs.values())
                }).sort_values(by='Probability', ascending=False)
                
                top_cond = df_probs.iloc[0]['Condition']
                top_prob = df_probs.iloc[0]['Probability']

                with metric_col:
                    # Big bold metric
                    st.metric(label="Primary Diagnosis", value=top_cond, delta=f"{top_prob:.1%}")
                    st.caption("Based on current evidence.")

                with chart_col:
                    # Full width chart
                    st.bar_chart(df_probs.set_index('Condition'), height=200)
            else:
                st.info("Awaiting initial symptoms to generate a prediction.")

        st.divider()

        for q, a in st.session_state['triage_history']:
            q_text = q.replace("_", " ").title()
            ans_text = "✅ Yes" if a == 1 else "❌ No"
            st.info(f"**{q_text}?** — {ans_text}")

        # 2. Ask Next Question
        if not st.session_state['triage_finished']:
            # AI Logic: Find best question
            next_q = agent.get_next_best_question(st.session_state['triage_evidence'])
            
            if next_q:
                st.subheader(f"Do you have: {next_q.replace('_', ' ').title()}?")
                
                # Buttons
                c1, c2, _ = st.columns([1, 1, 3])
                if c1.button("Yes", key=f"yes_{next_q}", type="primary", use_container_width=True):
                    st.session_state['triage_evidence'][next_q] = 1
                    st.session_state['triage_history'].append((next_q, 1))
                    st.rerun()
                    
                if c2.button("No", key=f"no_{next_q}", use_container_width=True):
                    st.session_state['triage_evidence'][next_q] = 0
                    st.session_state['triage_history'].append((next_q, 0))
                    st.rerun()
            else:
                st.session_state['triage_finished'] = True
                st.rerun()
        else:
            st.success("Diagnostic Interview Complete.")
            if st.button("Restart Interview"):
                st.session_state['triage_evidence'] = {}
                st.session_state['triage_history'] = []
                st.session_state['triage_finished'] = False
                st.rerun()

except Exception as e:
    st.error(f"System Error: {e}")