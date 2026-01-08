import streamlit as st
import pandas as pd
from src.network_structure import load_and_train_model
from src.triage_feature import TriageAgent
import itertools

st.set_page_config(page_title="AI Diagnostic Assistant", page_icon="🩺", layout="wide")

MEDICAL_DISCLAIMER = """
**Disclaimer:** This app provides a preliminary assessment based on the symptoms you reported. 
It is intended for self-evaluation purposes only and does not constitute a medical diagnosis. 
Please consult a qualified healthcare professional for medical advice, diagnosis, or treatment.
"""

# Memastikan modelnya hanya diload sekali saja
@st.cache_resource
def get_ai_system():
    # Load DAG dan inferencenya
    model, infer = load_and_train_model('data/dataset.csv')
    
    # Inisialisasi triage agentnya
    agent = TriageAgent(model, infer)
    return model, infer, agent

# Function untuk display
def display_results(probs):
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

def render_network_graph(model):
    # Code untuk generate graphnya supaya sekalian ditampilkan di main Streamlit pagenya
    try:
        dot_code = 'digraph G {\n'
        
        dot_code += '  rankdir=TB;\n'
        dot_code += '  splines=ortho;\n'
        dot_code += '  nodesep=0.4;\n'
        dot_code += '  ranksep=0.8;\n'
        dot_code += '  node [shape=box, style=filled, fillcolor="#f9f9f9", fontname="Sans-Serif"];\n'
        
        dot_code += '  "TYPE" [fillcolor="#ffcccc", penwidth=2];\n'
        
        for u, v in model.edges():
            dot_code += f'  "{u}" -> "{v}";\n'
            
        dot_code += '}'
        st.graphviz_chart(dot_code, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error visualizing graph: {e}")

def render_cpt_viewer(model):
    st.markdown("### 📊 Conditional Probability Tables (CPT)")
    
    # 1. Ambil daftar node
    nodes = sorted(model.nodes())
    if 'TYPE' in nodes:
        nodes.remove('TYPE')
        nodes.insert(0, 'TYPE')

    selected_node = st.selectbox("Choose Node:", nodes)

    if selected_node:
        try:
            # Ambil CPD
            cpd = model.get_cpds(selected_node)
            st.write(f"**Node:** {selected_node}")

            # Ambil states (kemungkinan nilai) untuk Node ini (Baris)
            child_states = cpd.state_names[selected_node]
            
            # Ambil Parents (Kolom)
            evidence = cpd.variables[1:] 
            
            if not evidence:
                # Jika Root Node (tidak punya parent)
                st.info("Root Node.")
                df = pd.DataFrame(cpd.values, index=child_states, columns=["Probability"])
            else:
                st.write(f"**Influenced by:** {', '.join(evidence)}")
                
                # --- PERBAIKAN HEADER KOLOM ---
                # Ambil state names untuk setiap parent
                parent_states_list = [cpd.state_names[parent] for parent in evidence]
                
                # Buat kombinasi (Cartesian Product)
                col_combinations = list(itertools.product(*parent_states_list))
                
                # Format Header yang CANTIK dan BERSIH
                col_headers = []
                for combo in col_combinations:
                    # Gabungkan Nama Parent dan Nilainya
                    # Contoh: "TYPE=COVID" atau "TYPE=FLU | FEVER=1"
                    parts = []
                    for var_name, state_val in zip(evidence, combo):
                        parts.append(f"{var_name}={state_val}")
                    col_headers.append(" | ".join(parts))
                
                # Reshape values menjadi 2D
                values_2d = cpd.values.reshape(len(child_states), -1)
                
                # Buat DataFrame
                df = pd.DataFrame(values_2d, index=child_states, columns=col_headers)

            # Tampilkan Tabel
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying CPD: {e}")

# Main App Flow
try:
    model, infer, agent = get_ai_system()
    
    # Sidebar Nav dan pilihan menunya
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode:", ["Manual Checklist", "Smart Triage (AI)", "Network Visualization"])
    st.sidebar.markdown("---")

    # Menu untuk pengisian semua gejala secara manual
    if app_mode == "Manual Checklist":
        st.title("📋 Symptom Checklist")
        st.warning(MEDICAL_DISCLAIMER, icon="⚠️")
        st.markdown("Select all symptoms that apply to see an immediate diagnosis.")
        st.divider()

        # Ambil semua node kecuali targetnya dan disort secara alfabet
        all_nodes = sorted([n for n in model.nodes() if n != 'TYPE'])
        
        # Buat form untuk checklistnya
        with st.form("checklist_form"):
            col1, col2, col3 = st.columns(3)
            user_evidence = {}
            
            # Format formnya pakai 3 kolom
            for i, symptom in enumerate(all_nodes):
                label = symptom.replace("_", " ").title()
                
                # Logika peletakan kolomnya
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

    # Mode menu untuk fitur tambahan
    elif app_mode == "Smart Triage (AI)":
        st.title("🩺 Smart Triage Assistant")
        st.markdown("I will ask specific questions to narrow down the diagnosis efficiently.")
        st.warning(MEDICAL_DISCLAIMER, icon="⚠️")
        st.divider()

        # Inisialisasi session state untuk fiturnya
        if 'triage_evidence' not in st.session_state:
            st.session_state['triage_evidence'] = {}
        if 'triage_history' not in st.session_state:
            st.session_state['triage_history'] = []
        if 'triage_finished' not in st.session_state:
            st.session_state['triage_finished'] = False

        # Proses perubahan probabilitasnya yang ditampilkan secara live
        with st.container():
            st.subheader("Live Analysis")
            current_probs = agent.get_current_prediction(st.session_state['triage_evidence'])
            
            if current_probs:
                # Buat layout atas
                metric_col, chart_col = st.columns([1, 2])
                
                # Helper buat prepare data
                df_probs = pd.DataFrame({
                    'Condition': list(current_probs.keys()),
                    'Probability': list(current_probs.values())
                }).sort_values(by='Probability', ascending=False)
                
                top_cond = df_probs.iloc[0]['Condition']
                top_prob = df_probs.iloc[0]['Probability']

                with metric_col:
                    st.metric(label="Primary Diagnosis", value=top_cond, delta=f"{top_prob:.1%}")
                    st.caption("Based on current evidence.")

                with chart_col:
                    st.bar_chart(df_probs.set_index('Condition'), height=200)
            else:
                st.info("Awaiting initial symptoms to generate a prediction.")

        st.divider()

        for q, a in st.session_state['triage_history']:
            q_text = q.replace("_", " ").title()
            ans_text = "✅ Yes" if a == 1 else "❌ No"
            st.info(f"**{q_text}?** — {ans_text}")

        # Buat pertanyaan selanjutnya
        if not st.session_state['triage_finished']:
            # AI logic: Find best question
            next_q = agent.get_next_best_question(st.session_state['triage_evidence'])
            
            if next_q:
                st.subheader(f"Do you have: {next_q.replace('_', ' ').title()}?")
                
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
    
    # Tab baru buat nampilin graphnya
    elif app_mode == "Network Visualization":
        st.title("🧠 Diagnostic Logic Map")
        st.markdown("Visual representation of the learned Bayesian Network structure.")
        st.divider()
        
        render_network_graph(model)
        st.divider()
        render_cpt_viewer(model)

except Exception as e:
    st.error(f"System Error: {e}")