import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BIC
from pgmpy.inference import VariableElimination

def load_and_train_model(csv_path):
    # 1. Load the dataset
    df = pd.read_csv(csv_path)
    
    # 2. Optimized Data Preparation
    # With 40k rows, ensure we use efficient types
    for col in df.columns:
        if col != 'TYPE':
            df[col] = df[col].astype(int)

    # 3. Structural Learning (The "Thinking" Phase)
    hc = HillClimbSearch(df)
    
    # With 40k rows, BIC is extremely accurate. 
    # We increase max_iter because there is more data to explore.
    best_structure = hc.estimate(
        scoring_method=BIC(df), 
        max_iter=2000, 
        show_progress=True
    )
    
    # 4. Create the Model
    model = DiscreteBayesianNetwork(best_structure.edges())
    model.add_nodes_from(df.columns)
    
    # 5. High-Volume Probability Fitting
    # BDeu prior with 40k rows will be very precise.
    model.fit(
        df, 
        estimator=BayesianEstimator, 
        prior_type="BDeu", 
        equivalent_sample_size=5
    )
    
    # 6. Initialize Inference
    infer = VariableElimination(model)
    return model, infer