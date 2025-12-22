import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BIC
from pgmpy.inference import VariableElimination

def load_and_train_model(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # 1. Prepare Data
    # Convert symptom columns to int (0/1) to ensure the math is smooth
    for col in df.columns:
        if col != 'TYPE':
            df[col] = df[col].astype(int)
    
    # 2. Hill Climbing Structural Learning
    hc = HillClimbSearch(df)
    scoring_method = BIC(df)
    
    # max_iter=1000 is plenty. For 21 variables, 100,000,000 is overkill.
    best_structure = hc.estimate(
        scoring_method=scoring_method, 
        max_iter=100, 
        show_progress=True
    )
    
    # 3. Create the Model
    model = DiscreteBayesianNetwork(best_structure.edges())
    
    # Safety Check: If data is too small, HC might find 0 edges.
    # We ensure all columns are at least nodes in the model.
    if len(model.edges()) == 0:
        model.add_nodes_from(df.columns)
    
    # 4. Fit and Infer
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(model)
    
    return model, infer