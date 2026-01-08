import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BIC
from imblearn.under_sampling import RandomUnderSampler

# 1. Setup Data Path
CSV_PATH = '../data/dataset.csv'

def evaluate_performance():
    print("Step 1: Loading Data...")
    df = pd.read_csv(CSV_PATH)
    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['TYPE'])
    
    # Apply undersampling to balance the training data
    print(f"Original training data: {len(train_data)} rows")
    print(f"Class distribution before undersampling:\n{train_data['TYPE'].value_counts()}\n")
    
    rus = RandomUnderSampler(random_state=42)
    X_train = train_data.drop(columns=['TYPE'])
    y_train = train_data['TYPE']
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    # Reconstruct train_data with original column order
    train_data = X_resampled.copy()
    train_data['TYPE'] = y_resampled
    
    print(f"After undersampling: {len(train_data)} rows")
    print(f"Class distribution after undersampling:\n{train_data['TYPE'].value_counts()}\n")
    
    print(f"Step 2: Training on {len(train_data)} rows...")
    hc = HillClimbSearch(train_data)
    best_structure = hc.estimate(scoring_method=BIC(train_data), max_iter=500)
    
    model = DiscreteBayesianNetwork(best_structure.edges())
    model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
    
    print(f"Step 3: Testing on {len(test_data)} rows...")
    X_test = test_data.drop(columns=['TYPE'])
    y_true = test_data['TYPE']
    
    # Only use features that are in the model
    model_features = [node for node in model.nodes() if node != 'TYPE']
    X_test_filtered = X_test[model_features]
    
    y_pred_df = model.predict(X_test_filtered)
    y_pred = y_pred_df['TYPE']

    # 4. Calculate & Print Metrics
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE REPORT")
    print("="*50)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.2%}\n")

    print("--- Detailed Metrics by Disease ---")
    print(classification_report(y_true, y_pred))

    # 5. Text-Based Confusion Matrix
    print("\n--- Confusion Matrix ---")
    print("(Rows = Actual Disease, Columns = Predicted Disease)\n")
    
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(df['TYPE'].unique())
    
    # Create a nice Pandas DataFrame for display
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    print("="*50)

if __name__ == "__main__":
    evaluate_performance()