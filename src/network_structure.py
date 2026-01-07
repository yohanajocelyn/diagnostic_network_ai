import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BIC
from pgmpy.inference import VariableElimination

def load_and_train_model(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Preprocess the data,
    # TYPE adalah target, sisanya dimapping jadi 0 and 1
    for col in df.columns:
        if col != 'TYPE':
            df[col] = df[col].astype(int)

    # Set the HillClimbSearch buat df tersebut
    hc = HillClimbSearch(df)
    
    # Proses mencari struktur yang terbaik dengan BIC scoring
    # Max iteration dijadikan 2000
    best_structure = hc.estimate(
        scoring_method=BIC(df), 
        max_iter=2000, 
        show_progress=True
    )
    
    # Bikin gambar Bayesian Network berdasarkan struktur terbaik yang ditemukan + add nodes dari kolom-kolom datasetnya
    # karena best_structure hanya memberikan relationshipnya saja
    model = DiscreteBayesianNetwork(best_structure.edges())
    model.add_nodes_from(df.columns)
    
    # Dari relationship yang sudah ditemukan, fit ke modelnya untuk belajar probabilitynya
    model.fit(
        df, 
        estimator=BayesianEstimator, 
        prior_type="BDeu", # 
        equivalent_sample_size=5 # Untuk mencegah probabilitas 0 kalau semisal penyakit dan gejala yang tidak berhubungan
    )
    
    # Model matematika yang akan menghitungkan probabilitas berdasarkan evidence (gejala yang sudah dijawab) pakai .query()
    infer = VariableElimination(model)
    return model, inf