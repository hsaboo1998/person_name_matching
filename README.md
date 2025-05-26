# person name matching
This repository is solution for person name matching using machine learning models

## Model Usage (For False Positive reduction)

### Name matching using pandas dataframe
    
    import pandas as pd
    import pickle
    from helper import get_results
    #read data
    df = pd.read_excel('data/Baseline_Entity_Matching_Dataset.xlsx')
    # load model
    with open('xgb_name_matcher.pkl', 'rb') as f:
        xgb_name_matcher = pickle.load(f)
    # prediction
    df[['prediction', 'proba']] = xgb_name_matcher.predict_using_pandas_dataframe(df, 'Name1', 'Name2', require_proba=True)
    # visualization
    get_results(df['Label'], df['proba'], threshold=xgb_name_matcher.threshold)
    
### Name matching score using a single name pair

    import pickle
    # load model
    with open('xgb_name_matcher.pkl', 'rb') as f:
        xgb_name_matcher = pickle.load(f)
    # prediction
    prediction, proba = xgb_name_matcher.predict_using_names('Hridaya', 'Hradaya', require_proba=True)

