Anomaly Intelligence Dashboard engine leverages unsupervised clustering, density-based, and model based anomaly detection techniques to identify irregular and quantify risk levels.

Key Features:
- Multiple Anomaly Detection Models:
  - DBSCAN: Detects density based clusters and identifies isolated points. Indentifies loaner
  - OPTICS: Measures reachability distance to detect outliers within clusters. Identifies what dbscan cant
  - HDBSCAN: Probabiltiy clustering with outlier detection and confidence scoring. Combines DBSCAN and OPTICS.
  - ISOLATION FOREST: Detects anomalies by isolating points in a forest of trees. Bool is used.
  - ONE CLASS SVM: Model based anomaly scoring for outlier verification. Covers outer boundaries.

Scoring & Classification:
- Assigns numeric scores for cluster based anomalies.
- Converts scores into High/Medium/Low labes based on each score using quanitle: 'High' 0.75, 'Low' 0.25, else 'Medium'
  
- Combines model outputs into final alerts for actionable insight..
  -  Combine all DBSCAN, OPTICS, HDBSCAN, LOCAL OUTLIER FACTOR to one final category
  -  Combine all scores from final category to Isolation forest.
  -  Combine all scores from final category to One Class SVM.
  -  Final: Combine finall Isolation and final SVM.
 
Dashboard Integration:
- Interactive PowerBI visualize:
  - Isolation Risk: Combine all scores from final category to Isolation forest.
  - SVM Risk: Combine all scores from final category to One Class SVM.
  - Notification: Combine finall Isolation and final SVM.
 
Preprocessing & Future Enginnering:
- Handles numeric, categorical, and string features.
  - self.min = MinMaxScaler()
        self.one = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        self.num = self.copy.select_dtypes(include=['number']).columns
        self.obj = self.copy.select_dtypes(include=['object']).columns

        n_simple = SimpleImputer(strategy='median')
        o_simple = SimpleImputer(strategy='most_frequent')

        num_pipeline = Pipeline([('num', n_simple),
                                 ('scaler', self.min)])
        obj_pipeline = Pipeline([('obj', o_simple),
                                 ('encoder', self.one)])

        self.preprocessor = ColumnTransformer([('n', num_pipeline, self.num),
                                               ('o', obj_pipeline, self.obj)])

Flexible & Resuable Engine:
- Any dataset can be used.

Instructions:
- Data Preprocessing
- Dimensionality Reduction (PCA)
  
- Unsupervised Learning
  - DBSCAN, OPTICS, HDBSCAN, PCA, LOCAL OUTLIER FACTOR, ISOLATION FOREST, ONE CLASS SVM
    
- Risk Classification:
  - Final iso combine alert: DBSCAN, OPTICS, HDBSCAN, LOCAL OUTLIER FACTOR, ISOLATION FOREST
  - Final svm combine alert: DBSCAN, OPTICS, HDBSCAN, LOCAL OUTLIER FACOTR, ONE CLASS SVM
  - Final Notification: FINAL ISO COMBINE ALERT + FINAL SVM COMBINE ALERT
 
Final:
-Real-life anomaly detection engine that can consume any dataset and convert raw patterns into actionable insights.

Applicable to:

- Fraud detection

- Missing data or irregularities

- Operational risk and decision support
    
 

    
