Anomaly Intelligence Dashboard engine leverages unsupervised clustering, density-based, and model based anomaly detection techniques to identify irregular and quantify risk levels.

Key Features:
- Multiple Anomaly Detection Models:
  - DBSCAN: Detects density based clusters and identifies isolated points. Density severity.
  - OPTICS: Measures reachability distance to detect outliers within clusters. Reachability severity.
  - HDBSCAN: Probabiltiy clustering with outlier detection and confidence scoring. Probability severity.
  - ISOLATION FOREST: Detects anomalies by isolating points in a forest of trees. Decision function severity.
  - ONE CLASS SVM: Model based anomaly scoring for outlier verification. Decision function severity.

Scoring & Classification:
- Assigns numeric scores for cluster based severity level.
- Converts scores into High/Medium/Low labes based on each score using quanitle: 'High' 0.75, 'Low' 0.25, else 'Medium'
  
- Combines model outputs into final alerts for actionable insight..
  -  Combine all DBSCAN, OPTICS, HDBSCAN, LOCAL OUTLIER FACTOR, ISOLATION FOREST
  -  Combine all scores from desnsity severity level to One Class SVM.
  -  Final: Combine all for severity level.
 
Dashboard Integration:
- Interactive PowerBI visualize:
  - Based on Severity Level
 
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
  - Final density severity: DBSCAN, OPTICS(REACHABILITY), HDBSCAN(PROBABILITY, OUTLIER SCORES, LOCAL OUTLIER FACTOR), ISOLATION FOREST(DECISION FUNCTION) 
  - Final svm severity: Final density severity + ONE CLASS SVM(DECISION FUNCTION)
  - Final SEVERITY LEVEL: Final density severity + Final svm severity 
 
Final:
-Real-life anomaly detection engine that can consume any dataset and convert raw patterns into actionable insights.

Applicable to:

- Fraud detection

- Missing data or irregularities

- Operational risk and decision support
    
 

    
