
# IMPLEMENTATION SCRIPT: Advanced Feature Engineering for m6A Prediction
# Use this code after loading your data from feature_engin.ipynb output

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(df):
    """
    Create advanced features building on your existing feature set
    Input: DataFrame with your current features from feature_engin.ipynb
    """
    df_enhanced = df.copy()

    # PHASE 1: POSITION-BASED INTERACTION FEATURES (HIGHEST PRIORITY)
    print("Creating Phase 1 features: Position-based interactions...")

    # 1.1 Signal Ratio Features (most effective according to m6Anet research)
    df_enhanced['central_flanking_time_ratio'] = df_enhanced['InTime'] / (df_enhanced['PreTime'] + df_enhanced['PostTime'] + 1e-8)
    df_enhanced['central_flanking_sd_ratio'] = df_enhanced['InSD'] / (df_enhanced['PreSD'] + df_enhanced['PostSD'] + 1e-8)
    df_enhanced['central_flanking_mean_ratio'] = df_enhanced['InMean'] / (df_enhanced['PreMean'] + df_enhanced['PostMean'] + 1e-8)

    df_enhanced['pre_post_time_ratio'] = df_enhanced['PreTime'] / (df_enhanced['PostTime'] + 1e-8)
    df_enhanced['pre_post_sd_ratio'] = df_enhanced['PreSD'] / (df_enhanced['PostSD'] + 1e-8)
    df_enhanced['pre_post_mean_ratio'] = df_enhanced['PreMean'] / (df_enhanced['PostMean'] + 1e-8)

    # 1.2 Signal Difference Features
    df_enhanced['central_pre_time_diff'] = df_enhanced['InTime'] - df_enhanced['PreTime']
    df_enhanced['central_post_time_diff'] = df_enhanced['InTime'] - df_enhanced['PostTime']
    df_enhanced['central_pre_mean_diff'] = df_enhanced['InMean'] - df_enhanced['PreMean']
    df_enhanced['central_post_mean_diff'] = df_enhanced['InMean'] - df_enhanced['PostMean']
    df_enhanced['central_pre_sd_diff'] = df_enhanced['InSD'] - df_enhanced['PreSD']
    df_enhanced['central_post_sd_diff'] = df_enhanced['InSD'] - df_enhanced['PostSD']

    # 1.3 Signal Gradient Features
    df_enhanced['mean_gradient_slope'] = (df_enhanced['PostMean'] - df_enhanced['PreMean']) / 2  # Slope across 3 positions
    df_enhanced['time_gradient_slope'] = (df_enhanced['PostTime'] - df_enhanced['PreTime']) / 2
    df_enhanced['sd_gradient_slope'] = (df_enhanced['PostSD'] - df_enhanced['PreSD']) / 2

    # PHASE 2: SIGNAL PROCESSING FEATURES 
    print("Creating Phase 2 features: Signal processing...")

    # 2.1 Signal Complexity Features
    df_enhanced['pre_signal_to_noise'] = df_enhanced['PreMean'] / (df_enhanced['PreSD'] + 1e-8)
    df_enhanced['in_signal_to_noise'] = df_enhanced['InMean'] / (df_enhanced['InSD'] + 1e-8)
    df_enhanced['post_signal_to_noise'] = df_enhanced['PostMean'] / (df_enhanced['PostSD'] + 1e-8)

    # 2.2 Position-wise Signal Strength
    df_enhanced['central_signal_strength'] = df_enhanced['InMean'] * df_enhanced['InTime'] / (df_enhanced['InSD'] + 1e-8)
    df_enhanced['flanking_signal_strength'] = (df_enhanced['PreMean'] * df_enhanced['PreTime'] + df_enhanced['PostMean'] * df_enhanced['PostTime']) / (df_enhanced['PreSD'] + df_enhanced['PostSD'] + 1e-8)

    # PHASE 3: MOTIF-AWARE FEATURES
    print("Creating Phase 3 features: Motif-aware features...")

    if 'SEQ' in df_enhanced.columns:
        # 3.1 Sequence composition features
        df_enhanced['gc_content'] = df_enhanced['SEQ'].apply(lambda x: (x.count('G') + x.count('C')) / len(x) if len(x) > 0 else 0)
        df_enhanced['purine_content'] = df_enhanced['SEQ'].apply(lambda x: (x.count('A') + x.count('G')) / len(x) if len(x) > 0 else 0)

        # 3.2 DRACH pattern strength (assuming 7-mer sequences)
        def drach_strength(seq):
            if len(seq) != 7:
                return 0
            # Check for DRACH pattern: D[AGT]R[AG]A[A]C[C]H[ACT]
            drach_score = 0
            if seq[2] in ['A', 'G', 'T']:  # D position
                drach_score += 1
            if seq[3] in ['A', 'G']:       # R position  
                drach_score += 1
            if seq[4] == 'A':              # A position (modification site)
                drach_score += 1
            if seq[5] == 'C':              # C position
                drach_score += 1
            if seq[6] in ['A', 'C', 'T']:  # H position
                drach_score += 1
            return drach_score / 5  # Normalize to 0-1

        df_enhanced['drach_strength'] = df_enhanced['SEQ'].apply(drach_strength)

    # PHASE 4: STATISTICAL INTERACTIONS (selected high-impact ones)
    print("Creating Phase 4 features: Statistical interactions...")

    df_enhanced['time_sd_interaction'] = df_enhanced['InTime'] * df_enhanced['InSD']
    df_enhanced['mean_time_interaction'] = df_enhanced['InMean'] * df_enhanced['InTime']
    df_enhanced['central_dominance'] = df_enhanced['InMean'] / (df_enhanced['PreMean'] + df_enhanced['PostMean'] + df_enhanced['InMean'] + 1e-8)

    print(f"Feature engineering complete. Added {len(df_enhanced.columns) - len(df.columns)} new features.")
    print(f"Total features: {len(df_enhanced.columns)}")

    return df_enhanced

def select_best_features(X, y, method='mutual_info', top_k=50):
    """
    Select best features using different methods
    """
    print(f"Selecting top {top_k} features using {method}...")

    if method == 'mutual_info':
        # Remove non-numeric columns for MI calculation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
        feature_scores = pd.DataFrame({
            'feature': X_numeric.columns,
            'score': mi_scores
        }).sort_values('score', ascending=False)

    elif method == 'f_score':
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        f_scores, p_values = f_classif(X_numeric, y)
        feature_scores = pd.DataFrame({
            'feature': X_numeric.columns,
            'score': f_scores,
            'p_value': p_values
        }).sort_values('score', ascending=False)

    elif method == 'random_forest':
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_numeric, y)

        feature_scores = pd.DataFrame({
            'feature': X_numeric.columns,
            'score': rf.feature_importances_
        }).sort_values('score', ascending=False)

    return feature_scores.head(top_k)

def evaluate_feature_set_performance(X, y, feature_list, model_type='rf'):
    """
    Evaluate performance of a specific feature set
    """
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Use only numeric features
    numeric_features = [f for f in feature_list if f in X.select_dtypes(include=[np.number]).columns]

    if len(numeric_features) == 0:
        return {'mean_auc': 0, 'std_auc': 0, 'num_features': 0}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X[numeric_features], y, cv=cv, scoring='roc_auc')

    return {
        'mean_auc': scores.mean(),
        'std_auc': scores.std(),
        'num_features': len(numeric_features),
        'features': numeric_features
    }

# MAIN WORKFLOW IMPLEMENTATION
def run_advanced_feature_engineering_workflow(df_path=None, df=None):
    """
    Complete workflow for advanced feature engineering
    """
    print("=== ADVANCED FEATURE ENGINEERING WORKFLOW ===")

    # Load data (either from path or passed dataframe)
    if df is None and df_path:
        df = pd.read_csv(df_path)
    elif df is None:
        print("Please provide either df_path or df parameter")
        return None

    print(f"Loaded data: {df.shape}")
    print(f"Original features: {len(df.columns)}")

    # Identify label column
    if 'label' in df.columns:
        label_col = 'label'
        y = df[label_col]
        X = df.drop(columns=[label_col])
    else:
        print("Warning: No 'label' column found. Please specify target variable.")
        return df

    print(f"Class distribution: {y.value_counts().to_dict()}")

    # 1. Create advanced features
    df_enhanced = create_advanced_features(df)
    X_enhanced = df_enhanced.drop(columns=[label_col])

    # 2. Baseline performance evaluation  
    print("=== BASELINE PERFORMANCE ===")
    baseline_features = [col for col in X.columns if col not in ['ID', 'POS', 'SEQ', 'gene_id']]
    baseline_perf = evaluate_feature_set_performance(X, y, baseline_features)
    print(f"Baseline AUC: {baseline_perf['mean_auc']:.4f} ± {baseline_perf['std_auc']:.4f}")
    print(f"Baseline features: {baseline_perf['num_features']}")

    # 3. Feature selection with different methods
    print("=== FEATURE SELECTION ===")

    methods = ['mutual_info', 'f_score', 'random_forest']
    feature_rankings = {}

    for method in methods:
        rankings = select_best_features(X_enhanced, y, method=method, top_k=50)
        feature_rankings[method] = rankings
        print(f"Top 10 features by {method}:")
        print(rankings.head(10)[['feature', 'score']].to_string(index=False))

    # 4. Evaluate different feature sets
    print("=== PERFORMANCE EVALUATION ===")

    # Test different numbers of top features
    feature_counts = [20, 30]
    results = {}

    for method in methods:
        results[method] = {}
        for k in feature_counts:
            top_features = feature_rankings[method].head(k)['feature'].tolist()
            perf = evaluate_feature_set_performance(X_enhanced, y, top_features)
            results[method][k] = perf
            print(f"{method} (top {k}): AUC = {perf['mean_auc']:.4f} ± {perf['std_auc']:.4f}")

    # 5. Find best feature set
    best_performance = 0
    best_config = None

    for method in methods:
        for k in feature_counts:
            if results[method][k]['mean_auc'] > best_performance:
                best_performance = results[method][k]['mean_auc']
                best_config = (method, k)

    print(f"=== BEST CONFIGURATION ===")
    print(f"Best method: {best_config[0]}")
    print(f"Best feature count: {best_config[1]}")  
    print(f"Best AUC: {best_performance:.4f}")
    print(f"Improvement over baseline: {(best_performance - baseline_perf['mean_auc']):.4f}")

    # 6. Get final feature list
    best_features = feature_rankings[best_config[0]].head(best_config[1])['feature'].tolist()
    print(f"FINAL SELECTED FEATURES ({len(best_features)}):")
    for i, feature in enumerate(best_features, 1):
        print(f"{i:2d}. {feature}")

    return {
        'enhanced_data': df_enhanced,
        'best_features': best_features,
        'feature_rankings': feature_rankings,
        'performance_results': results,
        'baseline_performance': baseline_perf,
        'best_performance': best_performance
    }
