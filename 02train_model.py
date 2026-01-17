import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

print("="*60)
print("TRAIN DISTANCE-BASED MODEL (CLEAN VERSION)")
print("="*60)

# Load features
df_back = pd.read_csv('features_back_view.csv')
df_side = pd.read_csv('features_side_view.csv')

print(f"\nData loaded:")
print(f"  Back: {len(df_back)} videos")
print(f"  Side: {len(df_side)} videos")

results = {}

for view_name, df in [('back', df_back), ('side', df_side)]:
    print(f"\n{'='*60}")
    print(f"PROCESSING {view_name.upper()} VIEW")
    print(f"{'='*60}")
    
    # Get features
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    X = df[feature_cols].values
    
    print(f"Shape: {X.shape}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate PRO centroid
    centroid = np.mean(X_scaled, axis=0)
    
    # Calculate distances
    distances = np.linalg.norm(X_scaled - centroid, axis=1)
    
    # Statistics
    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    dist_max = np.max(distances)
    
    # Threshold: mean + 1.5*std
    threshold = dist_mean + 1.5 * dist_std
    
    print(f"\n📊 Statistics:")
    print(f"  Distance mean: {dist_mean:.4f}")
    print(f"  Distance std: {dist_std:.4f}")
    print(f"  Distance max: {dist_max:.4f}")
    print(f"  PRO threshold: {threshold:.4f}")
    
    # Save model
    model_data = {
        'centroid': centroid,
        'dist_mean': float(dist_mean),
        'dist_std': float(dist_std),
        'threshold_pro': float(threshold),
        'type': 'distance'
    }
    
    scaler_file = f'scaler_{view_name}_v2.pkl'
    model_file = f'model_{view_name}_v2.pkl'
    
    joblib.dump(scaler, scaler_file)
    joblib.dump(model_data, model_file)
    
    print(f"\n💾 Saved:")
    print(f"  {scaler_file}")
    print(f"  {model_file}")
    
    results[view_name] = {
        'threshold': float(threshold),
        'samples': len(X)
    }

# Save config
with open('model_config.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\nFiles created:")
print("  - scaler_back_v2.pkl")
print("  - model_back_v2.pkl")
print("  - scaler_side_v2.pkl")
print("  - model_side_v2.pkl")
print("  - model_config.json")
print("="*60)
