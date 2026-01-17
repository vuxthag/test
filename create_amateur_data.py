import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("="*60)
print("CREATE SYNTHETIC AMATEUR DATA")
print("="*60)

# Load PRO data
df_back = pd.read_csv('features_back_view.csv')
df_side = pd.read_csv('features_side_view.csv')

print(f"\n📊 Original PRO data:")
print(f"  Back: {len(df_back)} videos")
print(f"  Side: {len(df_side)} videos")

def create_amateur_variants(df, n_variants=2):
    """
    Tạo amateur data bằng cách:
    1. Thêm noise vào features
    2. Scale một số features (simulate bad posture)
    3. Randomize timing (simulate inconsistency)
    """
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    X = df[feature_cols].values
    
    amateur_samples = []
    
    for i in range(len(X)):
        pro_sample = X[i]
        
        # Tạo nhiều variants cho mỗi PRO sample
        for variant in range(n_variants):
            amateur = pro_sample.copy()
            
            # Strategy 1: Add Gaussian noise (simulate inconsistency)
            noise_level = np.random.uniform(0.15, 0.35)
            noise = np.random.normal(0, noise_level, amateur.shape)
            amateur = amateur + noise * np.std(amateur)
            
            # Strategy 2: Distort specific features (simulate bad technique)
            # Angles (first 6*12=72 features)
            angle_indices = np.arange(0, 72)
            distort_mask = np.random.choice([0, 1], size=len(angle_indices), p=[0.7, 0.3])
            distort_amount = np.random.uniform(-0.3, 0.3, size=len(angle_indices))
            amateur[angle_indices] = amateur[angle_indices] * (1 + distort_mask * distort_amount)
            
            # Strategy 3: Reduce smoothness (affect velocity features)
            # Velocity features (indices 144-216)
            velocity_indices = np.arange(144, 216)
            velocity_noise = np.random.normal(0, 0.4, len(velocity_indices))
            amateur[velocity_indices] = amateur[velocity_indices] + velocity_noise * np.std(amateur[velocity_indices])
            
            amateur_samples.append(amateur)
    
    return np.array(amateur_samples)

# Generate amateur data
print("\n🔄 Generating synthetic amateur data...")

amateur_back = create_amateur_variants(df_back, n_variants=2)
amateur_side = create_amateur_variants(df_side, n_variants=2)

print(f"\n✅ Generated amateur data:")
print(f"  Back: {len(amateur_back)} samples")
print(f"  Side: {len(amateur_side)} samples")

# Save amateur data
for view, amateur_data, pro_df in [('back', amateur_back, df_back), 
                                     ('side', amateur_side, df_side)]:
    # Create amateur DataFrame
    feature_cols = [c for c in pro_df.columns if c.startswith('feat_')]
    
    amateur_df = pd.DataFrame(amateur_data, columns=feature_cols)
    amateur_df['view'] = view
    amateur_df['level'] = 'amateur'
    amateur_df['video_id'] = [f'amateur_synthetic_{i}' for i in range(len(amateur_data))]
    
    # Save
    filename = f'features_{view}_amateur.csv'
    amateur_df.to_csv(filename, index=False)
    print(f"  💾 Saved: {filename}")

print("\n" + "="*60)
print("✅ AMATEUR DATA CREATION COMPLETE!")
print("="*60)
