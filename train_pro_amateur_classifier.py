import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

print("="*60)
print("TRAIN PRO vs AMATEUR CLASSIFIER")
print("="*60)

results = {}

for view in ['back', 'side']:
    print(f"\n{'='*60}")
    print(f"TRAINING {view.upper()} VIEW CLASSIFIER")
    print(f"{'='*60}")
    
    # Load PRO data
    df_pro = pd.read_csv(f'features_{view}_view.csv')
    feature_cols = [c for c in df_pro.columns if c.startswith('feat_')]
    X_pro = df_pro[feature_cols].values
    y_pro = np.ones(len(X_pro))  # Label 1 = PRO
    
    # Load Amateur data
    df_amateur = pd.read_csv(f'features_{view}_amateur.csv')
    X_amateur = df_amateur[feature_cols].values
    y_amateur = np.zeros(len(X_amateur))  # Label 0 = Amateur
    
    print(f"\n📊 Data:")
    print(f"  PRO: {len(X_pro)} samples")
    print(f"  Amateur: {len(X_amateur)} samples")
    
    # Combine
    X = np.vstack([X_pro, X_amateur])
    y = np.hstack([y_pro, y_amateur])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📂 Split:")
    print(f"  Train: {len(X_train)} ({np.sum(y_train==1)} PRO, {np.sum(y_train==0)} Amateur)")
    print(f"  Test: {len(X_test)} ({np.sum(y_test==1)} PRO, {np.sum(y_test==0)} Amateur)")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print(f"\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = rf.score(X_train_scaled, y_train)
    test_acc = rf.score(X_test_scaled, y_test)
    
    print(f"\n📈 Performance:")
    print(f"  Train Accuracy: {train_acc*100:.1f}%")
    print(f"  Test Accuracy: {test_acc*100:.1f}%")
    
    # Detailed metrics
    y_pred = rf.predict(X_test_scaled)
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Amateur', 'PRO']))
    
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # Feature importance (top 10)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    print(f"\n🎯 Top 10 Important Features:")
    for idx in top_indices:
        print(f"  feat_{idx}: {importances[idx]:.4f}")
    
    # Save models
    scaler_file = f'scaler_classifier_{view}_v3.pkl'
    model_file = f'classifier_{view}_v3.pkl'
    
    joblib.dump(scaler, scaler_file)
    joblib.dump(rf, model_file)
    
    print(f"\n💾 Saved:")
    print(f"  {scaler_file}")
    print(f"  {model_file}")
    
    results[view] = {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'n_pro': int(np.sum(y_test==1)),
        'n_amateur': int(np.sum(y_test==0))
    }

# Save results
with open('classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("✅ CLASSIFIER TRAINING COMPLETE!")
print("="*60)
print("\nFiles created:")
print("  - scaler_classifier_back_v3.pkl")
print("  - classifier_back_v3.pkl")
print("  - scaler_classifier_side_v3.pkl")
print("  - classifier_side_v3.pkl")
print("  - classifier_results.json")
print("\n⏭️  Next: Update Streamlit app")
print("="*60)
