import os
import glob
import pandas as pd

print("="*60)
print("STEP 0: SCAN PRO GOLF DATASET (BACK + SIDE VIEWS)")
print("="*60)

# Path chứa video
BASE_PATH = r"D:\Documents\VTK_Team_DataStorm2025\Final\Videos_data"

# Hoặc path từ ảnh
# BASE_PATH = r"D:\Documents\VTK_Team_DataStorm2025\TDTU-Golf-Pose-v1\Final\Videos_data"

data = []

# Kiểm tra path có tồn tại
if not os.path.exists(BASE_PATH):
    print(f"❌ Error: Path not found: {BASE_PATH}")
    print("Please update BASE_PATH to correct location")
    exit()

# Lấy tất cả file .mp4
all_videos = glob.glob(os.path.join(BASE_PATH, "*.mp4"))

if len(all_videos) == 0:
    print(f"⚠️  No videos found in {BASE_PATH}")
    exit()

# Parse từng video
for vpath in all_videos:
    fname = os.path.basename(vpath)
    fname_lower = fname.lower()
    
    # Detect view từ tên file
    if fname_lower.startswith('back'):
        view = 'back'
    elif fname_lower.startswith('side'):
        view = 'side'
    else:
        view = 'unknown'
        print(f"⚠️  Unknown view for: {fname}")
    
    # Extract số thứ tự (optional)
    import re
    match = re.search(r'\d+', fname)
    video_id = match.group() if match else None
    
    data.append({
        'filename': fname,
        'full_path': vpath,
        'view': view,
        'video_id': video_id,
        'level': 'pro',  # TẤT CẢ LÀ PRO
        'band': 'pro'     # hoặc có thể để 'unknown' nếu không biết band cụ thể
    })

# Tạo DataFrame
df = pd.DataFrame(data)

# Statistics
print(f"\n✅ Found {len(df)} videos")
print(f"\n📊 Distribution by view:")
print(df['view'].value_counts())

back_count = len(df[df['view'] == 'back'])
side_count = len(df[df['view'] == 'side'])

print(f"\n📈 Summary:")
print(f"  - Backview videos: {back_count}")
print(f"  - Sideview videos: {side_count}")
print(f"  - Total: {len(df)}")
print(f"  - All videos are PRO level ✅")

# Sort by view then video_id
df['video_id_int'] = df['video_id'].astype(int)
df = df.sort_values(['view', 'video_id_int'])
df = df.drop('video_id_int', axis=1)

# Save
output_file = 'dataset_metadata_pro.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Saved to {output_file}")

# Show sample
print(f"\n📋 Sample data (first 10 rows):")
print(df.head(10)[['filename', 'view', 'level', 'video_id']])

print("\n" + "="*60)
print("NEXT STEP: Run feature extraction")
print("python 01_extract_all_features.py")
print("="*60)