#!/bin/bash
set -e

echo "🔧 Fixing OpenCV conflicts..."

# Uninstall opencv-contrib-python if exists
/opt/venv/bin/pip uninstall -y opencv-contrib-python 2>/dev/null || echo "opencv-contrib-python not found"

# Reinstall opencv-python-headless
/opt/venv/bin/pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84

echo "✅ OpenCV fixed!"
echo "🚀 Starting Streamlit..."

# Start Streamlit
/opt/venv/bin/streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
