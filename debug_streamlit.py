import streamlit as st
import os
import sys

st.title("Streamlit Debug Information")

# 1. Check Python version
st.header("1. Python Version")
st.write(f"Python version: {sys.version}")

# 2. Check current working directory
st.header("2. Current Working Directory")
current_dir = os.getcwd()
st.write(f"Current directory: {current_dir}")

# 3. List all files and folders in root
st.header("3. Files in Root Directory")
try:
    root_contents = os.listdir('.')
    st.write("Contents:", root_contents)
    
    # Show more details
    for item in root_contents:
        item_path = os.path.join('.', item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            st.write(f"📄 {item} ({size} bytes)")
        elif os.path.isdir(item_path):
            st.write(f"📁 {item}/")
except Exception as e:
    st.error(f"Error listing root: {e}")

# 4. Check if models folder exists
st.header("4. Models Folder Check")
models_path = 'models'
if os.path.exists(models_path):
    st.success(f"✅ '{models_path}' folder EXISTS!")
    
    # List contents of models folder
    try:
        models_contents = os.listdir(models_path)
        st.write(f"Files in models folder: {models_contents}")
        
        # Show details of each file
        for item in models_contents:
            item_full_path = os.path.join(models_path, item)
            if os.path.isfile(item_full_path):
                size = os.path.getsize(item_full_path)
                st.write(f"  📄 {item} ({size} bytes)")
            else:
                st.write(f"  📁 {item}/")
    except Exception as e:
        st.error(f"Error reading models folder: {e}")
else:
    st.error(f"❌ '{models_path}' folder NOT FOUND!")
    
    # Check alternate paths
    st.write("Checking alternate paths:")
    alternate_paths = [
        './models',
        'Models',
        '/mount/src/ml-assignment-2/models',
    ]
    for alt_path in alternate_paths:
        if os.path.exists(alt_path):
            st.write(f"  ✅ Found at: {alt_path}")

# 5. Check specific model file
st.header("5. Specific Model File Check")
model_file = 'models/logistic_regression.pkl'
if os.path.exists(model_file):
    st.success(f"✅ '{model_file}' EXISTS!")
    size = os.path.getsize(model_file)
    st.write(f"File size: {size} bytes ({size/1024:.2f} KB)")
    
    # Try to read it
    try:
        with open(model_file, 'rb') as f:
            content = f.read(100)  # Read first 100 bytes
        st.success("✅ File is readable!")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.error(f"❌ '{model_file}' NOT FOUND!")

# 6. Check all .pkl files recursively
st.header("6. Search for ALL .pkl files")
pkl_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.pkl'):
            full_path = os.path.join(root, file)
            size = os.path.getsize(full_path)
            pkl_files.append((full_path, size))
            st.write(f"Found: {full_path} ({size} bytes)")

if not pkl_files:
    st.warning("⚠️ No .pkl files found anywhere in the repository!")

# 7. Environment info
st.header("7. Environment Information")
st.write(f"sys.path: {sys.path[:3]}...")  # Show first 3 paths

# 8. Try to import pickle
st.header("8. Pickle Module Check")
try:
    import pickle
    st.success("✅ pickle module imported successfully")
except Exception as e:
    st.error(f"❌ Error importing pickle: {e}")
