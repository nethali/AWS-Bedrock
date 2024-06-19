# Ensure that all your PDF documents are placed within the 'data' folder.

# Step 1: Create a virtual environment named 'myenv'
python3 -m venv myenv

# Step 2: Activate the virtual environment
source myenv/bin/activate  # On Unix or MacOS
myenv\Scripts\activate     # On Windows

# Step 3: Install necessary packages
pip install -r requirements.txt

# Step 4: Deactivate the virtual environment when done
deactivate

# Launch and interact with a Streamlit application defined in the Python script myapp.py
streamlit run myapp.py  
