## RAG Application based on AWS Bedrock
This RAG (Retrieval Augmented Generation) application on AWS Bedrock enables users 
to read documents stored in the data folder and ask questions about their contents.

Ensure that all your PDF documents are placed within the 'data' folder.

You should enable the correct bedrock model in your account and log into your AWS account using the appropriate credentials.

Step 1: Create a virtual environment named 'myenv'
```bash
python3 -m venv myenv
```
Step 2: Activate the virtual environment
```bash
source myenv/bin/activate  # On Unix or MacOS
myenv\Scripts\activate     # On Windows
```
Step 3: Install necessary packages
```bash
pip install -r requirements.txt
```
Step 4: Launch and interact with a Streamlit application defined in the Python script myapp.py
```bash
streamlit run app.py  
```
Step 5: Deactivate the virtual environment when done
```bash
deactivate
```
