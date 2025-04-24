# Medical Specialist Consultation App

A Streamlit-based application that analyzes medical records, routes users to appropriate medical specialists, and provides AI-powered medical consultations.

## Features

- **Medical Record Analysis**: Upload and process PDF or Word documents containing medical records
- **Specialist Routing**: Automatically classifies health queries and routes to the appropriate specialist
- **Multiple Specialist Domains**: Supports General Health, Neurology, Cardiology, and Orthopedics
- **Chat History**: Save and retrieve previous consultations
- **Detailed Logging**: Track system operations with comprehensive logging

## Specialists Available

- **General Health**: Common health concerns and wellness information
- **Neurology**: Brain, nervous system, headaches, seizures, etc.
- **Cardiology**: Heart health, blood pressure, circulation, etc.
- **Orthopedics**: Bones, joints, muscles, arthritis, etc.

## Installation

1. Clone this repository or download the source code

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your NVIDIA API key (if not already in the code):
   ```bash
   export NVIDIA_API_KEY="your-api-key-here"
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run medical_specialist_app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your medical records (PDF or Word document)

4. Describe your symptoms or ask questions in the chat interface

5. The AI will classify your query and route you to the appropriate specialist

6. Previous consultations can be accessed from the sidebar

## Deployment on Streamlit Cloud

1. Create a GitHub repository and push your code

2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)

3. Create a new app and connect it to your GitHub repository

4. Set the main file path to `medical_specialist_app.py`

5. Add your NVIDIA API key as a secret in the Streamlit Cloud dashboard:
   - Key: `NVIDIA_API_KEY`
   - Value: `your-api-key-here`

6. Deploy the application

## Requirements

See `requirements.txt` for a complete list of dependencies.

## Disclaimer

This application is for informational purposes only and does not replace professional medical advice. Always consult with qualified healthcare professionals for diagnosis and treatment.

## License

MIT

