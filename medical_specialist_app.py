import streamlit as st
import os
import tempfile
import docx
import PyPDF2
import pickle
import asyncio
import logging
from datetime import datetime
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.llms import ChatMessage, MessageRole

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MedicalAssistant")

# Custom log handler to capture logs in session state
class SessionStateLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        if 'log_messages' in st.session_state:
            st.session_state.log_messages.append(log_entry)
            # Keep only the last 100 log messages
            if len(st.session_state.log_messages) > 100:
                st.session_state.log_messages = st.session_state.log_messages[-100:]

# Add the custom handler to the logger
session_handler = SessionStateLogHandler()
session_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
logger.addHandler(session_handler)

# Set environment variables
if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = "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"

# Initialize LLM
@st.cache_resource
def get_llm():
    return NVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif uploaded_file.name.endswith(('.docx', '.doc')):
        doc = docx.Document(uploaded_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    else:
        return "Unsupported file format. Please upload a PDF or Word document."

# Function to save chat history
def save_chat_history(chat_id, messages, file_content=None, classification=None):
    # Create directory if it doesn't exist
    if not os.path.exists("chat_history"):
        os.makedirs("chat_history")
    
    # Save the chat data
    chat_data = {
        "messages": messages,
        "file_content": file_content,
        "classification": classification,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"chat_history/{chat_id}.pkl", "wb") as f:
        pickle.dump(chat_data, f)

# Function to load chat history
def load_chat_history(chat_id):
    try:
        with open(f"chat_history/{chat_id}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# List available chat histories
def list_chat_histories():
    if not os.path.exists("chat_history"):
        return []
    
    histories = []
    for filename in os.listdir("chat_history"):
        if filename.endswith(".pkl"):
            chat_id = filename[:-4]
            try:
                chat_data = load_chat_history(chat_id)
                histories.append({
                    "id": chat_id,
                    "last_updated": chat_data["last_updated"],
                    "preview": chat_data["messages"][0]["content"][:50] + "..." if chat_data["messages"] else "New conversation"
                })
            except:
                pass
    
    # Sort by last updated time (newest first)
    histories.sort(key=lambda x: x["last_updated"], reverse=True)
    return histories

# Base Medical Agent class for specialized medical domains
class MedicalAgent:
    """Base class for medical agents with specialized domain knowledge"""
    
    def __init__(self, classification="GENERAL"):
        self.classification = classification
        self.name = classification.title()
        self.description = "Medical specialist"
    
    async def generate_response(self, query, medical_context, conversation_context):
        """Generate a response (to be implemented by specialized agents)"""
        raise NotImplementedError("This method should be implemented by subclasses")
    
    async def format_response(self, response):
        """Format the final response with consistent styling and disclaimer"""
        # Add a disclaimer
        disclaimer = """

**Medical Disclaimer**: The information provided is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
"""
        
        formatted_response = f"{response}\n{disclaimer}"
        
        logger.info("Response formatting complete")
        return formatted_response
    
    async def process_query(self, query, medical_context, conversation_context):
        """Process a query using this agent"""
        logger.info(f"Processing query with {self.name} agent: {query}")
        
        # Generate specialized response
        response = await self.generate_response(query, medical_context, conversation_context)
        
        # Format the response with disclaimer
        formatted_response = await self.format_response(response)
        
        logger.info(f"{self.name} agent processing complete")
        return formatted_response

# General Health Agent
class GeneralHealthAgent(MedicalAgent):
    """Agent for general health queries"""
    
    def __init__(self):
        super().__init__("GENERAL")
        self.name = "General Health"
        self.description = "General health information and wellness"
    
    async def generate_response(self, query, medical_context, conversation_context):
        """Generate response for general health query"""
        logger.info("Generating general health response")
        
        system_prompt = """
        You are a helpful general health assistant providing information about common health concerns.
        Provide CONCISE, clear, and accurate health information. Keep your responses brief and to the point.
        If the query requires specialist knowledge, acknowledge that limitation and suggest consulting a healthcare professional.
        When medical history information is available, use it to provide personalized advice.
        Always include important disclaimers about seeking professional medical advice, but keep them brief.
        Avoid unnecessary explanations and focus on answering the specific question asked.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL HISTORY:
        {medical_context}
        
        PREVIOUS CONVERSATION:
        {conversation_context}
        
        Provide a CONCISE, clear response addressing the patient's query.
        Limit your response to 3-5 sentences when possible.
        When referencing the medical history, be specific but brief about how it relates to the current question.
        Use bullet points instead of paragraphs when appropriate to improve readability.
        """
        
        llm = get_llm()
        response = await llm.acomplete(final_prompt)
        
        logger.info("General health response generated")
        return str(response)

# Neurology Agent
class NeurologyAgent(MedicalAgent):
    """Agent for neurology specialist queries"""
    
    def __init__(self):
        super().__init__("NEUROLOGY")
        self.name = "Neurology"
        self.description = "Brain, nervous system, headaches, seizures"
    
    async def generate_response(self, query, medical_context, conversation_context):
        """Generate response for neurology query"""
        logger.info("Generating neurology specialist response")
        
        system_prompt = """
        You are a specialized Neurology assistant with expertise in neurological conditions.
        Provide CONCISE information about brain, nervous system, headaches, strokes, seizures, 
        and other neurological topics, drawing on your specialized knowledge.
        Keep responses brief and focused on the specific question asked.
        When medical history information is available, use it to provide personalized advice 
        relevant to neurological conditions, but be succinct.
        Include a brief disclaimer about seeking professional medical advice.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL HISTORY:
        {medical_context}
        
        PREVIOUS CONVERSATION:
        {conversation_context}
        
        Provide a CONCISE response addressing the patient's query.
        Limit your response to 3-5 sentences when possible.
        When referencing the medical history, be specific but brief about how it relates to the current question.
        Use bullet points instead of paragraphs when appropriate to improve readability.
        Use your neurology expertise to provide specialized insights, but keep them focused and brief.
        """
        
        llm = get_llm()
        response = await llm.acomplete(final_prompt)
        
        logger.info("Neurology specialist response generated")
        return str(response)

# Cardiology Agent
class CardiologyAgent(MedicalAgent):
    """Agent for cardiology specialist queries"""
    
    def __init__(self):
        super().__init__("CARDIOLOGY")
        self.name = "Cardiology"
        self.description = "Heart health, blood pressure, circulation"
    
    async def generate_response(self, query, medical_context, conversation_context):
        """Generate response for cardiology query"""
        logger.info("Generating cardiology specialist response")
        
        system_prompt = """
        You are a specialized Cardiology assistant with expertise in cardiovascular conditions.
        Provide CONCISE information about heart health, blood pressure, circulation, chest pain, 
        and other cardiovascular topics, drawing on your specialized knowledge.
        Keep responses brief and focused on the specific question asked.
        When medical history information is available, use it to provide personalized advice 
        relevant to cardiovascular conditions, but be succinct.
        Include a brief disclaimer about seeking professional medical advice.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL HISTORY:
        {medical_context}
        
        PREVIOUS CONVERSATION:
        {conversation_context}
        
        Provide a CONCISE response addressing the patient's query.
        Limit your response to 3-5 sentences when possible.
        When referencing the medical history, be specific but brief about how it relates to the current question.
        Use bullet points instead of paragraphs when appropriate to improve readability.
        Use your cardiology expertise to provide specialized insights, but keep them focused and brief.
        """
        
        llm = get_llm()
        response = await llm.acomplete(final_prompt)
        
        logger.info("Cardiology specialist response generated")
        return str(response)

# Orthopedics Agent
class OrthopedicsAgent(MedicalAgent):
    """Agent for orthopedics specialist queries"""
    
    def __init__(self):
        super().__init__("ORTHOPEDICS")
        self.name = "Orthopedics"
        self.description = "Bones, joints, muscles, arthritis"
    
    async def generate_response(self, query, medical_context, conversation_context):
        """Generate response for orthopedics query"""
        logger.info("Generating orthopedics specialist response")
        
        system_prompt = """
        You are a specialized Orthopedics assistant with expertise in musculoskeletal conditions.
        Provide CONCISE information about bones, joints, muscles, arthritis, fractures, 
        and other orthopedic topics, drawing on your specialized knowledge.
        Keep responses brief and focused on the specific question asked.
        When medical history information is available, use it to provide personalized advice 
        relevant to orthopedic conditions, but be succinct.
        Include a brief disclaimer about seeking professional medical advice.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL HISTORY:
        {medical_context}
        
        PREVIOUS CONVERSATION:
        {conversation_context}
        
        Provide a CONCISE response addressing the patient's query.
        Limit your response to 3-5 sentences when possible.
        When referencing the medical history, be specific but brief about how it relates to the current question.
        Use bullet points instead of paragraphs when appropriate to improve readability.
        Use your orthopedics expertise to provide specialized insights, but keep them focused and brief.
        """
        
        llm = get_llm()
        response = await llm.acomplete(final_prompt)
        
        logger.info("Orthopedics specialist response generated")
        return str(response)

# Query classifier to determine if a question is general or domain-specific
async def classify_query(query):
    """Classify a health query as general or domain-specific (with specialist area)"""
    logger.info("Classifying query as general health or domain-specific")
    
    classify_prompt = f"""
    Classify the following health query as either GENERAL or one of these SPECIALIST domains:
    - NEUROLOGY (brain, nervous system, headaches, strokes, seizures, etc.)
    - CARDIOLOGY (heart, blood pressure, circulation, chest pain, etc.)
    - ORTHOPEDICS (bones, joints, muscles, arthritis, fractures, etc.)
    
    Query: {query}
    
    Return only one of these answers with no explanation:
    - GENERAL
    - NEUROLOGY
    - CARDIOLOGY
    - ORTHOPEDICS
    """
    
    llm = get_llm()
    response = await llm.acomplete(classify_prompt)
    classification = str(response).strip().upper()
    
    valid_classes = ["GENERAL", "NEUROLOGY", "CARDIOLOGY", "ORTHOPEDICS"]
    if classification not in valid_classes:
        # If classification is unclear, default to GENERAL
        logger.warning(f"Unclear classification '{classification}', defaulting to GENERAL")
        classification = "GENERAL"
    
    logger.info(f"Query classified as: {classification}")
    return classification

# Process query using the appropriate specialized agent
async def process_with_agent(query, medical_context, conversation_context, classification):
    """Process a medical query using the appropriate specialist agent"""
    logger.info(f"Processing {classification} query using agent: {query}")
    
    # Select the appropriate agent based on classification
    agent_map = {
        "GENERAL": GeneralHealthAgent(),
        "NEUROLOGY": NeurologyAgent(),
        "CARDIOLOGY": CardiologyAgent(),
        "ORTHOPEDICS": OrthopedicsAgent()
    }
    
    agent = agent_map.get(classification, GeneralHealthAgent())
    
    # Process the query with the selected agent
    result = await agent.process_query(query, medical_context, conversation_context)
    
    logger.info(f"{classification} agent processing completed")
    return result

# Set up Streamlit page
st.set_page_config(page_title="Medical Specialist Consultation", page_icon="🩺", layout="wide")
st.title("Medical Specialist Consultation")
st.markdown("""
Upload your medical records, and our AI will analyze them, 
discuss your symptoms, and route you to the appropriate medical specialist.
""")

# Initialize session state
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = datetime.now().strftime("%Y%m%d%H%M%S")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "file_content" not in st.session_state:
    st.session_state.file_content = None
    
if "classification" not in st.session_state:
    st.session_state.classification = None

if "processing" not in st.session_state:
    st.session_state.processing = False
    
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []

# Sidebar for chat management
with st.sidebar:
    st.header("Chat Management")
    
    # Option to create a new chat
    if st.button("Start New Consultation"):
        st.session_state.active_chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.messages = []
        st.session_state.file_content = None
        st.session_state.classification = None
        st.rerun()
    
    # Display previous chats
    st.subheader("Previous Consultations")
    histories = list_chat_histories()
    
    for history in histories:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"{history['last_updated']}: {history['preview']}", key=f"history_{history['id']}"):
                chat_data = load_chat_history(history['id'])
                st.session_state.active_chat_id = history['id']
                st.session_state.messages = chat_data["messages"]
                st.session_state.file_content = chat_data["file_content"]
                st.session_state.classification = chat_data.get("classification")
                st.rerun()
        with col2:
            if st.button("Delete", key=f"delete_{history['id']}"):
                try:
                    os.remove(f"chat_history/{history['id']}.pkl")
                    st.success("Chat deleted")
                    st.rerun()
                except:
                    st.error("Failed to delete chat")
    
    st.divider()
    st.header("How to Use This App")
    st.markdown("""
    1. Upload your medical records (PDF or Word document)
    2. Describe your symptoms or ask questions
    3. The AI will classify your query and route you to the appropriate specialist
    4. You'll receive insights from the relevant medical specialist
    
    **Note:** This app is for informational purposes only and does not replace 
    professional medical advice. Always consult with healthcare professionals 
    for medical concerns.
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader widget
    uploaded_file = st.file_uploader("Upload your medical records (PDF or Word)", type=["pdf", "docx", "doc"])

    # Process the uploaded file
    if uploaded_file and st.session_state.file_content is None:
        with st.spinner("Processing your medical records..."):
            st.session_state.file_content = extract_text_from_file(uploaded_file)
            
            # Add confirmation to chat
            st.session_state.messages.append({"role": "assistant", "content": 
                f"I've processed your medical records from '{uploaded_file.name}'. What symptoms or concerns would you like to discuss?"})
            
            # Save chat history
            save_chat_history(st.session_state.active_chat_id, st.session_state.messages, st.session_state.file_content)

with col2:
    if st.session_state.file_content:
        st.subheader("Uploaded Medical Records")
        with st.expander("View Extracted Text", expanded=False):
            st.text_area("Medical Record Content", st.session_state.file_content, height=300)
    
    # Display current specialist if classified
    if st.session_state.classification:
        specialist_emoji = {
            "GENERAL": "👨‍⚕️",
            "NEUROLOGY": "🧠",
            "CARDIOLOGY": "❤️",
            "ORTHOPEDICS": "🦴"
        }
        emoji = specialist_emoji.get(st.session_state.classification, "👨‍⚕️")
        st.info(f"{emoji} Currently consulting with: **{st.session_state.classification.title()} Specialist**")
    
    # Display logs in the sidebar
    st.subheader("Processing Logs")
    with st.expander("View System Logs", expanded=False):
        if st.session_state.log_messages:
            logs = "\n".join(st.session_state.log_messages)
            st.text_area("System Logs", logs, height=300)
        else:
            st.info("No logs available yet.")

# Display chat interface
st.subheader("Consultation Chat")
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What symptoms are you experiencing?"):
    if st.session_state.processing:
        st.warning("Please wait while I process your previous question.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Set processing flag
        st.session_state.processing = True
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Extract previous conversation context
                chat_history = []
                for msg in st.session_state.messages[-10:]:  # Get last 10 messages
                    if msg["role"] == "user":
                        chat_history.append(f"Patient: {msg['content']}")
                    else:
                        chat_history.append(f"Assistant: {msg['content']}")
                
                conversation_context = "\n".join(chat_history)
                medical_context = st.session_state.file_content or "No medical records provided."
                
                # Run async processing
                async def process_query_async():
                    # Step 1: Classify the query if not already classified
                    if not st.session_state.classification:
                        classification = await classify_query(prompt)
                        st.session_state.classification = classification
                    else:
                        # Use existing classification
                        classification = st.session_state.classification
                    
                    # Step 2: Process with appropriate specialist agent
                    response = await process_with_agent(
                        prompt, 
                        medical_context, 
                        conversation_context,
                        classification
                    )
                    
                    return response
                
                # Run the async function
                response_text = asyncio.run(process_query_async())
                
                # Display response
                st.markdown(response_text)
                
                # Save assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Save chat history with classification
                save_chat_history(
                    st.session_state.active_chat_id, 
                    st.session_state.messages, 
                    st.session_state.file_content,
                    st.session_state.classification
                )
                
                # Reset processing flag
                st.session_state.processing = False

# Footer
st.divider()
st.caption("Medical Specialist Consultation powered by NVIDIA AI and Llama 3.3")
