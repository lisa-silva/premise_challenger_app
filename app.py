streamlit
requests

import streamlit as st
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---

# This is the base model used for analytical tasks
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# The API Key is expected to be provided by the Canvas environment
# NOTE: When deploying to Streamlit Cloud, you will need to add this key
# as a secret named 'GEMINI_API_KEY' in your app's secrets management.
API_KEY = "" 

# Define the structure for the LLM's response using a JSON Schema
# This forces the model to categorize its arguments, making the output predictable and machine-readable.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {
            "type": "STRING",
            "description": "A neutral, one-sentence restatement of the original premise."
        },
        "counterArguments": {
            "type": "ARRAY",
            "description": "A list of 3-5 distinct, evidence-based arguments that challenge the premise.",
            "items": {"type": "STRING"}
        },
        "supportingEvidence": {
            "type": "ARRAY",
            "description": "A list of 3-5 distinct, evidence-based points that support the premise.",
            "items": {"type": "STRING"}
        },
        "conclusion": {
            "type": "STRING",
            "description": "A final, balanced conclusion on the strength of the original premise based on the conflicting evidence."
        }
    },
    "required": ["summary", "counterArguments", "supportingEvidence", "conclusion"]
}

# --- System Prompt Engineering ---
# This is the "secret sauce" that dictates the model's persona and behavior.
SYSTEM_INSTRUCTION = """
You are a highly analytical, unbiased "Premise Challenger" and "Devil's Advocate." 
Your sole function is to take an initial user premise (a statement or argument) and provide a completely balanced, evidence-based, structured analysis.

RULES OF ENGAGEMENT:
1. Objectivity: Maintain an absolutely neutral tone. Do not introduce personal opinions or bias.
2. Structure Mandate: You MUST adhere strictly to the provided JSON schema. Do not include any introductory or concluding text outside of the JSON block.
3. Balance: You must provide a nearly equal number of strong, distinct arguments for both the 'counterArguments' and 'supportingEvidence' arrays.
4. Depth: All points must be substantive, logical, and evidence-based.
5. Grounding: You must use Google Search to find current, real-time information to support or challenge the premise.
"""

# --- API and Error Handling ---

# Configure retries for robust API calls
def get_session():
    # Set up retry mechanism for handling transient errors
    retry_strategy = Retry(
        total=5,
        backoff_factor=2, # Exponential backoff (1s, 2s, 4s, 8s, 16s delay)
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"POST"}
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    return session

@st.cache_data(show_spinner="Analyzing premise...")
def challenge_premise(premise: str):
    """Calls the Gemini API to analyze the premise and return structured JSON."""
    
    payload = {
        "contents": [{"parts": [{"text": f"Analyze the following premise: {premise}"}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "config": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA
        },
        # Enable Google Search grounding tool
        "tools": [{"google_search": {}}]
    }

    session = get_session()
    
    try:
        response = session.post(
            f"{API_URL}?key={API_KEY}",
            headers={'Content-Type': 'application/json'},
            json=payload
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        
        # Extract the JSON text from the response
        json_text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
        
        # Parse the JSON output
        parsed_output = json.loads(json_text)
        
        # The API response does not include citations in the JSON output text itself.
        # We manually check the grounding metadata for citation sources.
        grounding_metadata = data.get('candidates', [{}])[0].get('groundingMetadata', {})
        sources = []
        if grounding_metadata and grounding_metadata.get('groundingAttributions'):
            sources = [
                {
                    'title': attr['web']['title'],
                    'uri': attr['web']['uri']
                }
                for attr in grounding_metadata['groundingAttributions']
                if 'web' in attr and 'title' in attr['web']
            ]
            
        return parsed_output, sources
    
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: Could not reach the AI service. Status: {e.response.status_code}. Please try again later.")
        return None, []
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, []

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="The Premise Challenger", initial_sidebar_state="collapsed")

# Custom CSS for aesthetics and mobile responsiveness
st.markdown("""
<style>
    /* Use Inter font for clean professional look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="st-emotion-cache"] { font-family: 'Inter', sans-serif; }

    /* Header styling */
    .st-emotion-cache-18ni7ap { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #5B21B6; } /* Deep Purple */
    .st-emotion-cache-10trblm { 
        text-align: center;
        background-color: #F3F4F6; /* Light gray background */
        padding: 1rem;
        border-radius: 0.75rem;
    }
    
    /* Input/Button styling */
    .st-emotion-cache-13m9kkn { 
        border: 2px solid #5B21B6; 
        border-radius: 0.5rem; 
    }
    .st-emotion-cache-l3z0o5 { /* Button style */
        background-color: #5B21B6;
        color: white;
        border-radius: 0.5rem;
        transition: background-color 0.3s;
    }
    .st-emotion-cache-l3z0o5:hover { background-color: #4C1D95; } /* Darker purple on hover */

    /* Output card styling */
    .card {
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .support { background-color: #ECFDF5; border-left: 5px solid #059669; } /* Green accent */
    .counter { background-color: #FEF2F2; border-left: 5px solid #EF4444; } /* Red accent */
    .conclusion { background-color: #E0E7FF; border-left: 5px solid #4F46E5; } /* Blue accent */

    /* List item spacing */
    .st-emotion-cache-1g8y7j4 ul {
        padding-left: 20px;
    }
    
    /* Responsive layout using columns */
    @media (max-width: 600px) {
        .card { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)


st.title("🧠 The Premise Challenger")
st.markdown("---")

st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <p>Submit any complex statement or argument. The AI, acting as a neutral Devil's Advocate, will challenge the premise by providing balanced, evidence-based arguments both **for** and **against** its validity.</p>
</div>
""", unsafe_allow_html=True)

# User Input
premise = st.text_area(
    "Enter a Premise to Challenge:",
    placeholder="e.g., 'Universal Basic Income (UBI) is the most effective solution to income inequality.'",
    height=150
)

# Challenge Button
if st.button("⚖️ Analyze Premise", use_container_width=True):
    if premise:
        try:
            # 1. Clear cache to ensure fresh run if run previously
            st.cache_data.clear()

            # 2. Call the analysis function
            result, sources = challenge_premise(premise)

            if result:
                st.subheader(f"Analysis: {result.get('summary', 'Summary not available')}")
                st.markdown("---")

                # 3. Display results in a two-column, structured format
                col1, col2 = st.columns(2)

                # Column 1: Counter-Arguments (Red/Challenge)
                with col1:
                    st.markdown('<div class="card counter">', unsafe_allow_html=True)
                    st.markdown("### 🛑 Counter-Arguments (Why the Premise is Weak)")
                    st.markdown("---")
                    
                    if result.get('counterArguments'):
                        st.markdown(f'<ul>{"".join([f"<li>{arg}</li>" for arg in result["counterArguments"]])}</ul>', unsafe_allow_html=True)
                    else:
                        st.write("No counter-arguments were generated.")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Column 2: Supporting Evidence (Green/Support)
                with col2:
                    st.markdown('<div class="card support">', unsafe_allow_html=True)
                    st.markdown("### ✅ Supporting Evidence (Why the Premise is Strong)")
                    st.markdown("---")
                    if result.get('supportingEvidence'):
                         st.markdown(f'<ul>{"".join([f"<li>{arg}</li>" for arg in result["supportingEvidence"]])}</ul>', unsafe_allow_html=True)
                    else:
                        st.write("No supporting evidence was generated.")
                    st.markdown('</div>', unsafe_allow_html=True)

                # 4. Conclusion (Full width, Blue)
                st.markdown('<div class="card conclusion">', unsafe_allow_html=True)
                st.markdown("### ⚖️ Final Conclusion")
                st.write(result.get('conclusion', 'Conclusion not available.'))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 5. Display Sources
                if sources:
                    st.caption("---")
                    st.caption("🔎 Grounding Sources (Real-time Web Search)")
                    source_links = [f"[{i+1}. {s['title']}]({s['uri']})" for i, s in enumerate(sources)]
                    st.markdown(" | ".join(source_links))

        except Exception as e:
            st.error(f"An unexpected deployment error occurred: {e}")
    else:
        st.warning("Please enter a premise to begin the analysis.")

st.markdown("---")
st.markdown("Powered by Gemini 2.5 Flash with Google Search Grounding.")
