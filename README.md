*Tech Stack*
- Langchain & Langgraph: for developing AI agents workflow.
- Langserve: simplify API development & deployment (using FastAPI).
- Groq and Gemini APIs: for LLMs access.
- Google Gmail API


*How to Run*
- Python 3.11
- Groq api key
- Google Gemini api key (for embeddings)
- Gmail API credentials

*Setup*
1. Clone the repository:
- git clone https://github.com/Gashiykh/homeproduct.git
- cd homeproduct

2. Create and activate a virtual environment:
- python -m venv venv
- source venv/bin/activate (Wind venv\Scripts\activate)

3. Install the required packages:
- pip install -r requirements.txt

4. Set up environment variables:
MY_EMAIL=your_email@gmail.com  
GROQ_API_KEY=your_groq_api_key  
GOOGLE_API_KEY=your_gemini_api_key  


*How to start*
- python main.py 