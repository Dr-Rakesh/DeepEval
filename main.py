import nest_asyncio
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
import pandas as pd
import os
from dotenv import load_dotenv
from back import evaluate_with_deepeval
import logging
import io

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Load the OpenAI API key from the environment (ensure it's defined in your .env file)
#os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Generate a random string of 32 bytes for secret key (unused at the moment)
secret_key = secrets.token_hex(32)

# CORS configuration
origins = ["http://127.0.0.1:8000"]  # Update this list based on deployment environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 template configuration
templates = Jinja2Templates(directory="templates")

# Request model for evaluation
class EvaluationRequest(BaseModel):
    paragraph_content: str
    user_question: str
    user_answer: str
    llm_answer: str

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """
    Render the homepage (index.html).
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/az_deepeval", response_class=StreamingResponse)
async def azure_deep_eval(request: EvaluationRequest):
    """
    Evaluate the provided data with DeepEval and return a PDF report.
    """
    # Validation checks on request data
    if len(request.paragraph_content) < 100:
        logging.error('Paragraph content must be at least 100 characters long.')
        raise HTTPException(status_code=400, detail='Paragraph content must be at least 100 characters long.')

    if len(request.user_question) < 20:
        logging.error('User question must be at least 20 characters long.')
        raise HTTPException(status_code=400, detail='User question must be at least 20 characters long.')
    
    # Prepare DataFrame for evaluation
    df = pd.DataFrame({
        'questions': [request.user_question],
        'answers': [request.user_answer],
        'contexts': [[request.paragraph_content]],
        'llm_answer': [request.llm_answer]
    })

    # Call the evaluation function
    eval_results = evaluate_with_deepeval(df)

    # Return the PDF file as a streaming response
    return StreamingResponse(
        eval_results,  # eval_results is already a buffer
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=evaluation_report.pdf"}
    )

if __name__ == '__main__':
    # Apply nest_asyncio to allow event loop re-entry
    nest_asyncio.apply()

    # Run Uvicorn server
    uvicorn.run(app, host='0.0.0.0', port=8000)