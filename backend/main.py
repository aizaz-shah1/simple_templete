import requests
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import uvicorn
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger()

# Define FastAPI app
app = FastAPI(
    title="Car Damage Estimation Vision API",
    version="1.1",
    description="API for car damage estimation using NVIDIA's LLaMA-3.2 Vision model with optional car details."
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# NVIDIA API endpoint and configurations
INVOKE_URL = os.getenv("NVIDIA_API_URL", "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions")
API_KEY = os.getenv("NVIDIA_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

# Helper function to encode the uploaded image
def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@app.post("/estimate-damage")
async def estimate_car_damage(
    file: UploadFile = File(...), 
    car_name: Optional[str] = Query(None, description="Optional specified car name"),
    car_model: Optional[str] = Query(None, description="Optional specified car model")
):
    """
    Endpoint to upload a car image and get damage estimation.
    Optionally specify car name and model for more precise analysis.
    """
    try:
        # Save the uploaded file locally
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Encode image as base64
        image_b64 = encode_image_to_base64(file_path)

        # # Check image size
        # if len(image_b64) > 180_000:
        #     raise HTTPException(status_code=400, detail="Image too large. Max size is 180KB.")

        # Construct prompt with optional car details
        optional_context = ""
        if car_name:
            optional_context += f"The car is a {car_name}. "
        if car_model:
            optional_context += f"The model year is {car_model}. "

        # Prompt with the image for NVIDIA LLaMA Vision API
        payload = {
            "model": "meta/llama-3.2-90b-vision-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": f'{optional_context}Analyze this car image and provide details on visible damage, repair costs, and car price.ALWAYS REVERIFY IF THE DAMAGES ARE REAL AS THIS IS VERY IMPORTANT '
                               f'Response format:\nCar Name: (e.g., Toyota Corolla)\nModel: (e.g., 2022)\nCar Price: '
                               f'(Estimated base price of the car)\nDamage Description: (e.g., scratches, dents)\n'
                               f'Damage Estimation: (Repair cost in USD)\nTotal Estimated Price: '
                               f'(Car Price - Repair Cost)\n\n<img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 1.0,
            "stream": False
        }

        # Send request to NVIDIA API
        response = requests.post(INVOKE_URL, headers=HEADERS, json=payload)
        logger.info(response)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"API Error: {response.json()}")

        # Extract model response
        model_response = response.json()
        generated_text = model_response["choices"][0]["message"]["content"]
        return generated_text

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/")
async def root():
    return {
        "message": "Car Damage Estimation Vision API is up and running!",
        "endpoints": {
            "/estimate-damage": "POST endpoint for car damage estimation",
            "query_params": {
                "car_name": "Optional car name",
                "car_model": "Optional car model year"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("BACKEND_PORT", 8000))
    )