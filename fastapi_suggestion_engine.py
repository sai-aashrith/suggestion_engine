from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import json
import os
from openai import OpenAI

app = FastAPI(title="Field Suggester API")

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Field(BaseModel):
    label: str
    type: str

class SuggestFieldsRequest(BaseModel):
    existing_fields: List[Field]

class SuggestFieldsResponse(BaseModel):
    suggested_fields: Dict[str, Field]

def openai_api_call(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests form fields."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

def suggest_fields(existing_fields: List[Field]) -> Dict[str, Field]:
    fields_str = json.dumps([field.dict() for field in existing_fields])
    
    prompt = f"""Given the following existing form fields:
    {fields_str}
    Suggest 5 relevant additional fields that would complement these existing fields.
    Return the suggestions as a JSON object where each key is the field name and the value is an object with 'label' and 'type' properties.
    Ensure the output is valid JSON format."""

    llm_response = openai_api_call(prompt)

    try:    
        suggested_fields = json.loads(llm_response)
        if len(suggested_fields) > 5:
            suggested_fields = dict(list(suggested_fields.items())[:5])
        elif len(suggested_fields) < 5:
            for i in range(len(suggested_fields), 5):
                suggested_fields[f"additional_field_{i+1}"] = {
                    "label": f"Additional Field {i+1}",
                    "type": "text"
                }
        return {k: Field(**v) for k, v in suggested_fields.items()}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI API")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Field Suggester API</title>
        </head>
        <body>
            <h1>Welcome to the Field Suggester API</h1>
            <p>This API provides endpoints for suggesting and managing form fields.</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><a href="/redoc">/redoc</a> - Alternative API documentation</li>
                <li>/suggest-fields (POST) - Suggest additional fields</li>
                <li>/form-fields (GET) - Retrieve current form fields</li>
                <li>/add-field (POST) - Add a new field to the form</li>
                <li>/clear-fields (DELETE) - Clear all fields from the form</li>
            </ul>
        </body>
    </html>
    """

@app.post("/suggest-fields", response_model=SuggestFieldsResponse)
async def api_suggest_fields(request: SuggestFieldsRequest):
    suggested_fields = suggest_fields(request.existing_fields)
    return SuggestFieldsResponse(suggested_fields=suggested_fields)

# In-memory storage for the form fields
form_fields: List[Field] = []

@app.get("/form-fields", response_model=List[Field])
async def get_form_fields():
    return form_fields

@app.post("/add-field")
async def add_field(field: Field):
    form_fields.append(field)
    return {"message": f"Added field: {field.label}"}

@app.delete("/clear-fields")
async def clear_fields():
    form_fields.clear()
    return {"message": "All fields cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)