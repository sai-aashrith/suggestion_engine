from flask import Flask, request, jsonify
import json
import os
from typing import List, Dict
from openai import OpenAI

app = Flask(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variable to store fields
fields_data = []

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
        print(f"Error calling OpenAI API: {e}")
        return "{}"

def suggest_fields(existing_fields: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    fields_str = json.dumps(existing_fields)
    
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
        return suggested_fields
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from OpenAI API")
        return {}

@app.route('/', methods=['GET'])
def home():
    print("Root route accessed")
    return jsonify({
        "message": "Welcome to the Field Suggester API",
        "endpoints": {
            "/suggest": "POST - Suggest additional fields",
            "/add_fields": "POST - Add new fields to existing fields",
            "/interactive_suggest": "POST - Get suggestions for interactive field addition",
            "/interactive_add": "POST - Add fields interactively",
            "/view_fields": "GET - View current fields"
        }
    })

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.json
    existing_fields = data.get('existing_fields', [])
    suggestions = suggest_fields(existing_fields)
    return jsonify(suggestions)

@app.route('/add_fields', methods=['POST'])
def add_fields():
    global fields_data
    data = request.json
    existing_fields = data.get('existing_fields', [])
    new_fields = data.get('new_fields', [])
    
    existing_fields.extend(new_fields)
    fields_data = existing_fields
    
    return jsonify({
        'message': 'Fields added successfully',
        'updated_fields': fields_data
    })

@app.route('/interactive_suggest', methods=['POST'])
def interactive_suggest():
    data = request.json
    existing_fields = data.get('existing_fields', [])
    
    suggestions = suggest_fields(existing_fields)
    
    return jsonify({
        'current_fields': existing_fields,
        'suggested_fields': suggestions
    })

@app.route('/interactive_add', methods=['POST'])
def interactive_add():
    global fields_data
    data = request.json
    existing_fields = data.get('existing_fields', [])
    chosen_indices = data.get('chosen_indices', [])
    suggestions = data.get('suggestions', {})
    
    valid_indices = [idx for idx in chosen_indices if 0 <= idx < len(suggestions)]
    
    for idx in valid_indices:
        selected_field = list(suggestions.values())[idx]
        existing_fields.append(selected_field)
    
    fields_data = existing_fields
    
    return jsonify({
        'message': 'Fields added successfully',
        'updated_fields': fields_data
    })

@app.route('/view_fields', methods=['GET'])
def view_fields():
    global fields_data
    return jsonify({
        'message': 'Current fields',
        'fields': fields_data
    })

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    else:
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)