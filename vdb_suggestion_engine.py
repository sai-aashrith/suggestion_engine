from flask import Flask, request, jsonify, send_from_directory
import json
import os
from typing import List, Dict
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Create or get the collection
field_collection = chroma_client.get_or_create_collection(
    name="form_fields",
    embedding_function=openai_ef
)

# Global variable to store fields
fields_data = []

def openai_api_call(prompt: str) -> str:
    """Make a call to OpenAI API for field suggestions."""
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

def add_field_to_chromadb(field_name: str, field_type: str, metadata: str) -> None:
    """Add a single field to ChromaDB."""
    try:
        # Create a unique ID for the field
        field_id = f"{field_name}_{field_type}".lower().replace(" ", "_")
        
        # Combine field information into a searchable text
        field_text = f"Field name: {field_name}. Field type: {field_type}. Description: {metadata}"
        
        # Store the field in ChromaDB
        field_collection.add(
            ids=[field_id],
            documents=[field_text],
            metadatas=[{
                "field_name": field_name,
                "field_type": field_type,
                "metadata": metadata
            }]
        )
    except Exception as e:
        print(f"Error adding field to ChromaDB: {e}")
        raise

def query_similar_fields(field_description: str, n_results: int = 5) -> List[Dict]:
    """Query ChromaDB for similar fields based on description."""
    try:
        results = field_collection.query(
            query_texts=[field_description],
            n_results=n_results
        )
        
        return [
            {
                "field_name": meta["field_name"],
                "field_type": meta["field_type"],
                "metadata": meta["metadata"]
            }
            for meta in results["metadatas"][0]
        ]
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def suggest_fields(existing_fields: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Suggest fields based on existing fields using both ChromaDB and OpenAI."""
    try:
        # Create a description of existing fields
        fields_description = " ".join([
            f"Field: {field.get('label', '')}, Type: {field.get('type', '')}"
            for field in existing_fields
        ])
        
        # Get similar fields from ChromaDB
        similar_fields = query_similar_fields(fields_description)
        
        # Use OpenAI to generate suggestions
        prompt = f"""Given the following existing form fields:
        {json.dumps(existing_fields)}
        
        And these similar fields from our database:
        {json.dumps(similar_fields)}
        
        Suggest 5 relevant additional fields that would complement these existing fields.
        Return the suggestions as a JSON object where each key is the field name and the value is an object with 'label' and 'type' properties.
        Ensure the output is valid JSON format."""

        llm_response = openai_api_call(prompt)
        
        suggested_fields = json.loads(llm_response)
        if len(suggested_fields) > 5:
            suggested_fields = dict(list(suggested_fields.items())[:5])
        return suggested_fields
    except Exception as e:
        print(f"Error in suggest_fields: {e}")
        return {}

# Root route and basic error handlers
@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Welcome to the Field Suggester API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This help message",
            "POST /suggest": "Get field suggestions",
            "POST /add_to_db": "Add new field to database",
            "POST /query_similar": "Query similar fields",
            "POST /interactive_suggest": "Get interactive suggestions",
            "POST /interactive_add": "Add fields interactively",
            "GET /view_fields": "View all current fields"
        }
    })

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/add_to_db', methods=['POST'])
def add_to_db():
    """Add a new field to the ChromaDB database."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        field_name = data.get('field_name')
        field_type = data.get('field_type')
        metadata = data.get('metadata')
        
        if not all([field_name, field_type, metadata]):
            return jsonify({
                'error': 'Missing required fields. Please provide field_name, field_type, and metadata.'
            }), 400
        
        add_field_to_chromadb(field_name, field_type, metadata)
        return jsonify({
            'message': 'Field added to database successfully',
            'field': {
                'name': field_name,
                'type': field_type,
                'metadata': metadata
            }
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to add field to database: {str(e)}'
        }), 500

@app.route('/query_similar', methods=['POST'])
def query_similar():
    """Query for similar fields based on description."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        description = data.get('description')
        n_results = data.get('n_results', 5)
        
        if not description:
            return jsonify({
                'error': 'Missing description in request'
            }), 400
        
        similar_fields = query_similar_fields(description, n_results)
        return jsonify({
            'similar_fields': similar_fields
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to query similar fields: {str(e)}'
        }), 500

@app.route('/suggest', methods=['POST'])
def suggest():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        existing_fields = data.get('existing_fields', [])
        suggestions = suggest_fields(existing_fields)
        return jsonify(suggestions)
    except Exception as e:
        return jsonify({
            'error': f'Failed to generate suggestions: {str(e)}'
        }), 500

@app.route('/interactive_suggest', methods=['POST'])
def interactive_suggest():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        existing_fields = data.get('existing_fields', [])
        suggestions = suggest_fields(existing_fields)
        
        return jsonify({
            'current_fields': existing_fields,
            'suggested_fields': suggestions
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to generate interactive suggestions: {str(e)}'
        }), 500

@app.route('/interactive_add', methods=['POST'])
def interactive_add():
    global fields_data
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
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
    except Exception as e:
        return jsonify({
            'error': f'Failed to add fields: {str(e)}'
        }), 500

@app.route('/view_fields', methods=['GET'])
def view_fields():
    global fields_data
    return jsonify({
        'message': 'Current fields',
        'fields': fields_data
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "message": "The requested resource was not found",
        "error": str(e)
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "status": "error",
        "message": "An internal server error occurred",
        "error": str(e)
    }), 500

def seed_initial_data():
    """Seed the database with some initial field examples."""
    initial_fields = [
        {
            "field_name": "first_name",
            "field_type": "text",
            "metadata": "Text field containing user's first name, used for personal identification"
        },
        {
            "field_name": "last_name",
            "field_type": "text",
            "metadata": "Text field containing user's last name, used for personal identification"
        },
        {
            "field_name": "email",
            "field_type": "email",
            "metadata": "Email field for user contact information and account authentication"
        },
        {
            "field_name": "birth_date",
            "field_type": "date",
            "metadata": "Date field for user's date of birth, used for age verification and demographics"
        }
    ]
    
    for field in initial_fields:
        add_field_to_chromadb(
            field["field_name"],
            field["field_type"],
            field["metadata"]
        )

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    else:
        print("Starting Flask server...")
        # Uncomment to seed initial data
        seed_initial_data()
        app.run(debug=True, host='0.0.0.0', port=5000)