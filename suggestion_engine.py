import json
import os
from typing import List, Dict
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def interactive_field_suggester():
    existing_fields = []

    while True:
        print("\nCurrent fields:")
        for i, field in enumerate(existing_fields, 1):
            print(f"{i}. {field['label']} ({field['type']})")

        suggestions = suggest_fields(existing_fields)

        print("\nSuggested fields:")
        for i, (label, field) in enumerate(suggestions.items(), 1):
            print(f"{i}. {field['label']} ({field['type']})")

        choice = input("\nEnter the numbers of the fields you want to add (comma-separated) or 'q' to quit: ")

        if choice.lower() == 'q':
            break

        try:
            chosen_indices = [int(idx.strip()) - 1 for idx in choice.split(',')]
            valid_indices = [idx for idx in chosen_indices if 0 <= idx < len(suggestions)]

            if valid_indices:
                for idx in valid_indices:
                    selected_field = list(suggestions.values())[idx]
                    existing_fields.append(selected_field)
                    print(f"Added {selected_field['label']} to the form.")
            else:
                print("No valid choices were made. Please try again.")
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers or 'q' to quit.")

    print("\nFinal form fields:")
    print(json.dumps(existing_fields, indent=2))

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    else:
        interactive_field_suggester()