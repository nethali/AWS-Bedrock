import boto3
import json

# Define the prompt data for the AI model
prompt_text = """
Act as baby and write a poem on Generative AI.
"""

# Initialize the Bedrock client for the bedrock-runtime service
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Define the payload for the model
model_payload = {
    "prompt": "[INST]" + prompt_text + "[/INST]",  # Encapsulate the prompt in instruction tags
    "max_gen_len": 512,                            # Maximum length of the generated text
    "temperature": 0.5,                            # Temperature setting for text generation (controls randomness)
    "top_p": 0.9                                   # Top-p setting for nucleus sampling
}

# Convert the payload to a JSON string
payload_json = json.dumps(model_payload)

# Specify the model ID
model_identifier = "meta.llama2-70b-chat-v1"

try:
    # Invoke the model with the given payload
    response = bedrock_client.invoke_model(
        body=payload_json,
        modelId=model_identifier,
        accept="application/json",
        contentType="application/json"
    )

    # Parse the JSON response
    response_data = json.loads(response.get("body").read())

    # Extract the generated text from the response
    generated_text = response_data['generation']

    # Print the generated text
    print(generated_text)

except Exception as e:
    print(f"An error occurred while invoking the model: {e}")

