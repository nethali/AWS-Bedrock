import boto3
import json

# Define the prompt data for the AI model
prompt_text = """
Act as developer and write a poem on AI and ML.
"""

# Initialize the Bedrock client for the bedrock-runtime service
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Define the payload for the model
model_payload = {
    "prompt": prompt_text,
    "maxTokens": 512,              # Maximum number of tokens to generate
    "temperature": 0.8,            # Temperature setting for text generation (controls randomness)
    "topP": 0.8                    # Top-p setting. Use a lower value to ignore less probable options
}

# Convert the payload to a JSON string
payload_json = json.dumps(model_payload)

# Specify the model ID
model_identifier = "ai21.j2-mid-v1"

try:
    # Invoke the model with the given payload
    response = bedrock_client.invoke_model(
        body=payload_json,
        modelId=model_identifier,
        accept="application/json",
        contentType="application/json",
    )

    # Parse the JSON response
    response_body = json.loads(response.get("body").read())

    # Extract the generated text from the response
    response_text = response_body.get("completions")[0].get("data").get("text")

    # Print the generated text
    print(response_text)

except Exception as e:
    print(f"An error occurred while invoking the model: {e}")
