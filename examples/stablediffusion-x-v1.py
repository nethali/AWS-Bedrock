import boto3
import json
import base64
import os

# Define the prompt data for the AI model
prompt_text = """
Provide me a Full HD image of a jungle, with full of flowers.
"""

# Define the prompt template for the model
prompt_template = [{"text": prompt_text, "weight": 1}]

# Initialize the Bedrock client for the bedrock-runtime service
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Define the payload for the model
model_payload = {
    "text_prompts": prompt_template,
    "cfg_scale": 10,
    "seed": 0,
    "steps": 50,
    "width": 16*64,
    "height": 9*64
}

# Convert the payload to a JSON string
payload_json = json.dumps(model_payload)

# Specify the model ID
model_identifier = "stability.stable-diffusion-xl-v1"

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

    # Print the response body for debugging or analysis
    print(response_body)

    # Extract the artifact (generated image) from the response
    artifact = response_body.get("artifacts")[0]
    image_encoded = artifact.get("base64").encode("utf-8")
    image_bytes = base64.b64decode(image_encoded)

    # Save the image to a file in the output directory
    output_dir = "pictures"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/generated-img.png"
    with open(file_name, "wb") as f:
        f.write(image_bytes)

    print(f"Generated image saved to: {file_name}")

except Exception as e:
    print(f"An error occurred while invoking the model: {e}")

