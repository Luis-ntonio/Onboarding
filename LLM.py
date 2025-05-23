import json
import boto3

# Calling env variables
import os
from dotenv import load_dotenv

load_dotenv()

bedrock_runtime =  boto3.client( 
                                service_name = 'bedrock-runtime', 
                                region_name='us-east-1', 
                            )     
                       
def claude_body(prompt : str, query : str):
        
    query = [{
        "role": "user",
        "content": query
    }]
    
    return json.dumps({              
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4090,
        "system": prompt,
        "messages": query,
        "temperature": 0.0,
        
    })
    

def embed_body(chunk_message : str):
    return json.dumps({
        'inputText' : chunk_message,
        
    })


# Llamada al LLM
def claude_call( bedrock : boto3.client, 
                user_message : str, 
                query : str,
                model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'):
    
    body = claude_body(user_message, query=query)

    response = bedrock.invoke_model(
        body = body,
        modelId = model_id,
        contentType = 'application/json',
        accept = 'application/json'
    )    

    return json.loads(response['body'].read().decode('utf-8'))



# Llamada al modelo de embedding
def embed_call(bedrock : boto3.client, chunk_message : str):
    
    model_id = "amazon.titan-embed-text-v2:0"
    body = embed_body(chunk_message)

    response = bedrock.invoke_model(
        body = body,
        modelId = model_id,
        contentType = 'application/json',
        accept = 'application/json'        
    )    

    return json.loads(response['body'].read().decode('utf-8'))
 
def call_differences(bedrock : boto3.client, 
                     user_message : str, 
                     query : str,
                     model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'):
    
    body = claude_body(user_message, query=query)

    response = bedrock.invoke_model(
        body = body,
        modelId = model_id,
        contentType = 'application/json',
        accept = 'application/json'
    )    

    return json.loads(response['body'].read().decode('utf-8'))


if __name__ == "__main__":
    print(claude_call(bedrock_runtime, "Hello", "How are you?"))
    print(embed_call(bedrock_runtime, "Hello, how are you?"))