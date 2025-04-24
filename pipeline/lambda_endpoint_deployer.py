# Code for Step 7: Deploy the Model so it can be used in production.


import boto3

sm_client = boto3.client("sagemaker")

def lambda_handler(event, context):
    endpoint_name = event["endpoint_name"]
    model_name = event["model_name"] 
    instance_type = event["instance_type"]
    
    
    # Create Endpoint Configuration
    endpoint_config_name = f"{endpoint_name}-config"
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": instance_type,
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            },
        ],
        # Set up Data Capture (could be used for model monitoring if desired)
        DataCaptureConfig={
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": "s3://cs401finalpipelineprocessingdata/datacapture/",
        "CaptureOptions": [
            {"CaptureMode": "Input"},
            {"CaptureMode": "Output"}
        ],
        "CaptureContentTypeHeader": {
            "CsvContentTypes": ["text/csv"],
            "JsonContentTypes": ["application/json"]
            }
        }
    )
    
    print(f"create_endpoint_config_response: {create_endpoint_config_response}")
    
    # Check if an endpoint exists. If no - Create new endpoint, if yes - Update existing endpoint
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print("Updating existing endpoint")
        sm_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
    except:
        print("Creating new endpoint")
        sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
    
    return {"statusCode": 200, "body": "Endpoint deployment successful"}