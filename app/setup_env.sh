#!/bin/bash

# S3 Configuration
export S3_BUCKET_NAME="your-s3-bucket-name"  # Replace with your actual S3 bucket name
export AWS_ACCESS_KEY_ID="your-access-key"   # Replace with your AWS access key
export AWS_SECRET_ACCESS_KEY="your-secret-key"  # Replace with your AWS secret key
export AWS_DEFAULT_REGION="your-region"  # Replace with your AWS region (e.g., us-east-1)

# Print confirmation
echo "Environment variables set:"
echo "S3_BUCKET_NAME: $S3_BUCKET_NAME"
echo "AWS_DEFAULT_REGION: $AWS_DEFAULT_REGION"
echo "AWS Access Key and Secret Key have been set" 