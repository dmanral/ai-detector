#!/usr/bin/env bash
set -e

echo "Checking model files..."

# Image classifier — correct destination path
if [ ! -f "./app/models/classifier/model.onnx" ]; then
    echo "Downloading image classifier model..."
    mkdir -p ./app/models/classifier
    aws s3 cp s3://ai-detector-models-dmanral/image-classifier/model.onnx \
        ./app/models/classifier/model.onnx
fi

# Text classifier
if [ ! -f "./app/models/text_classifier/model.onnx" ]; then
    echo "Downloading text classifier model..."
    mkdir -p ./app/models/text_classifier
    aws s3 cp s3://ai-detector-models-dmanral/text-classifier/model.onnx \
        ./app/models/text_classifier/model.onnx
fi

# Landmark predictor — verify exact S3 key
if [ ! -f "./app/models/shape_predictor_68_face_landmarks.dat" ]; then
    echo "Downloading landmark predictor..."
    aws s3 cp s3://ai-detector-models-dmanral/shape_predictor_68_face_landmarks.dat \
        ./app/models/shape_predictor_68_face_landmarks.dat
fi

# Tokenizer files
if [ ! -d "./app/models/text_classifier/tokenizer" ]; then
    echo "Downloading tokenizer..."
    mkdir -p ./app/models/text_classifier/tokenizer
    aws s3 cp s3://ai-detector-models-dmanral/text-classifier/tokenizer/ \
        ./app/models/text_classifier/tokenizer/ --recursive
fi

echo "All models ready. Starting server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000s