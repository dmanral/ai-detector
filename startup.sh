#!/usr/bin/env bash
set -e

echo "Checking model files..."

# Image classifier.
if [ ! -f "./app/models/classifier/model.onnx" ]; then
    echo "Downloading image classifier model..."
    mkdir -p ./app/models/classifier
    aws s3 cp s3://ai-detector-models-dmanral/image-classifier/model.onnx ./app/models/text_classifier/model.onnx

fi

# Landmark predictor.
if [ ! -f "./app/models/shape_predictor_68_face_landmarks.dat" ]; then
    echo "Downloading landmark predictor..."
    aws s3 cp s3://ai-dector-models-dmanral/shape_predictor_68_face_landmarks.dat ./app/models/shape_predictor_68_face_landmarks.dat

fi

# Tokenizer files.
if [ ! -d "./app/models/text_classifier/tokenizer" ]; then
    echo "Downloading tokenizer files..."
    mkdir -p ./app/models/text_classifier/tokenizer
    aws s3 cp s3://ai-detector-models-dmanral/text-classifier/tokenizer ./app/models/text_classifier/tokenizer --recursive
fi

echo "All models ready. Starting server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
