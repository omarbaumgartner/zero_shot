#!/bin/bash

# Set the image name
IMAGE_NAME="zero_shot_object_classification_streamlit"

# Build the Docker image
echo "Building the Docker image..."
docker build -t $IMAGE_NAME .

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Docker image build failed. Exiting."
    exit 1
fi

# Run the Docker container
echo "Running the Docker container..."
docker run -p 81:81 -p 8502:8502 $IMAGE_NAME

# Check if the container run was successful
if [ $? -ne 0 ]; then
    echo "Failed to run the Docker container. Exiting."
    exit 1
fi

echo "Streamlit app is running at https://localhost"

echo "Pruning system"
docker system prune -af

# Step 4: Remove unused volumes
echo "Removing unused volumes"
docker volume prune -f
