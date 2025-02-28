#!/bin/bash

# Set variables for container and image
IMAGE="mongo:latest"
CONTAINER_NAME="mongo-container"
PORT="27017"

# Check if the Docker image is available
if ! sudo docker images | grep -q "$IMAGE"; then
    echo "Mongo image not found. Pulling the latest image..."
    sudo docker pull "$IMAGE"
else
    echo "Mongo image found. Skipping pull."
fi

# Check if the container is already running
if ! sudo docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" | grep -q "$CONTAINER_NAME"; then
    echo "Starting the Mongo container..."
    sudo docker run -d --name "$CONTAINER_NAME" -p "$PORT:$PORT" "$IMAGE"
else
    echo "Mongo container is already running."
fi

# Check if the container is stopped but exists
if sudo docker ps -a --filter "name=$CONTAINER_NAME" --filter "status=exited" | grep -q "$CONTAINER_NAME"; then
    echo "Mongo container is stopped. Restarting it..."
    sudo docker start "$CONTAINER_NAME"
fi

# Verify that the container is running
if sudo docker ps --filter "name=$CONTAINER_NAME" | grep -q "$CONTAINER_NAME"; then
    echo "Mongo container is running on port $PORT."
else
    echo "Failed to start Mongo container."
fi
