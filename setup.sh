#!/bin/bash
set -e  # exit on error

echo "Check and update system packages..."
sudo apt update

echo "Installing dependencies..."
sudo apt install -y pciutils

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "Installing Python dependencies..."
pip install --upgrade \
    langchain_community \
    langchain-ollama \
    chromadb \
    torch \
    jq \
    rank-bm25 \
    langgraph

echo "Setup complete!"
