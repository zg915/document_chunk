#!/bin/bash

# Google Cloud Compute Engine Deployment Script
# Run this script on your GCP VM instance

echo "🚀 Starting deployment on Google Cloud Compute Engine..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null; then
    echo "🔧 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your actual API keys before continuing"
    echo "   Required variables:"
    echo "   - MARKER_API_KEY"
    echo "   - WEAVIATE_API_KEY"
    echo "   - WEAVIATE_URL"
    echo "   - GOOGLE_API_KEY"
    echo ""
    read -p "Press Enter after you've updated the .env file..."
fi

# Set webhook URL to your VM's external IP
EXTERNAL_IP=$(curl -s ifconfig.me)
echo "🌐 Detected external IP: $EXTERNAL_IP"
export WEBHOOK_URL="http://$EXTERNAL_IP:3000/webhook"

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.gcp.yml down
docker-compose -f docker-compose.gcp.yml build --no-cache
docker-compose -f docker-compose.gcp.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose -f docker-compose.gcp.yml ps

# Test API health
echo "🏥 Testing API health..."
curl -f http://localhost:8001/health || echo "❌ API health check failed"

# Test webhook health
echo "🔗 Testing webhook health..."
curl -f http://localhost:3000/ || echo "❌ Webhook health check failed"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Service URLs:"
echo "   API Server: http://$EXTERNAL_IP:8001"
echo "   Webhook: http://$EXTERNAL_IP:3000/webhook"
echo "   API Health: http://$EXTERNAL_IP:8001/health"
echo "   Webhook Health: http://$EXTERNAL_IP:3000/"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose -f docker-compose.gcp.yml logs -f"
echo "   Stop services: docker-compose -f docker-compose.gcp.yml down"
echo "   Restart services: docker-compose -f docker-compose.gcp.yml restart"
echo ""
echo "🔧 Don't forget to:"
echo "   1. Open firewall ports 8001 and 3000 in GCP Console"
echo "   2. Update your datalab webhook URL to: http://$EXTERNAL_IP:3000/webhook"
