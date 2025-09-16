# Google Cloud Compute Engine Deployment Guide

## Prerequisites

1. **Google Cloud Project** with Compute Engine API enabled
2. **VM Instance** running Ubuntu 20.04+ or similar
3. **Firewall Rules** configured for ports 8001 and 3000
4. **API Keys** for required services

## Quick Start

### 1. Connect to your VM
```bash
gcloud compute ssh your-vm-name --zone=your-zone
```

### 2. Clone the repository
```bash
git clone https://github.com/zg915/document_chunk.git
cd document_chunk
git checkout webhook
```

### 3. Run the deployment script
```bash
chmod +x deploy-gcp.sh
./deploy-gcp.sh
```

## Manual Deployment

### 1. Install Docker and Docker Compose
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Configure Environment Variables
```bash
cp env.example .env
nano .env
```

Required variables:
```bash
MARKER_API_KEY=your_datalab_api_key
MARKER_API_URL=https://www.datalab.to/api/v1/marker
USE_LOCAL_MARKER=false
WEAVIATE_API_KEY=your_weaviate_api_key
WEAVIATE_URL=your_weaviate_url
GOOGLE_API_KEY=your_google_api_key
WEBHOOK_URL=http://your-vm-external-ip:3000/webhook
```

### 3. Deploy Services
```bash
# Build and start services
docker-compose -f docker-compose.gcp.yml up -d

# Check status
docker-compose -f docker-compose.gcp.yml ps

# View logs
docker-compose -f docker-compose.gcp.yml logs -f
```

## Firewall Configuration

### Create Firewall Rules
```bash
# Allow API server port
gcloud compute firewall-rules create allow-api-server \
    --allow tcp:8001 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow API server access"

# Allow webhook port
gcloud compute firewall-rules create allow-webhook \
    --allow tcp:3000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow webhook access"
```

### Or via GCP Console:
1. Go to VPC Network → Firewall
2. Create firewall rule for port 8001 (API server)
3. Create firewall rule for port 3000 (webhook)

## Service URLs

After deployment, your services will be available at:

- **API Server**: `http://YOUR_VM_EXTERNAL_IP:8001`
- **Webhook**: `http://YOUR_VM_EXTERNAL_IP:3000/webhook`
- **API Health**: `http://YOUR_VM_EXTERNAL_IP:8001/health`
- **Webhook Health**: `http://YOUR_VM_EXTERNAL_IP:3000/`

## Testing the Deployment

### Test API Health
```bash
curl http://YOUR_VM_EXTERNAL_IP:8001/health
```

### Test Webhook
```bash
curl http://YOUR_VM_EXTERNAL_IP:3000/
```

### Test Document Conversion
```bash
curl -X POST "http://YOUR_VM_EXTERNAL_IP:8001/convert-doc-to-markdown-webhook?webhook_url=http://YOUR_VM_EXTERNAL_IP:3000/webhook" \
  -F "file=@your-document.pdf"
```

## Management Commands

### View Logs
```bash
# All services
docker-compose -f docker-compose.gcp.yml logs -f

# Specific service
docker-compose -f docker-compose.gcp.yml logs -f document-api
docker-compose -f docker-compose.gcp.yml logs -f webhook
```

### Restart Services
```bash
# Restart all
docker-compose -f docker-compose.gcp.yml restart

# Restart specific service
docker-compose -f docker-compose.gcp.yml restart document-api
```

### Stop Services
```bash
docker-compose -f docker-compose.gcp.yml down
```

### Update Services
```bash
# Pull latest changes
git pull origin webhook

# Rebuild and restart
docker-compose -f docker-compose.gcp.yml down
docker-compose -f docker-compose.gcp.yml build --no-cache
docker-compose -f docker-compose.gcp.yml up -d
```

## Monitoring

### Check Service Status
```bash
docker-compose -f docker-compose.gcp.yml ps
```

### Check Resource Usage
```bash
docker stats
```

### Check Disk Space
```bash
df -h
```

## Troubleshooting

### Common Issues:

1. **Port Already in Use**
   ```bash
   sudo netstat -tulpn | grep :8001
   sudo netstat -tulpn | grep :3000
   ```

2. **Permission Denied**
   ```bash
   sudo usermod -aG docker $USER
   # Log out and log back in
   ```

3. **Out of Memory**
   ```bash
   # Check memory usage
   free -h
   # Restart services
   docker-compose -f docker-compose.gcp.yml restart
   ```

4. **Firewall Issues**
   - Check GCP firewall rules
   - Verify VM has external IP
   - Test with `telnet YOUR_VM_IP 8001`

### Logs Location:
- Container logs: `docker-compose -f docker-compose.gcp.yml logs`
- System logs: `/var/log/syslog`

## Security Considerations

1. **Firewall Rules**: Only open necessary ports
2. **API Keys**: Store securely in .env file
3. **HTTPS**: Consider setting up SSL/TLS for production
4. **Updates**: Regularly update system and containers

## Scaling

For production use:
1. Use a load balancer for multiple instances
2. Set up auto-scaling groups
3. Use managed databases (Cloud SQL)
4. Implement proper monitoring and alerting
