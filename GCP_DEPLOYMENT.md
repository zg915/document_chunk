# Google Cloud Deployment

## Setup

1. **Install Docker**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

2. **Install Docker Compose**
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

3. **Clone Repository**
```bash
git clone https://github.com/zg915/document_chunk.git
cd document_chunk
git checkout webhook
```

4. **Configure Environment**
```bash
cp env.example .env
nano .env
```

5. **Deploy**
```bash
docker-compose -f docker-compose.gcp.yml up -d
```

## Firewall
Open ports 8001 and 3000 in GCP Console

## URLs
- API: `http://YOUR_VM_IP:8001`
- Webhook: `http://YOUR_VM_IP:3000/webhook`
