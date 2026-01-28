#!/bin/bash

# HOPE Model Interface - Quick Start Script

echo "========================================"
echo "🧠 HOPE Model Interface - Quick Start"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if services are running
echo -e "${BLUE}📊 Checking services status...${NC}"
sudo supervisorctl status | grep -E "backend|frontend"
echo ""

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p /app/artifacts/checkpoints
mkdir -p /app/artifacts/examples
mkdir -p /app/data/uploads
mkdir -p /app/data/datasets
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Check backend health
echo -e "${BLUE}🔍 Checking backend health...${NC}"
HEALTH=$(curl -s http://localhost:8001/api/health)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Backend is running${NC}"
    echo "$HEALTH" | python -m json.tool
else
    echo -e "${YELLOW}⚠️  Backend is not responding${NC}"
    echo "Starting backend..."
    sudo supervisorctl restart backend
fi
echo ""

# Check frontend
echo -e "${BLUE}🔍 Checking frontend...${NC}"
FRONTEND=$(curl -s http://localhost:3000 -o /dev/null -w '%{http_code}')
if [ "$FRONTEND" = "200" ]; then
    echo -e "${GREEN}✅ Frontend is running${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend is not responding${NC}"
    echo "Starting frontend..."
    sudo supervisorctl restart frontend
fi
echo ""

# Display access information
echo "========================================"
echo -e "${GREEN}🎉 HOPE Model Interface is Ready!${NC}"
echo "========================================"
echo ""
echo -e "${BLUE}🌐 Access URLs:${NC}"
echo "   Frontend UI:     http://localhost:3000"
echo "   Backend API:     http://localhost:8001"
echo "   API Docs:        http://localhost:8001/docs"
echo ""
echo -e "${BLUE}📁 Important Directories:${NC}"
echo "   Checkpoints:     /app/artifacts/checkpoints/"
echo "   Training Data:   /app/data/"
echo "   Uploads:         /app/data/uploads/"
echo "   Configs:         /app/configs/"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "   Interface Guide: /app/INTERFACE_README.md"
echo "   Colab Notebook:  /app/notebooks/hope_model_training_inference.ipynb"
echo "   Main README:     /app/README.md"
echo ""
echo -e "${BLUE}🛠️  Useful Commands:${NC}"
echo "   Restart all:     sudo supervisorctl restart all"
echo "   View logs:       tail -f /var/log/supervisor/backend.out.log"
echo "   Stop all:        sudo supervisorctl stop all"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "   1. Place model checkpoints in /app/artifacts/checkpoints/"
echo "   2. Open http://localhost:3000 in your browser"
echo "   3. Load a model from the Models tab"
echo "   4. Start chatting or training!"
echo ""
echo "========================================"
echo -e "${GREEN}Happy training! 🚀${NC}"
echo "========================================"
