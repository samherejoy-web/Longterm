#!/bin/bash
# Complete setup script for HOPE Model UI with training

set -e  # Exit on error

echo "🚀 HOPE Model Interactive Setup"
echo "================================"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /app/data/synthetic
mkdir -p /app/artifacts/demo
mkdir -p /app/artifacts/checkpoints

# Step 1: Create synthetic training data
echo ""
echo "📝 Step 1: Creating synthetic training data..."
cd /app
python scripts/create_synthetic_data.py

# Step 2: Train demo model
echo ""
echo "🎓 Step 2: Training demo HOPE model (500 steps)..."
echo "⏱️  This will take a few minutes..."
python scripts/quick_train_demo.py \
  --data /app/data/synthetic/train.txt \
  --output /app/artifacts/demo \
  --steps 500 \
  --batch-size 4 \
  --device cpu

# Step 3: Copy final checkpoint to standard location
echo ""
echo "💾 Step 3: Setting up checkpoint..."
cp /app/artifacts/demo/demo_final.pt /app/artifacts/checkpoints/demo_model.pt
cp /app/artifacts/demo/config.json /app/artifacts/demo/model_config.json

# Step 4: Check if services are running
echo ""
echo "🔍 Step 4: Checking services..."

# Check if supervisor is running services
if ! sudo supervisorctl status backend | grep -q RUNNING; then
    echo "⚠️  Backend not running, starting..."
    sudo supervisorctl start backend
fi

if ! sudo supervisorctl status frontend | grep -q RUNNING; then
    echo "⚠️  Frontend not running, starting..."
    sudo supervisorctl start frontend
fi

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start..."
sleep 5

# Check backend health
for i in {1..10}; do
    if curl -s http://localhost:8001/api/health > /dev/null; then
        echo "✅ Backend is healthy!"
        break
    fi
    echo "   Waiting for backend... ($i/10)"
    sleep 2
done

# Check frontend
for i in {1..10}; do
    if curl -s http://localhost:3000 > /dev/null; then
        echo "✅ Frontend is healthy!"
        break
    fi
    echo "   Waiting for frontend... ($i/10)"
    sleep 2
done

echo ""
echo "✨ Setup Complete!"
echo "=================="
echo ""
echo "📊 Model Information:"
echo "  - Checkpoint: /app/artifacts/demo/demo_final.pt"
echo "  - Config: /app/artifacts/demo/config.json"
echo "  - Architecture: HOPE with Self-Modifying Titans"
echo "  - Training Steps: 500"
echo ""
echo "🌐 Access the UI:"
echo "  - Frontend: Open your browser to the application"
echo "  - Backend API: http://localhost:8001"
echo ""
echo "📝 Next Steps:"
echo "  1. Go to the 'Models' tab"
echo "  2. Select 'demo_final.pt' checkpoint"
echo "  3. Select 'model_config.json' config"
echo "  4. Click 'Load Model'"
echo "  5. Go to 'Traits' tab to verify all features"
echo "  6. Chat with the model in the 'Chat' tab"
echo ""
echo "🎉 Happy testing!"
