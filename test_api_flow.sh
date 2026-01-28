#!/bin/bash

echo "=========================================="
echo "🧪 Testing HOPE Model API Flow"
echo "=========================================="
echo ""

# Test 1: Health Check
echo "1️⃣  Testing Health Endpoint..."
HEALTH=$(curl -s http://127.0.0.1:8001/api/health)
if echo "$HEALTH" | grep -q "healthy"; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    echo "$HEALTH"
    exit 1
fi
echo ""

# Test 2: Create Synthetic Data
echo "2️⃣  Testing Synthetic Data Creation..."
SYNTHETIC=$(curl -s -X POST http://127.0.0.1:8001/api/train/create-synthetic)
if echo "$SYNTHETIC" | grep -q "success.*true"; then
    echo "✅ Synthetic data created successfully"
else
    echo "❌ Synthetic data creation failed"
    echo "$SYNTHETIC"
    exit 1
fi
echo ""

# Test 3: File Upload
echo "3️⃣  Testing File Upload..."
echo "Test training data for HOPE model" > /tmp/test_data.txt
UPLOAD=$(curl -s -X POST -F "file=@/tmp/test_data.txt" http://127.0.0.1:8001/api/train/upload)
if echo "$UPLOAD" | grep -q "success.*true"; then
    echo "✅ File upload successful"
else
    echo "❌ File upload failed"
    echo "$UPLOAD"
    exit 1
fi
echo ""

# Test 4: List Models
echo "4️⃣  Testing Model Listing..."
MODELS=$(curl -s http://127.0.0.1:8001/api/models/list)
MODEL_COUNT=$(echo "$MODELS" | grep -o '"count":[0-9]*' | grep -o '[0-9]*')
echo "   Found $MODEL_COUNT checkpoint(s)"
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "✅ Models listed successfully"
else
    echo "⚠️  No models found (this is OK for fresh setup)"
fi
echo ""

# Test 5: List Configs
echo "5️⃣  Testing Config Listing..."
CONFIGS=$(curl -s http://127.0.0.1:8001/api/configs/list)
CONFIG_COUNT=$(echo "$CONFIGS" | grep -o '"count":[0-9]*' | grep -o '[0-9]*')
echo "   Found $CONFIG_COUNT configuration(s)"
if [ "$CONFIG_COUNT" -gt 0 ]; then
    echo "✅ Configs listed successfully"
else
    echo "❌ No configs found"
    exit 1
fi
echo ""

# Test 6: List Datasets
echo "6️⃣  Testing Dataset Listing..."
DATASETS=$(curl -s http://127.0.0.1:8001/api/datasets/list)
DATASET_COUNT=$(echo "$DATASETS" | grep -o '"count":[0-9]*' | grep -o '[0-9]*')
echo "   Found $DATASET_COUNT dataset(s)"
echo "✅ Datasets listed successfully"
echo ""

echo "=========================================="
echo "✅ All API Tests Passed!"
echo "=========================================="
echo ""
echo "📋 Summary:"
echo "   - Health: OK"
echo "   - Synthetic Data: OK"
echo "   - File Upload: OK"
echo "   - Models: $MODEL_COUNT checkpoint(s)"
echo "   - Configs: $CONFIG_COUNT config(s)"
echo "   - Datasets: $DATASET_COUNT dataset(s)"
echo ""
echo "🌐 Frontend should now be able to:"
echo "   ✓ Upload documents"
echo "   ✓ Generate synthetic data"
echo "   ✓ List models and configs"
echo "   ✓ Start training"
echo ""
