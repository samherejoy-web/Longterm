# 🧠 HOPE Model - Web Interface

This is a full-stack web application for training and interacting with HOPE (Nested Learning) models.

## 🚀 Features

### 💬 Chat Interface
- Interactive chat with loaded HOPE models
- Real-time text generation
- Message history
- Model status indicator

### 🤖 Model Management
- Browse available model checkpoints
- Load models with configuration files
- View checkpoint metadata (size, step, date)
- Select from multiple model configurations

### 🎓 Training Dashboard
- Continue training from existing checkpoints
- Train new models from scratch
- Two data input options:
  - Upload custom training data (.txt, .json, .jsonl, .csv)
  - Use existing datasets from the project
- Configure training steps and parameters
- Monitor training progress in real-time

## 📁 Project Structure

```
/app/
├── backend/              # FastAPI backend server
│   ├── server.py        # Main API server
│   ├── requirements.txt # Python dependencies
│   └── .env            # Backend configuration
│
├── frontend/            # React frontend application
│   ├── src/
│   │   ├── App.js      # Main React component
│   │   ├── App.css     # Styling
│   │   ├── index.js    # React entry point
│   │   └── index.css   # Global styles
│   ├── public/
│   │   └── index.html  # HTML template
│   ├── package.json    # Node.js dependencies
│   └── .env           # Frontend configuration
│
├── notebooks/           # Google Colab notebooks
│   └── hope_model_training_inference.ipynb
│
└── src/nested_learning/ # Original HOPE model implementation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- Node.js 18+ & Yarn
- PyTorch 2.9.0
- CUDA-capable GPU (optional, for faster training)

### Backend Setup

1. Install Python dependencies:
```bash
cd /app/backend
pip install -r requirements.txt
```

2. Environment variables are already configured in `/app/backend/.env`:
```
PORT=8001
HOST=0.0.0.0
CORS_ORIGINS=*
```

### Frontend Setup

1. Install Node.js dependencies (already done):
```bash
cd /app/frontend
yarn install
```

2. Environment variables are configured in `/app/frontend/.env`:
```
REACT_APP_BACKEND_URL=http://localhost:8001
PORT=3000
```

## 🚀 Running the Application

### Option 1: Using Supervisor (Recommended)

Start both frontend and backend together:
```bash
sudo supervisorctl restart all
```

Check status:
```bash
sudo supervisorctl status
```

View logs:
```bash
# Backend logs
tail -f /var/log/supervisor/backend.out.log
tail -f /var/log/supervisor/backend.err.log

# Frontend logs
tail -f /var/log/supervisor/frontend.out.log
tail -f /var/log/supervisor/frontend.err.log
```

### Option 2: Manual Start

**Backend:**
```bash
cd /app/backend
python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

**Frontend:**
```bash
cd /app/frontend
yarn start
```

## 🌐 Accessing the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## 📚 API Endpoints

### Health & Status
- `GET /` - API information
- `GET /api/health` - Health check

### Models
- `GET /api/models/list` - List all available checkpoints
- `GET /api/configs/list` - List configuration files
- `POST /api/models/load` - Load a specific checkpoint

### Chat
- `POST /api/chat` - Generate text response

### Training
- `POST /api/train/upload` - Upload training data
- `POST /api/train/start` - Start training
- `GET /api/train/status` - Get training status
- `GET /api/datasets/list` - List available datasets

## 📦 Model Checkpoints

Place your trained model checkpoints in:
```
/app/artifacts/checkpoints/
/app/artifacts/examples/
/app/checkpoints/
/app/models/
```

The application will automatically scan these directories for `.pt` checkpoint files.

## 🎓 Training Data

Place training data files in:
```
/app/data/
```

Supported formats:
- `.txt` - Plain text files
- `.json` - JSON data
- `.jsonl` - JSON Lines format
- `.csv` - CSV files

## 🔧 Configuration Files

HOPE model configurations are located in:
```
/app/configs/
```

Available configurations:
- `pilot.yaml` - Pilot configuration
- `pilot_smoke.yaml` - Smoke test configuration
- `mid.yaml` - Mid-size configuration
- And more...

## 📓 Google Colab Notebook

A comprehensive Colab notebook is available at:
```
/app/notebooks/hope_model_training_inference.ipynb
```

Features:
- ✅ Load models from Google Drive
- ✅ Run inference (text generation)
- ✅ Continue training with new data
- ✅ Save checkpoints back to Drive

### Using the Colab Notebook:

1. Upload the notebook to Google Colab
2. Mount your Google Drive
3. Upload checkpoints to `/content/drive/MyDrive/hope_models/checkpoints/`
4. Upload training data to `/content/drive/MyDrive/hope_models/data/`
5. Run all cells

## 🎨 UI Components

### Chat Interface
- Real-time message streaming
- User and assistant message bubbles
- System notifications
- Model status indicator

### Model Manager
- Checkpoint browser
- Configuration selector
- One-click model loading
- Checkpoint metadata display

### Training Dashboard
- Base checkpoint selection
- Training configuration
- Data source selector (upload or existing)
- File upload interface
- Training progress monitor

## 🔍 Troubleshooting

### Backend not starting
```bash
# Check logs
tail -n 100 /var/log/supervisor/backend.err.log

# Verify Python path
which python
python --version

# Test import
cd /app
python -c "from src.nested_learning.model import HOPEModel"
```

### Frontend not loading
```bash
# Check logs
tail -n 100 /var/log/supervisor/frontend.err.log

# Verify Node.js
node --version
yarn --version

# Clear cache and rebuild
cd /app/frontend
rm -rf node_modules
yarn install
```

### No checkpoints found
```bash
# Create checkpoint directories
mkdir -p /app/artifacts/checkpoints
mkdir -p /app/artifacts/examples

# Copy existing checkpoints
cp /path/to/your/model.pt /app/artifacts/checkpoints/
```

### Training not starting
```bash
# Check if model loaded
curl http://localhost:8001/api/health

# Check training data
ls -lh /app/data/

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚧 Development Status

### ✅ Implemented
- Full-stack web interface
- Model loading and management
- Chat interface (UI ready)
- Training configuration
- File upload system
- Google Colab notebook template

### 🚧 In Progress
- Actual HOPE model inference integration
- Real-time text generation
- Training loop implementation
- Progress monitoring

### 📝 Notes
- The inference and training functions are templates that need to be connected to the actual HOPE model implementation
- The current chat generates placeholder responses
- Training initiation is set up but needs the actual training loop

## 🤝 Contributing

To extend the interface:

1. **Add new API endpoints**: Edit `/app/backend/server.py`
2. **Modify UI**: Edit `/app/frontend/src/App.js`
3. **Update styling**: Edit `/app/frontend/src/App.css`
4. **Add features**: Follow the existing code structure

## 📄 License

This interface is part of the Nested Learning project and follows the same Apache 2.0 license.

## 🎉 Getting Started

1. Start the services:
   ```bash
   sudo supervisorctl restart all
   ```

2. Open your browser to http://localhost:3000

3. Go to the "Models" tab and load a checkpoint

4. Switch to "Chat" tab and start interacting with your model

5. Use the "Training" tab to continue training or train new models

For Google Colab usage, upload the notebook from `/app/notebooks/` and follow the instructions inside.

Happy training! 🚀
