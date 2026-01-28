# HOPE Model Training Interface - Fixes & Usage Guide

## 🔧 Issues Fixed

### 1. **Backend Server Crash (Network Error)**
**Problem**: The backend was crashing on startup due to pydantic/pydantic-core version incompatibility, causing all API calls (including file uploads) to fail with "Network Error".

**Solution**:
- Upgraded `pydantic` from 2.9.2 to 2.10.6
- Added `pydantic-core==2.27.2` to ensure compatibility
- Updated `/app/backend/requirements.txt` with correct versions
- Backend now starts successfully on port 8001

### 2. **Synthetic Data Creation**
**Problem**: The synthetic data import path was incorrect in server.py.

**Solution**:
- Fixed import path: `sys.path.insert(0, '/app/scripts')` before importing
- Added comprehensive error handling with traceback
- Endpoint `/api/train/create-synthetic` now works correctly
- Generates 1000 training samples with various patterns:
  - Simple sequences
  - Question-Answer pairs
  - Math sequences
  - Narrative patterns

### 3. **Training Module Import Issues**
**Problem**: The `nested_learning` module was not importable, causing training to fail.

**Solution**:
- Added path setup to `/app/train.py`: `sys.path.insert(0, str(Path(__file__).parent / "src"))`
- Created `/app/scripts/train_simple.py` - a wrapper script for easy training execution
- Training now works successfully from both CLI and API

### 4. **Training Backend Integration**
**Problem**: The training API endpoint wasn't actually executing training - just updating status.

**Solution**:
- Implemented `run_training_background()` function to execute training in background tasks
- Added real-time status updates with progress tracking
- Training results are now stored and displayed in the UI
- Checkpoints are automatically saved and listed

### 5. **Frontend Enhancements**
**Added Features**:
- "Create Synthetic Data" button in Training tab (prominent blue button)
- Real-time training status polling (updates every 2 seconds)
- Enhanced status display showing:
  - Training progress percentage
  - Status messages (starting, running, completed, failed)
  - Last training results with checkpoint location
  - Color-coded status panels (blue=running, green=completed, red=failed)
- Automatic checkpoint list refresh after training completes

---

## 🚀 Quick Start Guide

### Step 1: Create Synthetic Training Data
1. Open the web interface at `http://localhost:3000`
2. Navigate to the **Training** tab
3. Click the blue **"✨ Create Synthetic Data"** button
4. Wait for confirmation - it will create:
   - `/app/data/synthetic/train.jsonl` (1000 samples)
   - `/app/data/synthetic/val.jsonl` (100 samples)
   - `/app/data/synthetic/train.txt` (text format)

### Step 2: Train a Small Model
1. Configuration: Select **"pilot_smoke"** (default - fast, CPU-friendly)
2. Training Steps: Set to **100** (or keep 1000 for longer training)
3. Data Source: Select **"Use existing datasets"**
4. Dataset: Choose **"train.jsonl"** or **"train.txt"** from the dropdown
5. Click **"🚀 Start Training"**
6. Watch the real-time progress bar and status updates

### Step 3: Load and Test the Trained Model
1. Wait for training to complete (status turns green)
2. Navigate to the **Models** tab
3. Select your trained checkpoint from the dropdown (e.g., `step_000100.pt`)
4. Select configuration: **"pilot_smoke"**
5. Click **"🚀 Load Model"**
6. Navigate to the **Chat** tab
7. Type a message and test the model's responses

---

## 📁 File Locations

### Synthetic Data
- Training data: `/app/data/synthetic/train.jsonl`
- Validation data: `/app/data/synthetic/val.jsonl`
- Text format: `/app/data/synthetic/train.txt`

### Checkpoints
Training checkpoints are saved to:
- Pattern: `/app/artifacts/checkpoints/<config_name>_run/step_XXXXXX.pt`
- Example: `/app/artifacts/checkpoints/pilot_smoke_run/step_000100.pt`

### Logs
- Backend logs: `/var/log/supervisor/backend.*.log`
- Frontend logs: `/var/log/supervisor/frontend.*.log`
- Training logs: `/app/logs/pilot_smoke.json` (if using pilot_smoke config)

---

## 🔬 Training Configurations

### pilot_smoke (Recommended for Testing)
- **Layers**: 2
- **Dimension**: 128
- **Heads**: 4
- **Device**: CPU (fast testing)
- **Default Steps**: 10 (increase as needed)
- **Purpose**: Quick smoke test to verify everything works

### mid_stage2
- **Layers**: 12
- **Dimension**: 512
- **Purpose**: Mid-size model for actual experiments
- **Device**: Requires GPU for reasonable speed

### pilot
- **Layers**: 12
- **Dimension**: 768
- **Purpose**: Full pilot configuration
- **Training**: ~3B tokens (long training)

---

## 🛠️ Command Line Training (Alternative)

You can also train directly from the command line:

```bash
# Quick smoke test (10 steps)
cd /app
python scripts/train_simple.py --config-name=pilot_smoke --steps=10 --device=cpu

# Longer training (100 steps)
python scripts/train_simple.py --config-name=pilot_smoke --steps=100 --device=cpu

# Using GPU (if available)
python scripts/train_simple.py --config-name=pilot_smoke --steps=1000 --device=cuda

# Direct training with hydra
python train.py --config-name=pilot_smoke train.steps=100 train.device=cpu
```

---

## 🧪 Testing the System

### Test 1: Synthetic Data Creation
```bash
curl -X POST http://localhost:8001/api/train/create-synthetic
# Should return success with file paths
```

### Test 2: Check Backend Health
```bash
curl http://localhost:8001/api/health
# Should return: {"status": "healthy", ...}
```

### Test 3: List Available Datasets
```bash
curl http://localhost:8001/api/datasets/list
# Should list files in /app/data/ including synthetic data
```

### Test 4: Quick Training Test
```bash
cd /app
python scripts/train_simple.py --config-name=pilot_smoke --steps=10 --device=cpu
# Should complete in ~60 seconds and save checkpoint
```

---

## 📊 Understanding Training Output

During training, you'll see output like:
```
[train] step=1 loss=60.2345 ppl=1.23e15 teach_norm=1.1234
```

- **loss**: Training loss (should generally decrease)
- **ppl**: Perplexity (measure of model uncertainty)
- **teach_norm**: L2 norm of teach signals (HOPE-specific)

High initial perplexity is normal for randomly initialized models.

---

## 🎯 Model Architecture Details

This HOPE model implements Google's Nested Learning architecture with:

### Core Components
1. **HOPE Blocks**: Hierarchical Online Parameter Estimation layers
2. **CMS (Continual Memory System)**: Multi-level memory with different update periods
   - `cms_fast`: Updates every 1 step
   - `cms_mid`: Updates every 4 steps
   - `cms_slow`: Updates every 16 steps
3. **TITAN Memory**: Long-term memory with period-8 updates
4. **Fast State**: In-context learning without modifying meta-parameters
5. **Teach Signals**: δℓ computation for online memory updates

### View Model Details
Navigate to the **Traits** tab after loading a model to see:
- Architecture variant (hope_attention, hope_selfmod, etc.)
- CMS levels and update periods
- Self-modifying features (if enabled)
- Dimensions (vocab size, embedding dim, layers, heads)
- Fast state status
- Tensor invariants verification

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check logs
tail -n 50 /var/log/supervisor/backend.err.log

# Restart backend
sudo supervisorctl restart backend
```

### Training fails
```bash
# Check if synthetic data exists
ls -lh /app/data/synthetic/

# Verify training script works
cd /app && python scripts/train_simple.py --config-name=pilot_smoke --steps=10 --device=cpu
```

### Model won't load
- Ensure checkpoint path and config match
- Check that checkpoint file exists: `ls -lh /app/artifacts/checkpoints/`
- Verify config file is compatible with checkpoint

### Network errors in UI
- Check backend is running: `sudo supervisorctl status backend`
- Verify backend health: `curl http://localhost:8001/api/health`
- Check CORS settings in backend are correct

---

## 📝 Next Steps

1. **Experiment with Training**:
   - Try different step counts (100, 500, 1000)
   - Experiment with different configs (mid_smoke, pilot)
   - Monitor loss curves

2. **Test In-Context Learning**:
   - Load a trained model
   - Navigate to Traits tab
   - Click "Test Memorization" to verify teach signals work

3. **Scale Up**:
   - Use GPU: Change device to `cuda` in training config
   - Increase model size: Try `mid_stage2` config
   - Add more training data: Create larger synthetic datasets or use real data

4. **Evaluate Performance**:
   - Use evaluation scripts in `/app/scripts/eval/`
   - Run zero-shot evaluations
   - Test continual learning capabilities

---

## ✅ Verification Checklist

- [x] Backend server starts without errors
- [x] Synthetic data creation works
- [x] Training completes successfully
- [x] Checkpoints are saved correctly
- [x] Models can be loaded from checkpoints
- [x] Chat interface works with loaded models
- [x] Traits panel displays model architecture
- [x] Training status updates in real-time
- [x] File upload functionality works
- [x] All UI tabs are functional

---

## 📚 Additional Resources

- Main README: `/app/README.md`
- Training guide: `/app/docs/guide.md`
- Paper compliance: `/app/docs/PAPER_COMPLIANCE.md`
- Scaling guide: `/app/docs/FSDP_SCALING_GUIDE.md`
- Config examples: `/app/configs/`

---

**Status**: ✅ All systems operational - ready for training and experimentation!
