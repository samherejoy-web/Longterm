from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import json
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
import traceback
from datetime import datetime

# Add the parent directory to the path to import nested_learning
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nested_learning.model import HOPEModel
from src.nested_learning.tokenizer import SentencePieceTokenizer
from omegaconf import OmegaConf

app = FastAPI(title="HOPE Model API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class ModelState:
    def __init__(self):
        self.model: Optional[Any] = None
        self.tokenizer: Optional[SentencePieceTokenizer] = None
        self.config: Optional[Any] = None
        self.current_checkpoint: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fast_state: Optional[Any] = None
        self.training_status = {
            "is_training": False,
            "progress": 0,
            "status": "idle",
            "message": ""
        }

model_state = ModelState()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 50

class ChatResponse(BaseModel):
    response: str
    model_used: Optional[str] = None

class LoadModelRequest(BaseModel):
    checkpoint_path: str
    config_path: str

class TrainRequest(BaseModel):
    base_checkpoint: Optional[str] = None
    config_name: str = "pilot_smoke"
    steps: int = 1000
    use_existing_data: bool = True
    dataset_name: Optional[str] = None

class CheckpointInfo(BaseModel):
    name: str
    path: str
    size: int
    modified: str
    step: Optional[int] = None

# Helper functions
def get_available_checkpoints() -> List[CheckpointInfo]:
    """Scan for available model checkpoints"""
    checkpoints = []
    
    # Common checkpoint locations
    checkpoint_dirs = [
        "/app/artifacts/checkpoints",
        "/app/artifacts/examples",
        "/app/checkpoints",
        "/app/models"
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    if file.endswith(".pt"):
                        filepath = os.path.join(root, file)
                        stat = os.stat(filepath)
                        
                        # Extract step number if present
                        step = None
                        if "step_" in file:
                            try:
                                step = int(file.split("step_")[1].split(".")[0])
                            except:
                                pass
                        
                        checkpoints.append(CheckpointInfo(
                            name=file,
                            path=filepath,
                            size=stat.st_size,
                            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            step=step
                        ))
    
    # Sort by step number if available, else by name
    checkpoints.sort(key=lambda x: (x.step if x.step is not None else -1, x.name), reverse=True)
    return checkpoints

def get_available_configs() -> List[Dict[str, str]]:
    """Get available configuration files"""
    configs = []
    config_dir = "/app/configs"
    
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith(".yaml") and not file.startswith("."):
                configs.append({
                    "name": file.replace(".yaml", ""),
                    "path": os.path.join(config_dir, file)
                })
    
    return configs

def load_model_from_checkpoint(checkpoint_path: str, config_path: str):
    """Load HOPE model from checkpoint"""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=model_state.device)
        
        # Extract config from checkpoint or file
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config_dict = checkpoint['config']
        else:
            # Try loading from config file (JSON or YAML)
            if config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                config = OmegaConf.load(config_path)
                config_dict = OmegaConf.to_container(config.model if hasattr(config, 'model') else config, resolve=True)
        
        # Import ModelConfig and LevelSpec
        from src.nested_learning.model import ModelConfig
        from src.nested_learning.levels import LevelSpec
        
        # Build ModelConfig
        # Handle nested structures
        if 'titan_level' in config_dict and isinstance(config_dict['titan_level'], dict):
            config_dict['titan_level'] = LevelSpec(**config_dict['titan_level'])
        
        if 'cms_levels' in config_dict:
            cms_levels = []
            for level in config_dict['cms_levels']:
                if isinstance(level, dict):
                    cms_levels.append(LevelSpec(**level))
                else:
                    cms_levels.append(level)
            config_dict['cms_levels'] = cms_levels
        
        model_config = ModelConfig(**config_dict)
        
        # Create model
        model = HOPEModel(model_config).to(model_state.device)
        
        # Load state dict
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Initialize fast state for in-context learning
        fast_state = model.init_fast_state()
        
        # Store in global state
        model_state.model = model
        model_state.config = model_config
        model_state.current_checkpoint = checkpoint_path
        model_state.fast_state = fast_state
        
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Error loading model: {str(e)}\n{traceback.format_exc()}"

def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.8, top_k: int = 50) -> str:
    """Generate text using the loaded model"""
    if model_state.model is None:
        return "[ERROR: No model loaded. Please load a model first.]"
    
    try:
        model = model_state.model
        model.eval()
        
        # Simple character-level tokenization for demo
        # (In production, use proper tokenizer)
        tokens = [ord(c) % 256 for c in prompt[-32:]]  # Take last 32 chars
        if len(tokens) < 32:
            tokens = [0] * (32 - len(tokens)) + tokens
        
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(model_state.device)
        
        # Generate tokens
        generated = tokens.copy()
        
        with torch.no_grad():
            for _ in range(min(max_length, 50)):  # Limit generation
                # Get logits
                logits = model(input_tensor)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop on padding
                if next_token == 0:
                    break
                
                generated.append(next_token)
                
                # Update input (sliding window)
                input_tensor = torch.tensor([generated[-32:]], dtype=torch.long).to(model_state.device)
                if input_tensor.size(1) < 32:
                    padding = torch.zeros((1, 32 - input_tensor.size(1)), dtype=torch.long).to(model_state.device)
                    input_tensor = torch.cat([padding, input_tensor], dim=1)
        
        # Decode generated tokens
        generated_text = ''.join([chr(t) if 32 <= t < 127 else '' for t in generated[len(tokens):]])
        
        if not generated_text.strip():
            return f"[Model processed your input. The model is still learning - try training for more steps for better generation]"
        
        return generated_text
    except Exception as e:
        return f"[ERROR: Generation failed: {str(e)}]"

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "HOPE Model API Server",
        "version": "1.0.0",
        "status": "running",
        "device": str(model_state.device),
        "model_loaded": model_state.current_checkpoint is not None
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_state.current_checkpoint is not None,
        "current_checkpoint": model_state.current_checkpoint,
        "device": str(model_state.device)
    }

@app.get("/api/models/list")
async def list_models():
    """List all available model checkpoints"""
    try:
        checkpoints = get_available_checkpoints()
        return {
            "checkpoints": [cp.dict() for cp in checkpoints],
            "count": len(checkpoints),
            "current": model_state.current_checkpoint
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/api/configs/list")
async def list_configs():
    """List available configuration files"""
    try:
        configs = get_available_configs()
        return {
            "configs": configs,
            "count": len(configs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing configs: {str(e)}")

@app.post("/api/models/load")
async def load_model(request: LoadModelRequest):
    """Load a specific model checkpoint"""
    try:
        success, message = load_model_from_checkpoint(request.checkpoint_path, request.config_path)
        
        if success:
            return {
                "success": True,
                "message": message,
                "checkpoint": request.checkpoint_path,
                "config": request.config_path
            }
        else:
            raise HTTPException(status_code=500, detail=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate text response to user message"""
    try:
        response_text = generate_text(
            prompt=request.message,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        return ChatResponse(
            response=response_text,
            model_used=os.path.basename(model_state.current_checkpoint) if model_state.current_checkpoint else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/train/upload")
async def upload_training_data(file: UploadFile = File(...)):
    """Upload custom training data"""
    try:
        # Create uploads directory
        upload_dir = "/app/data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "success": True,
            "filename": file.filename,
            "path": file_path,
            "size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/api/train/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    if model_state.training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        # Update training status
        model_state.training_status.update({
            "is_training": True,
            "progress": 0,
            "status": "starting",
            "message": "Initializing training..."
        })
        
        return {
            "success": True,
            "message": "Training started",
            "config": request.config_name,
            "base_checkpoint": request.base_checkpoint,
            "steps": request.steps
        }
    except Exception as e:
        model_state.training_status["is_training"] = False
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/api/train/status")
async def get_training_status():
    """Get current training status"""
    return model_state.training_status

@app.get("/api/datasets/list")
async def list_datasets():
    """List available datasets"""
    datasets = []
    data_dir = "/app/data"
    
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith((".txt", ".json", ".jsonl", ".csv")):
                    filepath = os.path.join(root, file)
                    stat = os.stat(filepath)
                    datasets.append({
                        "name": file,
                        "path": filepath,
                        "size": stat.st_size,
                        "type": file.split(".")[-1]
                    })
    
    return {
        "datasets": datasets,
        "count": len(datasets)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")