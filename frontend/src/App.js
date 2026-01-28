import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import TraitsPanel from './TraitsPanel';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  // State management
  const [activeTab, setActiveTab] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [checkpoints, setCheckpoints] = useState([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [selectedConfig, setSelectedConfig] = useState('');
  const [configs, setConfigs] = useState([]);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [currentModel, setCurrentModel] = useState('No model loaded');
  
  // Training state
  const [trainingCheckpoint, setTrainingCheckpoint] = useState('');
  const [trainingConfig, setTrainingConfig] = useState('pilot_smoke');
  const [trainingSteps, setTrainingSteps] = useState(1000);
  const [useExistingData, setUseExistingData] = useState(true);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load initial data
  useEffect(() => {
    loadCheckpoints();
    loadConfigs();
    loadDatasets();
    checkHealth();
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // API calls
  const checkHealth = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/health`);
      if (response.data.model_loaded) {
        setModelLoaded(true);
        setCurrentModel(response.data.current_checkpoint);
      }
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const loadCheckpoints = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/models/list`);
      setCheckpoints(response.data.checkpoints);
      if (response.data.current) {
        setCurrentModel(response.data.current);
      }
    } catch (error) {
      console.error('Failed to load checkpoints:', error);
      setMessages(prev => [...prev, {
        type: 'system',
        content: 'Note: No model checkpoints found. You can train a new model or upload existing checkpoints to /app/artifacts/checkpoints/'
      }]);
    }
  };

  const loadConfigs = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/configs/list`);
      setConfigs(response.data.configs);
      if (response.data.configs.length > 0) {
        setSelectedConfig(response.data.configs[0].path);
        setTrainingConfig(response.data.configs[0].name);
      }
    } catch (error) {
      console.error('Failed to load configs:', error);
    }
  };

  const loadDatasets = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/datasets/list`);
      setDatasets(response.data.datasets);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadModel = async () => {
    if (!selectedCheckpoint || !selectedConfig) {
      alert('Please select both a checkpoint and configuration file');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/models/load`, {
        checkpoint_path: selectedCheckpoint,
        config_path: selectedConfig
      });
      
      setModelLoaded(true);
      setCurrentModel(selectedCheckpoint);
      setMessages(prev => [...prev, {
        type: 'system',
        content: `✅ Model loaded successfully: ${selectedCheckpoint.split('/').pop()}`
      }]);
    } catch (error) {
      console.error('Failed to load model:', error);
      setMessages(prev => [...prev, {
        type: 'system',
        content: `❌ Failed to load model: ${error.response?.data?.detail || error.message}`
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/chat`, {
        message: inputMessage,
        max_length: 100,
        temperature: 0.8,
        top_k: 50
      });

      const botMessage = {
        type: 'assistant',
        content: response.data.response,
        model: response.data.model_used,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        type: 'system',
        content: `Error: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/train/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setUploadedFile(response.data);
      alert(`File uploaded successfully: ${response.data.filename}`);
    } catch (error) {
      console.error('Upload error:', error);
      alert(`Upload failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const createSyntheticData = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/train/create-synthetic`);
      
      if (response.data.success) {
        alert(`✅ Synthetic data created successfully!\n\nFiles:\n- Train: ${response.data.files.train_jsonl}\n- Val: ${response.data.files.val_jsonl}\n- Text: ${response.data.files.train_txt}`);
        // Reload datasets
        await loadDatasets();
      }
    } catch (error) {
      console.error('Synthetic data error:', error);
      alert(`❌ Failed to create synthetic data: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const startTraining = async () => {
    if (trainingStatus?.is_training) {
      alert('Training already in progress');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/train/start`, {
        base_checkpoint: trainingCheckpoint || null,
        config_name: trainingConfig,
        steps: trainingSteps,
        use_existing_data: useExistingData,
        dataset_name: selectedDataset || null
      });

      setTrainingStatus({ is_training: true, progress: 0, status: 'running' });
      alert('Training started successfully!');
    } catch (error) {
      console.error('Training error:', error);
      alert(`Training failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="App min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-md border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white" data-testid="app-title">🧠 HOPE Model Interface</h1>
              <p className="text-white/80 text-sm mt-1">Nested Learning - Training & Inference Platform</p>
            </div>
            <div className="text-right">
              <div className="text-white/90 text-sm font-medium" data-testid="model-status">
                {modelLoaded ? '🟢 Model Loaded' : '🔴 No Model'}
              </div>
              <div className="text-white/70 text-xs mt-1" data-testid="current-model">
                {currentModel ? currentModel.split('/').pop() : 'Select a model'}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="flex space-x-2 bg-white/10 backdrop-blur-md rounded-lg p-1">
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-all ${
              activeTab === 'chat'
                ? 'bg-white text-purple-600 shadow-lg'
                : 'text-white hover:bg-white/20'
            }`}
            data-testid="tab-chat"
          >
            💬 Chat
          </button>
          <button
            onClick={() => setActiveTab('models')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-all ${
              activeTab === 'models'
                ? 'bg-white text-purple-600 shadow-lg'
                : 'text-white hover:bg-white/20'
            }`}
            data-testid="tab-models"
          >
            🤖 Models
          </button>
          <button
            onClick={() => setActiveTab('traits')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-all ${
              activeTab === 'traits'
                ? 'bg-white text-purple-600 shadow-lg'
                : 'text-white hover:bg-white/20'
            }`}
            data-testid="tab-traits"
          >
            🧬 Traits
          </button>
          <button
            onClick={() => setActiveTab('train')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-all ${
              activeTab === 'train'
                ? 'bg-white text-purple-600 shadow-lg'
                : 'text-white hover:bg-white/20'
            }`}
            data-testid="tab-train"
          >
            🎓 Training
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-2xl overflow-hidden" data-testid="chat-panel">
            <div className="chat-container p-6 space-y-4 scrollbar-thin" style={{ height: '500px' }}>
              {messages.length === 0 && (
                <div className="text-center text-gray-500 mt-20">
                  <p className="text-xl mb-2">👋 Welcome to HOPE Model Chat!</p>
                  <p className="text-sm">Load a model from the Models tab and start chatting</p>
                </div>
              )}
              
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`message-bubble flex ${
                    msg.type === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                  data-testid={`message-${msg.type}`}
                >
                  <div
                    className={`max-w-3xl px-4 py-3 rounded-2xl ${
                      msg.type === 'user'
                        ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                        : msg.type === 'system'
                        ? 'bg-yellow-100 text-yellow-900 border border-yellow-300'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                    {msg.model && (
                      <p className="text-xs mt-2 opacity-70">Model: {msg.model}</p>
                    )}
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 px-4 py-3 rounded-2xl">
                    <div className="flex items-center space-x-2">
                      <div className="loader"></div>
                      <span className="text-gray-600">Generating response...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input */}
            <div className="border-t bg-gray-50 p-4">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={modelLoaded ? "Type your message..." : "Load a model first..."}
                  disabled={!modelLoaded || isLoading}
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                  data-testid="chat-input"
                />
                <button
                  onClick={sendMessage}
                  disabled={!modelLoaded || isLoading || !inputMessage.trim()}
                  className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-medium hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  data-testid="send-button"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-2xl p-6" data-testid="models-panel">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">📦 Model Manager</h2>
            
            {/* Model Selection */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Checkpoint
                </label>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  data-testid="checkpoint-select"
                >
                  <option value="">-- Select a checkpoint --</option>
                  {checkpoints.map((cp, idx) => (
                    <option key={idx} value={cp.path}>
                      {cp.name} {cp.step ? `(Step ${cp.step})` : ''} - {(cp.size / 1024 / 1024).toFixed(2)} MB
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Configuration
                </label>
                <select
                  value={selectedConfig}
                  onChange={(e) => setSelectedConfig(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  data-testid="config-select"
                >
                  <option value="">-- Select a config --</option>
                  {configs.map((cfg, idx) => (
                    <option key={idx} value={cfg.path}>
                      {cfg.name}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={loadModel}
                disabled={!selectedCheckpoint || !selectedConfig || isLoading}
                className="w-full py-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg font-medium hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                data-testid="load-model-button"
              >
                {isLoading ? 'Loading...' : '🚀 Load Model'}
              </button>
            </div>

            {/* Checkpoint List */}
            <div className="mt-8">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Available Checkpoints</h3>
              {checkpoints.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p>No checkpoints found</p>
                  <p className="text-sm mt-2">Train a new model or place checkpoints in /app/artifacts/checkpoints/</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto scrollbar-thin">
                  {checkpoints.map((cp, idx) => (
                    <div
                      key={idx}
                      className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                      data-testid={`checkpoint-item-${idx}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <p className="font-medium text-gray-900">{cp.name}</p>
                          <p className="text-sm text-gray-500">{cp.path}</p>
                          {cp.step && (
                            <p className="text-xs text-purple-600 mt-1">Training Step: {cp.step}</p>
                          )}
                        </div>
                        <div className="text-right ml-4">
                          <p className="text-sm font-medium text-gray-700">
                            {(cp.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                          <p className="text-xs text-gray-500">
                            {new Date(cp.modified).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Traits Tab */}
        {activeTab === 'traits' && (
          <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-2xl p-6" data-testid="traits-panel">
            <TraitsPanel />
          </div>
        )}

        {/* Training Tab */}
        {activeTab === 'train' && (
          <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-2xl p-6" data-testid="training-panel">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">🎓 Model Training</h2>
            
            <div className="space-y-6">
              {/* Base Checkpoint Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Base Checkpoint (Optional - leave empty to train from scratch)
                </label>
                <select
                  value={trainingCheckpoint}
                  onChange={(e) => setTrainingCheckpoint(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  data-testid="training-checkpoint-select"
                >
                  <option value="">-- Train from scratch --</option>
                  {checkpoints.map((cp, idx) => (
                    <option key={idx} value={cp.path}>
                      {cp.name} {cp.step ? `(Step ${cp.step})` : ''}
                    </option>
                  ))}
                </select>
              </div>

              {/* Configuration */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Training Configuration
                </label>
                <select
                  value={trainingConfig}
                  onChange={(e) => setTrainingConfig(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  data-testid="training-config-select"
                >
                  {configs.map((cfg, idx) => (
                    <option key={idx} value={cfg.name}>
                      {cfg.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Training Steps */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Training Steps
                </label>
                <input
                  type="number"
                  value={trainingSteps}
                  onChange={(e) => setTrainingSteps(parseInt(e.target.value))}
                  min="100"
                  step="100"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  data-testid="training-steps-input"
                />
              </div>

              {/* Data Source Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Training Data Source
                </label>
                <div className="space-y-3">
                  <label className="flex items-center space-x-3 cursor-pointer">
                    <input
                      type="radio"
                      checked={useExistingData}
                      onChange={() => setUseExistingData(true)}
                      className="w-4 h-4 text-purple-600"
                      data-testid="use-existing-data-radio"
                    />
                    <span className="text-gray-700">Use existing datasets</span>
                  </label>
                  <label className="flex items-center space-x-3 cursor-pointer">
                    <input
                      type="radio"
                      checked={!useExistingData}
                      onChange={() => setUseExistingData(false)}
                      className="w-4 h-4 text-purple-600"
                      data-testid="upload-data-radio"
                    />
                    <span className="text-gray-700">Upload custom data</span>
                  </label>
                </div>
              </div>

              {/* Synthetic Data Generation */}
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
                <h3 className="font-semibold text-blue-900 mb-2">🔬 Quick Start: Generate Synthetic Data</h3>
                <p className="text-sm text-blue-700 mb-3">
                  Create synthetic training data with simple patterns for quick testing
                </p>
                <button
                  onClick={createSyntheticData}
                  disabled={isLoading}
                  className="w-full py-3 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-lg font-medium hover:shadow-lg transition-all disabled:opacity-50"
                  data-testid="create-synthetic-button"
                >
                  {isLoading ? '⏳ Creating...' : '✨ Create Synthetic Data'}
                </button>
              </div>

              {/* Dataset Selection or File Upload */}
              {useExistingData ? (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Dataset
                  </label>
                  <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    data-testid="dataset-select"
                  >
                    <option value="">-- Select a dataset --</option>
                    {datasets.map((ds, idx) => (
                      <option key={idx} value={ds.path}>
                        {ds.name} ({ds.type}) - {(ds.size / 1024).toFixed(2)} KB
                      </option>
                    ))}
                  </select>
                  {datasets.length === 0 && (
                    <p className="text-sm text-gray-500 mt-2">
                      No datasets found. Click "Create Synthetic Data" above to get started.
                    </p>
                  )}
                </div>
              ) : (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload Training Data
                  </label>
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleFileUpload}
                    accept=".txt,.json,.jsonl,.csv"
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isLoading}
                    className="w-full py-3 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-500 transition-colors disabled:opacity-50"
                    data-testid="file-upload-button"
                  >
                    {uploadedFile ? (
                      <span className="text-green-600">✅ {uploadedFile.filename}</span>
                    ) : (
                      <span className="text-gray-600">📁 Click to upload (.txt, .json, .jsonl, .csv)</span>
                    )}
                  </button>
                </div>
              )}

              {/* Start Training Button */}
              <button
                onClick={startTraining}
                disabled={isLoading || (trainingStatus?.is_training)}
                className="w-full py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-bold text-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                data-testid="start-training-button"
              >
                {trainingStatus?.is_training ? '⏳ Training in Progress...' : '🚀 Start Training'}
              </button>

              {/* Training Status */}
              {trainingStatus?.is_training && (
                <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h3 className="font-semibold text-blue-900 mb-2">Training Status</h3>
                  <div className="space-y-2">
                    <p className="text-sm text-blue-800">Status: {trainingStatus.status}</p>
                    <p className="text-sm text-blue-800">Progress: {trainingStatus.progress}%</p>
                    {trainingStatus.message && (
                      <p className="text-sm text-blue-700">{trainingStatus.message}</p>
                    )}
                  </div>
                  <div className="mt-3 w-full bg-blue-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${trainingStatus.progress}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="mt-12 pb-6 text-center text-white/70 text-sm">
        <p>HOPE Model Interface - Nested Learning Training & Inference Platform</p>
        <p className="mt-1">Built with React & FastAPI</p>
      </footer>
    </div>
  );
}

export default App;
