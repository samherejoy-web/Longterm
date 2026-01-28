import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function TraitsPanel() {
  const [traits, setTraits] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [memorizationTest, setMemorizationTest] = useState(null);
  const [testingMemorization, setTestingMemorization] = useState(false);

  const loadTraits = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${BACKEND_URL}/api/model/traits`);
      setTraits(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const testMemorization = async () => {
    setTestingMemorization(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/model/memorize`, {
        message: "Test memorization with teach signal",
        max_length: 50
      });
      setMemorizationTest(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setTestingMemorization(false);
    }
  };

  useEffect(() => {
    loadTraits();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loader"></div>
        <span className="ml-3 text-gray-600">Loading model traits...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <h3 className="text-red-800 font-semibold mb-2">❌ Error</h3>
        <p className="text-red-700">{error}</p>
        <button
          onClick={loadTraits}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!traits) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p>No model loaded. Load a model from the Models tab first.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="traits-panel">
      <h2 className="text-2xl font-bold text-gray-800">🧬 Model Architecture Traits</h2>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
          <div className="text-2xl mb-2">✅</div>
          <div className="text-sm font-medium text-blue-700">HOPE Architecture</div>
          <div className="text-2xl font-bold text-blue-900">{traits.hope_variant}</div>
        </div>
        
        <div className="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
          <div className="text-2xl mb-2">🧠</div>
          <div className="text-sm font-medium text-green-700">CMS Levels</div>
          <div className="text-2xl font-bold text-green-900">{traits.cms_features.num_levels}</div>
        </div>
        
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
          <div className="text-2xl mb-2">⚡</div>
          <div className="text-sm font-medium text-purple-700">Fast State</div>
          <div className="text-2xl font-bold text-purple-900">
            {traits.fast_state.enabled ? "Enabled" : "Disabled"}
          </div>
        </div>
      </div>

      {/* Detailed Traits */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* HOPE Features */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span className="text-2xl mr-2">🎯</span>
            HOPE Features
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Block Type:</span>
              <span className="font-medium text-gray-900">{traits.hope_features.block_type}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">QK L2 Norm:</span>
              <span className={`font-medium ${traits.hope_features.qk_l2_norm ? 'text-green-600' : 'text-gray-400'}`}>
                {traits.hope_features.qk_l2_norm ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Local Conv Window:</span>
              <span className="font-medium text-gray-900">
                {traits.hope_features.local_conv_window || 'None'}
              </span>
            </div>
          </div>
        </div>

        {/* CMS Features */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span className="text-2xl mr-2">💾</span>
            CMS (Continual Memory System)
          </h3>
          <div className="space-y-3 text-sm">
            {traits.cms_features.levels.map((level, idx) => (
              <div key={idx} className="flex justify-between items-center bg-gray-50 p-2 rounded">
                <span className="font-medium text-gray-700">{level.name}</span>
                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                  Period: {level.update_period}
                </span>
              </div>
            ))}
            <div className="mt-3 pt-3 border-t">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Use LayerNorm:</span>
                <span className={`font-medium ${traits.cms_features.use_layernorm ? 'text-green-600' : 'text-gray-400'}`}>
                  {traits.cms_features.use_layernorm ? '✓' : '✗'}
                </span>
              </div>
              <div className="flex justify-between items-center mt-1">
                <span className="text-gray-600">Flush Partial:</span>
                <span className={`font-medium ${traits.cms_features.flush_partial ? 'text-green-600' : 'text-gray-400'}`}>
                  {traits.cms_features.flush_partial ? '✓' : '✗'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Self-Modifying Titans */}
        {traits.selfmod_features.has_selfmod && (
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <span className="text-2xl mr-2">🤖</span>
              Self-Modifying Titans
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Chunk Size:</span>
                <span className="font-medium text-gray-900">{traits.selfmod_features.chunk_size}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Objective:</span>
                <span className="font-medium text-gray-900">{traits.selfmod_features.objective}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Rank-1 Precond:</span>
                <span className={`font-medium ${traits.selfmod_features.use_rank1_precond ? 'text-green-600' : 'text-gray-400'}`}>
                  {traits.selfmod_features.use_rank1_precond ? '✓' : '✗'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Adaptive Q:</span>
                <span className={`font-medium ${traits.selfmod_features.adaptive_q ? 'text-green-600' : 'text-gray-400'}`}>
                  {traits.selfmod_features.adaptive_q ? '✓' : '✗'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Local Conv:</span>
                <span className="font-medium text-gray-900">
                  {traits.selfmod_features.local_conv_window || 'None'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Titan Features */}
        {traits.titan_features.has_titans && (
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <span className="text-2xl mr-2">⚔️</span>
              TITAN Memory
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Level Name:</span>
                <span className="font-medium text-gray-900">{traits.titan_features.titan_level.name}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Update Period:</span>
                <span className="font-medium text-gray-900">{traits.titan_features.titan_level.update_period}</span>
              </div>
            </div>
          </div>
        )}

        {/* Fast State */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span className="text-2xl mr-2">⚡</span>
            Fast State (Nested Learning Semantics)
          </h3>
          <p className="text-sm text-gray-600 mb-4">{traits.fast_state.description}</p>
          <div className="flex items-center justify-between bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
            <div>
              <div className="font-semibold text-purple-900">Status</div>
              <div className="text-sm text-purple-700">
                {traits.fast_state.enabled ? '✅ Enabled and Active' : '❌ Not Available'}
              </div>
            </div>
            <button
              onClick={testMemorization}
              disabled={testingMemorization}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
            >
              {testingMemorization ? '⏳ Testing...' : '🧪 Test Memorization'}
            </button>
          </div>

          {memorizationTest && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h4 className="font-semibold text-green-900 mb-2">Memorization Test Results</h4>
              <div className="space-y-1 text-sm text-green-800">
                <div>✅ {memorizationTest.description}</div>
                <div>📊 Effect Magnitude: {memorizationTest.memorization_effect.toFixed(6)}</div>
                <div>⚡ Fast State Used: {memorizationTest.fast_state_used ? 'Yes' : 'No'}</div>
                {Object.keys(memorizationTest.update_metrics).length > 0 && (
                  <div className="mt-2">
                    <div className="font-medium">Update Metrics:</div>
                    <pre className="text-xs bg-green-100 p-2 rounded mt-1 overflow-x-auto">
                      {JSON.stringify(memorizationTest.update_metrics, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Teach Signal */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span className="text-2xl mr-2">📡</span>
            Teach Signal (δℓ Computation)
          </h3>
          <p className="text-sm text-gray-600 mb-3">{traits.teach_signal.description}</p>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-blue-50 p-3 rounded">
              <div className="text-gray-600">Scale</div>
              <div className="text-xl font-bold text-blue-900">{traits.teach_signal.scale}</div>
            </div>
            <div className="bg-blue-50 p-3 rounded">
              <div className="text-gray-600">Clip</div>
              <div className="text-xl font-bold text-blue-900">{traits.teach_signal.clip}</div>
            </div>
          </div>
        </div>

        {/* Test Coverage */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span className="text-2xl mr-2">✅</span>
            Tensor-Level Invariants (Unit Test Coverage)
          </h3>
          <div className="space-y-2">
            {traits.tensor_invariants.test_coverage.map((test, idx) => (
              <div key={idx} className="flex items-start space-x-2 text-sm">
                <span className="text-green-600 mt-0.5">✓</span>
                <span className="text-gray-700">{test}</span>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 bg-green-50 rounded-lg">
            <div className="text-sm text-green-800">
              🎉 All critical tensor invariants are verified by unit tests
            </div>
          </div>
        </div>
      </div>

      {/* Dimensions Info */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">📐 Model Dimensions</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-sm text-gray-600">Vocab Size</div>
            <div className="text-2xl font-bold text-indigo-900">{traits.dimensions.vocab_size}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Dimension</div>
            <div className="text-2xl font-bold text-indigo-900">{traits.dimensions.dim}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Layers</div>
            <div className="text-2xl font-bold text-indigo-900">{traits.dimensions.num_layers}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Heads</div>
            <div className="text-2xl font-bold text-indigo-900">{traits.dimensions.heads}</div>
          </div>
        </div>
      </div>

      {/* Refresh Button */}
      <div className="text-center">
        <button
          onClick={loadTraits}
          disabled={loading}
          className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors disabled:opacity-50"
        >
          🔄 Refresh Traits
        </button>
      </div>
    </div>
  );
}

export default TraitsPanel;
