from flask import Flask, request, jsonify
import torch
import os
import uuid
import time
from kernel_tester import evaluate_kernel, KernelExecResult

app = Flask(__name__)

@app.route('/test_kernel', methods=['POST'])
def test_kernel():
    data = request.json
    if not data or 'code' not in data:
        return jsonify({'error': 'No kernel code provided'}), 400
    
    kernel_code = data.get('code')
    ref_code = data.get('ref_code')
    n_correctness = data.get('n_correctness', 5)
    n_trials = data.get('n_trials', 10)
    
    # Generate unique ID for this test
    test_id = str(uuid.uuid4())
    build_dir = os.path.join('builds', test_id)
    os.makedirs(build_dir, exist_ok=True)
    
    try:
        # Evaluate the kernel
        result = evaluate_kernel(
            ref_code=ref_code, 
            kernel_code=kernel_code,
            build_dir=build_dir,
            num_correct_trials=n_correctness,
            num_perf_trials=n_trials
        )
        
        # Calculate reward based on correctness and performance
        reward = calculate_reward(result)
        
        return jsonify({
            'test_id': test_id,
            'compiled': result.compiled,
            'correct': result.correctness,
            'performance_multiplier': result.metadata.get('speedup', 0.0) if result.correctness else 0.0,
            'performance': result.runtime_stats if result.correctness else None,
            'reward': reward,
            'metadata': result.metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_reward(result: KernelExecResult) -> float:
    """Calculate reward based on correctness and performance"""
    if not result.compiled or not result.correctness:
        return 0.0
    
    # Get performance multiplier (baseline/kernel time)
    performance_multiplier = result.metadata.get('speedup', 0.0)
    
    if performance_multiplier < 1:
        return 0.5
    elif 1 <= performance_multiplier < 1.1:
        return 0.7
    else:  # performance_multiplier >= 1.1
        return 1.0

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)