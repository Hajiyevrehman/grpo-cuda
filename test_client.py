import requests
import json
import argparse

def test_api(url, ref_code_file, kernel_code_file, n_correctness=5, n_trials=10):
    """
    Test the kernel evaluation API
    
    Args:
        url: API endpoint URL
        ref_code_file: Path to reference model code file
        kernel_code_file: Path to kernel code file
        n_correctness: Number of correctness trials
        n_trials: Number of performance trials
    """
    # Read code files
    with open(ref_code_file, 'r') as f:
        ref_code = f.read()
        
    with open(kernel_code_file, 'r') as f:
        kernel_code = f.read()
    
    # Prepare request payload
    payload = {
        'ref_code': ref_code,
        'code': kernel_code,
        'n_correctness': n_correctness,
        'n_trials': n_trials
    }
    
    # Make API request
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    
    # Print results
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Test ID: {result.get('test_id')}")
        print(f"Compiled: {result.get('compiled')}")
        print(f"Correct: {result.get('correct')}")
        
        if result.get('correct'):
            print(f"Performance Multiplier: {result.get('performance_multiplier'):.2f}x")
            print(f"Reward: {result.get('reward')}")
            
            perf = result.get('performance')
            print(f"Kernel Time: {perf.get('mean'):.2f} ms ± {perf.get('std'):.2f} ms")
            
            ref_stats = result.get('metadata', {}).get('ref_stats', {})
            if ref_stats:
                print(f"Reference Time: {ref_stats.get('mean'):.2f} ms ± {ref_stats.get('std'):.2f} ms")
        else:
            print(f"Reward: {result.get('reward')}")
            
            if not result.get('compiled'):
                print("Compilation Error:")
                print(result.get('metadata', {}).get('compilation_error', 'Unknown error'))
            else:
                print("Correctness Error:")
                print(result.get('metadata', {}).get('correctness_error', 'Unknown error'))
    else:
        print(f"Error: {response.text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test kernel evaluation API')
    parser.add_argument('--url', default='http://localhost:5000/test_kernel', help='API endpoint URL')
    parser.add_argument('--ref', required=True, help='Path to reference model code file')
    parser.add_argument('--kernel', required=True, help='Path to kernel code file')
    parser.add_argument('--n_correctness', type=int, default=5, help='Number of correctness trials')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of performance trials')
    
    args = parser.parse_args()
    test_api(args.url, args.ref, args.kernel, args.n_correctness, args.n_trials)