# GRPO-CUDA

A CUDA-accelerated implementation of the GRPO algorithm for efficient model training.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Hajiyevrehman/grpo-cuda.git
cd grpo-cuda
```

## Dependencies

The project requires several dependencies:

```bash
# Install Python packages
pip install pyngrok

# Install system dependencies
sudo apt-get update
sudo apt-get install -y ninja-build
```

## Setting Up the API Server

This project includes an API server that can be exposed via ngrok for remote access:

```python
from pyngrok import ngrok

# Set your ngrok authentication token
# You can get a token by signing up at https://ngrok.com/
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

# Start the tunnel on port 5000
public_url = ngrok.connect(5000)
print(f"Your API is accessible at: {public_url.public_url}/test_kernel")
```

## Starting the Server

Make the start script executable and run it:

```bash
chmod +x start_server.sh
./start_server.sh
```

## Usage

Once the server is running, you can access the API at the ngrok URL printed to the console. The API endpoint is available at `/test_kernel`.

## Troubleshooting

- If you encounter permission issues when running apt commands, try using `sudo` instead.
- Make sure your ngrok authentication token is valid.
- Check system requirements for CUDA compatibility.

## License

[Add license information here]

## Contributors

[Add contributor information here]