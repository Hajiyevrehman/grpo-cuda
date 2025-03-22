!git clone https://github.com/Hajiyevrehman/grpo-cuda.git


%cd grpo-cuda

!pip install pyngrok

from pyngrok import ngrok
ngrok.set_auth_token("2uexGGjzXogkuUL6t7bkM7nJFNb_7stvg52fgVV7MrtFicnyr")
public_url = ngrok.connect(5000)
print(f"Your API is accessible at: {public_url.public_url}/test_kernel")



!apt-get update
!apt-get install -y ninja-build

you might need to change !apt to sudo


!chmod +x start_server.sh
!./start_server.sh

