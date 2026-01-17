import os
import sys
import time
import subprocess
import threading
import socket
from pyngrok import ngrok

def find_free_port():
    """Finds a free port on the localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        _, port = s.getsockname()
        return port

def run_streamlit(port):
    """Runs the streamlit app in a subprocess using the current python interpreter."""
    print(f"Starting Streamlit App on port {port}...")
    # --server.port ensures we know which port to tunnel
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)])

def start_tunnel(port):
    """Starts the ngrok tunnel."""
    print("Setting up the public link (allow 5-10 seconds)...")
    time.sleep(5) # Wait for streamlit to spin up
    
    try:
        # Try to connect
        public_url = ngrok.connect(port).public_url
        display_success(public_url)
    except Exception as e:
        error_msg = str(e)
        if "authentication failed" in error_msg or "ERR_NGROK_4018" in error_msg:
            print("\n" + "!"*60)
            print("NGROK AUTHENTICATION REQUIRED")
            print("!"*60)
            print("1. Go to: https://dashboard.ngrok.com/signup")
            print("2. Sign up and copy your Authtoken.")
            print("-" * 30)
            
            try:
                token = input("PASTE YOUR TOKEN HERE AND PRESS ENTER: ").strip()
                if token:
                    print("Setting token...")
                    ngrok.set_auth_token(token)
                    try:
                        public_url = ngrok.connect(port).public_url
                        display_success(public_url)
                    except Exception as e2:
                        print(f"Failed again: {e2}")
            except EOFError:
                print("Could not read input.")
        else:
            print(f"Error starting ngrok: {error_msg}")

def display_success(url):
    print("\n" + "="*60)
    print(f"   PUBLIC LINK: {url}")
    print("="*60 + "\n")
    print("Send this link to your friend! (Keep window OPEN)")

if __name__ == "__main__":
    # Dynamically find a free port so we never get "Port in use" errors
    PORT = find_free_port()
    
    # Start the tunnel thread
    thread = threading.Thread(target=start_tunnel, args=(PORT,))
    thread.daemon = True
    thread.start()

    # Run streamlit
    run_streamlit(PORT)
