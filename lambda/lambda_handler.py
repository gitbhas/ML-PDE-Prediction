import base64
import os
import sys
import subprocess
import streamlit.web.bootstrap as bootstrap
from streamlit.web.server.server import Server
import threading
from contextlib import contextmanager

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def handler(event, context):
    # Install dependencies if not already installed
    if not os.path.exists("/tmp/requirements_installed"):
        install_requirements()
        with open("/tmp/requirements_installed", "w") as f:
            f.write("installed")

    # Set environment variables for Streamlit
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    # Run the Streamlit application
    def run_streamlit():
        bootstrap.run("aws_agent_streamlit.py", "", [], {})

    thread = threading.Thread(target=run_streamlit)
    thread.daemon = True
    thread.start()

    # Wait for the server to start
    Server.get_current()._wait_for_start()

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
        },
        'body': 'Streamlit application is running'
    }