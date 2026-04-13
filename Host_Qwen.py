# KAGGLE H100 NOTEBOOK - TỰ HOST QWEN3-30B
# ============================================

import subprocess
import time
import requests
import os
import json
from typing import Optional

class QwenHostOnKaggle:
    """
    Tự host Qwen3-30B-A3B-Instruct trên Kaggle H100 GPU
    """
    
    def __init__(self):
        self.port = 8000
        self.host = "0.0.0.0"  # Cho phép truy cập từ bên ngoài
        self.model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        self.process = None
        self.ngrok_auth_token = "3CCzkWKo2VmsRd2kMkyMEsdBVv9_87FNueGVjD4KNhXTtoS6C"  # Your ngrok token
        
    def install_dependencies(self):
        """Cài đặt các thư viện cần thiết"""
        print("📦 Installing dependencies...")
        subprocess.run("pip install vllm ray pyngrok -q", shell=True)
        print("✓ Dependencies installed")
    
    def start_server(self):
        """Khởi động vLLM server với Qwen3-30B"""
        
        # Command tối ưu cho H100 (80GB VRAM)
        vllm_command = f"""
        python -m vllm.entrypoints.openai.api_server \
            --model {self.model_name} \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.90 \
            --max-model-len 8192 \
            --host {self.host} \
            --port {self.port} \
            --trust-remote-code \
            --dtype auto \
            --enforce-eager
        """
        
        print(f"🚀 Starting Qwen3-30B server on port {self.port}...")
        print(f"📍 Model: {self.model_name}")
        print(f"🖥️  Host: {self.host}:{self.port}")
        
        # Start server
        self.process = subprocess.Popen(
            vllm_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Wait for server to be ready
        return self.wait_for_server()
    
    def wait_for_server(self, max_retries=60):
        """Chờ server sẵn sàng"""
        api_url = f"http://localhost:{self.port}/v1/models"
        
        print("\n⏳ Waiting for server to be ready...")
        
        for i in range(max_retries):
            try:
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    models = response.json()
                    print(f"\n✅ Server is READY!")
                    print(f"📋 Available models: {models}")
                    
                    # Hiển thị thông tin truy cập
                    self.show_access_info()
                    return True
            except requests.exceptions.ConnectionError:
                pass
            
            # Hiển thị tiến trình
            if i % 10 == 0:
                print(f"   Still waiting... ({i*5}s)")
            
            time.sleep(5)
            
            # Kiểm tra process còn sống không
            if self.process.poll() is not None:
                print("\n❌ Server process died!")
                return False
        
        print("\n❌ Timeout waiting for server")
        return False
    
    def show_access_info(self):
        """Hiển thị thông tin để truy cập từ VS Code"""
        print("\n" + "="*60)
        print("🔗 ACCESS INFORMATION FOR VS CODE")
        print("="*60)
        
        # Lấy URL public từ Kaggle
        public_url = self.get_kaggle_public_url()
        
        print(f"\n📍 Local URL: http://localhost:{self.port}")
        print(f"📍 Public URL: {public_url}")
        
        print("\n📝 COPY THIS TO YOUR .env FILE:")
        print("-"*60)
        print(f"OPENAI_BASE_URL={public_url}/v1")
        print(f"OPENAI_API_KEY=EMPTY")
        print(f"MODEL_NAME={self.model_name}")
        print("-"*60)
        
        print("\n💡 Test with curl:")
        print(f'curl {public_url}/v1/models')
        
        print("\n💡 Python test code:")
        print(f"""
from openai import OpenAI

client = OpenAI(
    base_url="{public_url}/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="{self.model_name}",
    messages=[{{"role": "user", "content": "Hello, Qwen!"}}]
)
print(response.choices[0].message.content)
        """)
        print("="*60)
        
        # In thêm hướng dẫn cho Memento
        print("\n" + "="*60)
        print("📚 FOR MEMENTO PAPER - CONFIGURATION")
        print("="*60)
        print("\nAdd these to Memento/.env:")
        print("-"*60)
        print(f"OPENAI_BASE_URL={public_url}/v1")
        print(f"OPENAI_API_KEY=EMPTY")
        print(f"MODEL_NAME={self.model_name}")
        print(f"PLANNER_MODEL={self.model_name}")
        print(f"EXECUTOR_MODEL={self.model_name}")
        print("-"*60)
        print("="*60)
    
    def get_kaggle_public_url(self) -> str:
        """
        Lấy URL public từ Kaggle
        Sử dụng ngrok với auth token đã cấu hình
        """
        # Kiểm tra xem có đang chạy trên Kaggle không
        if os.path.exists('/kaggle'):
            print("\n⚠️  Running on Kaggle - Setting up ngrok tunnel...")
            return self.setup_ngrok()
        
        return f"http://localhost:{self.port}"
    
    def setup_ngrok(self) -> str:
        """Cài đặt và chạy ngrok để expose port với auth token"""
        try:
            # Cài ngrok (đã cài ở install_dependencies)
            from pyngrok import ngrok
            
            # Set auth token - QUAN TRỌNG!
            print("🔑 Configuring ngrok with your auth token...")
            ngrok.set_auth_token(self.ngrok_auth_token)
            
            # Kill existing tunnels
            ngrok.kill()
            
            # Create tunnel
            print(f"🔗 Creating tunnel to port {self.port}...")
            tunnel = ngrok.connect(self.port, "http")
            public_url = tunnel.public_url
            
            print(f"\n✅ Ngrok tunnel created successfully!")
            print(f"🌍 Public URL: {public_url}")
            print(f"📊 Tunnel Status: {tunnel.status}")
            
            # In thêm thông tin về tunnel
            print(f"\n📡 Ngrok Inspector: http://127.0.0.1:4040")
            
            return public_url
            
        except Exception as e:
            print(f"⚠️  Ngrok setup failed: {e}")
            print("   Trying alternative method...")
            return self.setup_cloudflared_fallback()
    
    def setup_cloudflared_fallback(self) -> str:
        """Fallback dùng cloudflared nếu ngrok fail"""
        try:
            print("🔄 Attempting cloudflared as fallback...")
            
            # Download cloudflared
            subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", shell=True)
            subprocess.run("chmod +x cloudflared-linux-amd64", shell=True)
            subprocess.run("mv cloudflared-linux-amd64 /usr/local/bin/cloudflared", shell=True)
            
            import threading
            import re
            
            public_url = [None]
            
            def run_tunnel():
                process = subprocess.Popen(
                    ["cloudflared", "tunnel", "--url", f"http://localhost:{self.port}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                for line in process.stdout:
                    match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
                    if match and not public_url[0]:
                        public_url[0] = match.group()
                        print(f"\n✅ Cloudflared tunnel created: {public_url[0]}")
                        return
            
            thread = threading.Thread(target=run_tunnel, daemon=True)
            thread.start()
            
            # Wait for URL
            for i in range(30):
                if public_url[0]:
                    return public_url[0]
                time.sleep(1)
            
            raise Exception("Cloudflared timeout")
            
        except Exception as e:
            print(f"⚠️  All tunneling methods failed: {e}")
            print("   Using localhost only (will not work from VS Code)")
            return f"http://localhost:{self.port}"
    
    def monitor_logs(self):
        """Monitor server logs"""
        print("\n📡 Server logs (Ctrl+C to stop monitoring):\n")
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    print(f"[vLLM] {line.strip()}")
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped monitoring")
    
    def stop_server(self):
        """Dừng server"""
        if self.process:
            print("\n🛑 Stopping server...")
            self.process.terminate()
            self.process.wait()
            print("✓ Server stopped")
            
            # Close ngrok tunnels
            try:
                from pyngrok import ngrok
                ngrok.kill()
                print("✓ Ngrok tunnels closed")
            except:
                pass

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🎯 QWEN3-30B HOSTING ON KAGGLE H100")
    print("="*60)
    
    # Khởi tạo host
    host = QwenHostOnKaggle()
    
    # Cài đặt dependencies
    host.install_dependencies()
    
    # Start server
    if host.start_server():
        print("\n" + "="*60)
        print("✅ Qwen3-30B is running successfully!")
        print("="*60)
        print("\n🎯 IMPORTANT:")
        print("1. Keep this notebook running")
        print("2. Copy the Public URL above")
        print("3. Paste into your VS Code Memento/.env file")
        print("4. Run Memento paper in VS Code")
        print("\n🛑 Press Ctrl+C to stop server\n")
        
        # Monitor logs
        host.monitor_logs()
    else:
        print("\n❌ Failed to start server")
        print("Check error messages above")