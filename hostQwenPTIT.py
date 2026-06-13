cat > run.sh << 'EOF'
set -e

echo "1. Syncing dependencies with uv..."
uv sync

echo "2. Patching outlines to fix pyairports error..."
SITE_PKG=$(uv run python -c "import site; print(site.getsitepackages()[0])")
echo "Site-packages path: $SITE_PKG"

cat > "$SITE_PKG/outlines/types/airports.py" << 'END'
AIRPORT_LIST = []
END

cat > "$SITE_PKG/outlines/types/countries.py" << 'END'
COUNTRY_LIST = []
END

echo "3. Starting vLLM server in background..."
uv run vllm serve "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code > vllm_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for vLLM server to start (90s)..."
sleep 90

if curl -s http://localhost:8000/health > /dev/null; then
    IP=$(curl -s ifconfig.me)
    echo ""
    echo "SERVER IS READY"
    echo "API URL: http://$IP:8000/v1"
    echo ""
    echo "To monitor logs: tail -f vllm_server.log"
    echo "To stop server: kill $SERVER_PID"
else
    echo "Server failed to start. Check vllm_server.log"
    tail -20 vllm_server.log
fi
EOF

chmod +x run.sh
./run.sh