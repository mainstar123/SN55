#!/bin/bash
# Easy script to run the mining dashboard

cd /home/ocean/nereus/precog

echo "🚀 Starting Precog Mining Dashboard..."
echo ""

# Check if miner is running
if ps aux | grep -v grep | grep "miner.py" > /dev/null; then
    echo "✅ Miner is running"
else
    echo "⚠️  Miner is not running - consider starting it"
fi

echo ""
echo "📊 Available Dashboards:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 🌐 Web Dashboard (Streamlit): http://localhost:8501"
echo "2. 💻 Terminal Dashboard: ./comprehensive_dashboard.sh"
echo "3. 📈 Simple Monitor: ./mining_dashboard.sh"
echo ""

# Start streamlit if not already running
if ! ps aux | grep -v grep | grep "streamlit" > /dev/null; then
    echo "Starting web dashboard..."
    source venv/bin/activate
    HOME=/home/ocean streamlit run mining_dashboard.py --server.port 8501 --server.headless true --browser.gatherUsageStats false &
    sleep 3
    echo "✅ Web dashboard started at: http://localhost:8501"
else
    echo "✅ Web dashboard already running at: http://localhost:8501"
fi

echo ""
echo "🎯 Dashboard Features:"
echo "- Real-time wallet balance & mining metrics"
echo "- Competitor rankings & analysis"
echo "- Performance charts & analytics"
echo "- System status monitoring"
echo ""

echo "💡 Tip: Open http://localhost:8501 in your browser for the full dashboard experience!"
