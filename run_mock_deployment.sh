#!/bin/bash
# Mock Deployment Runner
# Sets up and tests validator-miner communication locally

cd /home/ocean/nereus/precog

echo "🚀 PRECOG MOCK DEPLOYMENT"
echo "=========================="
echo "This script will:"
echo "1. Start a mock miner"
echo "2. Start a mock validator"
echo "3. Test their communication"
echo "4. Show results"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -f "python3 mock_miner.py" 2>/dev/null || true
    pkill -f "python3 mock_validator.py" 2>/dev/null || true
    pkill -f "python3 test_deployment.py" 2>/dev/null || true
    echo "Done."
    exit
}

# Set up cleanup on script exit
trap cleanup EXIT INT TERM

echo ""
echo "📋 DEPLOYMENT OPTIONS:"
echo "1. Run full test suite (recommended)"
echo "2. Run single query test only"
echo "3. Run continuous query test only"
echo "4. Start miner manually"
echo "5. Start validator manually"
echo ""

read -p "Choose an option (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🧪 Running full test suite..."
        python3 test_deployment.py
        ;;
    2)
        echo ""
        echo "🧪 Running single query test..."
        python3 test_deployment.py --single
        ;;
    3)
        echo ""
        echo "🧪 Running continuous query test..."
        python3 test_deployment.py --continuous --queries 10
        ;;
    4)
        echo ""
        echo "🏭 Starting miner manually..."
        echo "Miner will run until you press Ctrl+C"
        echo "Open another terminal and run: python3 standalone_mock_validator.py"
        echo ""
        python3 standalone_mock_miner.py
        ;;
    5)
        echo ""
        echo "🔍 Starting validator manually..."
        echo "Make sure miner is running first!"
        echo "Run: python3 standalone_mock_miner.py"
        echo "Then run this validator in another terminal"
        echo ""
        python3 standalone_mock_validator.py --continuous --interval 10
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🏁 Mock deployment complete!"
