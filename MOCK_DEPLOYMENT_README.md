# 🎭 Mock Deployment System

A complete local testing environment for Precog validator-miner communication without blockchain dependencies.

## 🎯 Overview

This mock deployment system allows you to test the complete validator-miner communication flow locally, simulating real-world deployment scenarios without needing:

- Blockchain connectivity
- Wallet registration
- TAO tokens
- Network infrastructure

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Mock Validator │────│   Mock Miner    │
│                 │    │                 │
│ • Sends Challenge│    │ • Receives      │
│   synapses       │    │   requests      │
│ • Queries miners │    │ • Processes     │
│ • Validates      │    │   predictions   │
│   responses      │    │ • Returns        │
│ • Calculates     │    │   responses     │
│   rewards        │    │                 │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
              Local Network
```

## 📁 Files

### Core Components
- **`standalone_mock_validator.py`** - HTTP-based validator (no dependencies)
- **`standalone_mock_miner.py`** - HTTP-based miner (no dependencies)
- **`mock_validator.py`** - Bittensor-based validator (requires bittensor)
- **`mock_miner.py`** - Bittensor-based miner (requires bittensor)
- **`test_deployment.py`** - Automated testing suite for validator-miner communication

### Utilities
- **`run_mock_deployment.sh`** - Interactive deployment runner
- **`start_local_miner.py`** - Alternative simplified miner (not recommended)
- **`standalone_miner.py`** - Basic miner without bittensor dependencies

## 🚀 Quick Start

### Option 1: Interactive Deployment (Recommended)

```bash
chmod +x run_mock_deployment.sh
./run_mock_deployment.sh
```

Choose option 1 for the full test suite.

### Option 2: Manual Testing

**Terminal 1 - Start Miner:**
```bash
python3 standalone_mock_miner.py --port 8092
```

**Terminal 2 - Start Validator:**
```bash
python3 standalone_mock_validator.py --miner-host 127.0.0.1 --miner-port 8092 --continuous --interval 5
```

### Option 3: Automated Testing

```bash
# Run full test suite
python3 test_deployment.py

# Run single test only
python3 test_deployment.py --single

# Run continuous test with custom query count
python3 test_deployment.py --continuous --queries 20
```

## 🔧 Configuration

### Mock Validator Options

```bash
python3 mock_validator.py [options]

Options:
  --miner-host HOST      Miner host (default: 127.0.0.1)
  --miner-port PORT      Miner port (default: 8092)
  --assets ASSET [ASSET ...]  Assets to query (default: tao_bittensor btc eth)
  --continuous           Run continuous queries
  --interval SECONDS     Query interval for continuous mode (default: 30)
  --max-queries N        Maximum queries for continuous mode
```

### Mock Miner Options

```bash
python3 mock_miner.py [options]

Options:
  --port PORT            Port to run miner on (default: 8092)
  --model-path PATH      Path to trained model file (default: elite_domination_model.pth)
```

## 📊 What Gets Tested

### Communication Flow
1. **Synapse Creation** - Validator creates Challenge synapse with timestamp/assets
2. **Request Transmission** - Validator sends synapse via dendrite
3. **Request Reception** - Miner receives synapse via axon
4. **Prediction Generation** - Miner processes request and generates predictions
5. **Response Transmission** - Miner sends response back
6. **Response Validation** - Validator receives and validates response

### Prediction Quality
- **Asset Coverage** - All requested assets have predictions
- **Data Types** - Predictions are proper numbers
- **Interval Validity** - Prediction intervals are properly formatted
- **Response Time** - Performance monitoring

### Error Handling
- **Network Issues** - Connection failures
- **Invalid Requests** - Malformed synapses
- **Model Errors** - Prediction generation failures
- **Timeout Handling** - Request timeouts

## 🎮 Example Output

### Successful Communication
```
🏭 Standalone Mock Miner initialized on port 8092
🔍 Standalone Mock Validator initialized
   Target miner: http://127.0.0.1:8092

📤 Query #1 - Requesting predictions for ['btc', 'eth', 'tao_bittensor']
   Timestamp: 2025-12-21 04:00:26

📥 Request #1 received
   Timestamp: 2025-12-21 04:00:26
   Assets: ['btc', 'eth', 'tao_bittensor']

📤 Response sent in 0.010s
   Predictions: {'btc': 51153.36, 'eth': 2930.84, 'tao_bittensor': 218.31}
   Intervals: {'btc': [46038.03, 56268.70], 'eth': [2637.75, 3223.92], 'tao_bittensor': [196.47, 240.14]}

✅ Response received in 0.01s
🔍 Validating response...
   ✅ Response validation complete
```

### Test Results Summary
```
📊 TEST RESULTS SUMMARY
============================================================
Single Query: ✅ PASS
Continuous Queries: ✅ PASS

Overall: 2/2 tests passed
🎉 All tests passed! Validator-miner communication is working.
```

## 🧠 Mock Predictions

The mock miner generates realistic predictions using:

- **Base Prices**: Realistic current market prices
- **Trend Component**: Sinusoidal trend simulation
- **Noise Component**: Random noise (±2% of base price)
- **Prediction Intervals**: ±10% confidence intervals
- **Deterministic Seeds**: Timestamp-based for reproducible testing

## 🔍 Monitoring & Debugging

### Real-time Logs
Both validator and miner provide detailed logging:
- Request/response timing
- Prediction values
- Error conditions
- Performance metrics

### Test Metrics
- **Success Rate**: Percentage of successful queries
- **Response Time**: Average time for miner responses
- **Throughput**: Queries per second
- **Error Rate**: Failed request percentage

## 🚨 Troubleshooting

### Common Issues

**"Connection refused"**
- Ensure miner is started before validator
- Check that ports match between validator and miner
- Verify miner process is running: `ps aux | grep mock_miner.py`

**"No predictions in response"**
- Check miner logs for processing errors
- Verify model file exists and loads correctly
- Ensure assets are supported (btc, eth, tao_bittensor)

**"Timeout errors"**
- Increase timeout in validator: modify `timeout` parameter
- Check miner performance and system resources
- Reduce query frequency for slower systems

**"Import errors"**
- Ensure you're in the project root directory
- Check Python path: `python3 -c "import sys; print(sys.path)"`
- Verify all dependencies are installed

### Debug Mode

Enable verbose logging by modifying the scripts:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Next Steps

After successful mock testing:

1. **Real Network Testing** - Use `start_testnet_miner.sh` for actual testnet
2. **Production Deployment** - Deploy on mainnet with real wallets
3. **Performance Optimization** - Optimize model inference speed
4. **Monitoring Setup** - Implement production monitoring
5. **Scaling** - Handle multiple validators/miners

## 📝 Notes

- Mock predictions are deterministic based on timestamp for reproducible testing
- The system simulates real bittensor networking without blockchain overhead
- All communication happens locally via HTTP (simulated axon/dendrite)
- Performance metrics reflect local system capabilities, not network latency

## 🤝 Contributing

When adding new features to the mock deployment:

1. Update both validator and miner if changing the protocol
2. Add corresponding tests in `test_deployment.py`
3. Update this documentation
4. Test with both single and continuous query modes
