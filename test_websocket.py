#!/usr/bin/env python3
"""
Test WebSocket connection to testnet
"""

import asyncio
import websockets
import sys

async def test_websocket():
    uri = "wss://test.finney.opentensor.ai:443"
    print(f"🔌 Testing WebSocket connection to: {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection successful!")
            # Send a simple ping
            await websocket.ping()
            print("✅ Ping successful!")
            return True
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_websocket())
    sys.exit(0 if result else 1)
