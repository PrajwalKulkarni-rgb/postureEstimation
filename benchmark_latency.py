import websocket
import json
import time
import numpy as np

def measure_latency():
    uri = "ws://localhost:8000/ws/predict"
    
    try:
        ws = websocket.create_connection(uri)
    except Exception as e:
        print(f"Connection failed: {e}")
        return
        
    # Fake sequence: 50 frames, 17 vertices, 6 channels (X,Y,Z pos & X,Y,Z vel)
    fake_data = np.random.rand(50, 17, 6).tolist()
    payload = json.dumps(fake_data)
    
    print("Connected to server. Warming up...")
    for _ in range(5):
        ws.send(payload)
        ws.recv()
        
    print("Running 100 inference iterations over WebSocket...")
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        ws.send(payload)
        res = ws.recv()  # Block until response is received
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        
    ws.close()
    
    avg_latency = sum(latencies) / len(latencies)
    p95 = np.percentile(latencies, 95)
    print("-" * 30)
    print(f"Average End-to-End Latency: {avg_latency:.2f} ms")
    print(f"95th Percentile Latency:    {p95:.2f} ms")
    print(f"Minimum Latency:            {min(latencies):.2f} ms")
    print(f"Maximum Latency:            {max(latencies):.2f} ms")
    print("-" * 30)

if __name__ == "__main__":
    measure_latency()
