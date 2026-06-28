import sys
import os
import json
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import uvicorn 
import asyncio
from collections import deque
import struct

def interpolate_sequence(timestamps, frames, target_timestamps):
    """Linearly interpolates a sequence of frames to match exact target timestamps."""
    N, V, C = frames.shape
    interpolated = np.zeros((len(target_timestamps), V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            interpolated[:, v, c] = np.interp(target_timestamps, timestamps, frames[:, v, c])
    return interpolated

# Fix path to allow importing from 'core' (assuming server/ is in project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn.model import STGCN

# --- CONFIG ---
MODEL_PATH = "server/STGCN.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global Model Variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Model
    global model
    try:
        # Initialize model structure (6 channels for pos+vel)
        model = STGCN(num_classes=3, in_channels=6).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ Server: AI Model Loaded on {DEVICE}")
    except Exception as e:
        print(f"❌ Server: Failed to load model: {e}")
    
    yield
    
    # Shutdown logic (if any)
    print("🛑 Server: Shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "Industrial Safety AI Online"}

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Generate unique session ID
    import time as pytime
    client_ip = websocket.client.host if websocket.client else "unknown"
    session_id = f"{client_ip}_{int(pytime.time())}"
    print(f"🔌 Client Connected: {session_id}")
    
    # Setup Logging
    os.makedirs("server/logs", exist_ok=True)
    log_filename = f"server/logs/session_{session_id}.csv"
    
    # Shared State
    frame_buffer = deque(maxlen=300) # Holds (timestamp, frame) tuples
    state = {"running": True}
    
    def run_inference(tensor):
        """Runs the PyTorch model synchronously (will be called in thread pool)"""
        if model is None: return -1, 0.0
        with torch.no_grad():
            logits = model(tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            conf = float(torch.max(torch.softmax(logits, dim=1)))
        return pred_idx, conf

    async def receiver_task():
        """Vacuum loop: Constantly pulls data from TCP buffer to prevent queuing."""
        try:
            while state["running"]:
                data = await websocket.receive_bytes()
                timestamp = struct.unpack('d', data[:8])[0]
                frame = np.frombuffer(data[8:], dtype=np.float32).reshape(17, 3)
                frame_buffer.append((timestamp, frame))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Receiver error: {e}")
        finally:
            state["running"] = False

    async def inference_task():
        """Brain loop: Analyzes the buffer every 10ms."""
        with open(log_filename, "w") as log_file:
            log_file.write("server_time,client_timestamp,latency_ms,inference_time_ms,buffer_depth\n")
            
            try:
                while state["running"]:
                    if len(frame_buffer) >= 10:
                        latest_ts = frame_buffer[-1][0]
                        
                        if latest_ts - frame_buffer[0][0] >= 5.0:
                            # Extract exactly the last 5.0 seconds
                            window_start = latest_ts - 5.0
                            window_data = [item for item in frame_buffer if item[0] >= window_start]
                            times = np.array([item[0] for item in window_data])
                            frames = np.array([item[1] for item in window_data])
                            
                            # FPS-Agnostic Interpolation
                            SEQ_LEN = 50
                            DT = 5.0 / SEQ_LEN # 0.1s
                            target_times = np.linspace(window_start, latest_ts, SEQ_LEN)
                            
                            S_interp = interpolate_sequence(times, frames, target_times)
                            V_interp = np.diff(S_interp, axis=0, prepend=S_interp[0:1]) / DT
                            full_seq = np.concatenate((S_interp, V_interp), axis=2)
                            
                            tensor = torch.tensor(full_seq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                            
                            # Asynchronous Inference
                            t0 = pytime.time()
                            pred_idx, conf = await asyncio.to_thread(run_inference, tensor)
                            inf_time_ms = (pytime.time() - t0) * 1000.0
                            
                            # Network + Processing Latency (Requires synchronized clocks, valid on localhost)
                            latency_ms = (pytime.time() - latest_ts) * 1000.0
                            
                            # Send Result Back
                            response = {"status": "OK", "prediction": pred_idx, "confidence": conf}
                            await websocket.send_text(json.dumps(response))
                            
                            # Log Diagnostics
                            log_file.write(f"{pytime.time():.3f},{latest_ts:.3f},{latency_ms:.1f},{inf_time_ms:.1f},{len(frame_buffer)}\n")
                            log_file.flush()
                            
                            # Keep buffer clean (retain slightly more than 5 seconds)
                            while len(frame_buffer) > 0 and latest_ts - frame_buffer[0][0] > 6.0:
                                frame_buffer.popleft()
                        else:
                            await websocket.send_text(json.dumps({"status": "Buffering", "prediction": -1, "confidence": 0}))
                    
                    await asyncio.sleep(0.01) # Yield to event loop
            except Exception as e:
                # Disconnects will naturally trigger an exception when send_text fails
                pass
            finally:
                state["running"] = False

    # Run both tasks concurrently
    await asyncio.gather(receiver_task(), inference_task())
    print(f"🔌 Client Disconnected: {session_id}")

if __name__ == "__main__":
    # This allows you to run the server with `python server/main.py`
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)