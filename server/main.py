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
    print(f"🔌 Client Connected: {websocket.client}")
    
    # Stateful buffer for this connection
    frame_buffer = deque(maxlen=50)
    
    def run_inference(tensor):
        """Runs the PyTorch model synchronously (will be called in thread pool)"""
        with torch.no_grad():
            logits = model(tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            conf = float(torch.max(torch.softmax(logits, dim=1)))
        return pred_idx, conf

    try:
        while True:
            # 1. Receive Binary Data (Expects exactly 1 frame: 17x3 float32)
            data = await websocket.receive_bytes()
            
            if model is None:
                await websocket.send_text(json.dumps({"status": "Error", "code": -1}))
                continue

            # 2. Decode Binary -> NumPy
            frame = np.frombuffer(data, dtype=np.float32).reshape(17, 3)
            frame_buffer.append(frame)
            
            # If buffer isn't full yet, tell client to wait
            if len(frame_buffer) < 50:
                await websocket.send_text(json.dumps({"status": "Buffering", "prediction": -1, "confidence": 0}))
                continue

            # 3. Calculate Velocity & Prepare Tensor
            pos_seq = np.array(frame_buffer) # (50, 17, 3)
            vel_seq = np.diff(pos_seq, axis=0, prepend=pos_seq[0:1]) # (50, 17, 3)
            full_seq = np.concatenate((pos_seq, vel_seq), axis=2) # (50, 17, 6)
            
            # Input: (Batch, Channels, Time, Vertices) -> (1, 6, 50, 17)
            tensor = torch.tensor(full_seq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # 4. Asynchronous Inference (Unblocks Event Loop)
            pred_idx, conf = await asyncio.to_thread(run_inference, tensor)
            
            # 5. Send Result Back
            response = {
                "status": "OK",
                "prediction": pred_idx, # 0=Safe, 1=Warn, 2=Critical
                "confidence": conf
            }
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print(f"🔌 Client Disconnected: {websocket.client}")
    except Exception as e:
        print(f"❌ Error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    # This allows you to run the server with `python server/main.py`
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)