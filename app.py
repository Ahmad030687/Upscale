import os
import sys

# --- 🔥 FIX FOR BASICSR BUG ---
import torchvision
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

from fastapi import FastAPI, Query
import requests
import cv2
import numpy as np
import torch
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

# Global variables
face_enhancer = None

def load_models():
    global face_enhancer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model Setup
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True, # RAM bachane ke liye half precision
        device=device
    )

    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler,
        device=device
    )
    print("✅ AHMAD RDX AI Models Loaded!")

@app.on_event("startup")
async def startup():
    load_models()

@app.get("/")
def home():
    return {"status": "Online", "owner": "AHMAD RDX"}

@app.get("/upscale")
async def upscale(url: str = Query(...)):
    if face_enhancer is None:
        return {"success": False, "error": "Models are still loading..."}
    try:
        resp = requests.get(url)
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        _, buffer = cv2.imencode(".jpg", output)
        
        # Upload to ImgBB
        up_resp = requests.post("https://api.imgbb.com/1/upload", 
                                {"key": "be2413580cb5722d0184157f49b74c0d"}, 
                                files={"image": ("result.jpg", buffer.tobytes())})
        return {"success": True, "result": up_resp.json()["data"]["url"]}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
