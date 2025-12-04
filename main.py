from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
import base64
import io
from PIL import Image
import numpy as np
import time
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

app = FastAPI(title="GAN Deblur API", version="0.2.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    global _model
    if _model is not None:
        return _model
    # Add training package path to import generator
    server_dir = Path(__file__).resolve().parent
    proj_root = server_dir.parent
    training_pkg = proj_root / 'training'
    sys.path.append(str(training_pkg))
    from models.generator import ResNetGenerator  # type: ignore

    ckpt_path = proj_root / 'runs' / 'deblurganv2' / 'ckpt_160.pt'
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")

    model = ResNetGenerator(in_ch=3, base_ch=64, blocks=8).to(_device)
    state = torch.load(str(ckpt_path), map_location=_device)
    model.load_state_dict(state['G'], strict=True)
    model.eval()
    _model = model
    return _model

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert('RGB')).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return t

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    img8 = (t * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(img8)


@app.post("/deblur")
async def deblur(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image bytes.")

    model = _load_model()

    # Preprocess
    x = _pil_to_tensor(img).to(_device)
    _, _, h, w = x.shape
    # Pad to multiple of 4 (due to two downsamples in generator)
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    start = time.perf_counter()
    with torch.no_grad():
        use_amp = _device.type == 'cuda'
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y = model(x)
        else:
            y = model(x)
    infer_ms = int((time.perf_counter() - start) * 1000)

    # Unpad
    if pad_h or pad_w:
        y = y[:, :, :h, :w]

    out_img = _tensor_to_pil(y)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Metrics: ground-truth not available at inference; return timing only
    return JSONResponse(
        content={
            "image": b64,
            "metrics": {},
            "inference_ms": infer_ms,
        }
    )
