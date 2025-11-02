from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import requests
from pathlib import Path

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Egyptian Museum Artifact Detection API",
    description="YOLO-based artifact detection for Egyptian museum exhibits",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Fixed Hugging Face URL - using direct download link
REMOTE_URL = (
    "https://huggingface.co/Nouran123/egyptian-artifact-yolo/resolve/main/best.pt"
)

# Global model variable
model = None
model_loading = False
model_error = None

# Egyptian Artifact ID mapping - 84 classes
ARTIFACT_MAPPING = {
    0: "Akhenaten",
    1: "Amenhotep III",
    2: "Amenhotep III and Tiye",
    3: "Amenhotep III with Plate",
    4: "Augustus",
    5: "Bent Pyramid of King Sneferu",
    6: "Black Granite Bust of Mentuemhat",
    7: "Bust of Isis",
    8: "Clossal Head of the god Serapis",
    9: "Clossal head of Senwosret 1",
    10: "Coffin of Ahmose I",
    11: "Colossal Statue of Amenhotep III",
    12: "Colossal Statue of God Ptah",
    13: "Colossal Statue of Hormoheb",
    14: "Colossal Statue of King Senwosret IlI",
    15: "Colossal Statue of Middle Kingdom King",
    16: "Colossal Statue of Queen Hatshepsut",
    17: "Colossal Statue of Ramesses II",
    18: "Colossal Statue of Ramesses II beloved of Ptah",
    19: "Colossoi of Memnon",
    20: "Colossus of Senuseret I",
    21: "Column of Merenptah",
    22: "Granite Statue of Osiris",
    23: "Granite Statue of Tutankhamun",
    24: "Great Pyramids of Giza",
    25: "Grey Granite of Ramesses II",
    26: "Hathor Capital",
    27: "Hatshepsut face",
    28: "Head Statue of Amenhotep III",
    29: "Head Statue of Amenhotep iii",
    30: "Head of Userkaf",
    31: "Hor I",
    32: "Isis with her child",
    33: "King Amenemhat 3",
    34: "King Thutmose III",
    35: "Mask of Thuya",
    36: "Mask of Tutankhamun",
    37: "Mask of Yuya",
    38: "Menkaure Statue",
    39: "Mentuhotep Nebhetpre",
    40: "Naos of Senwosert I",
    41: "Nefertiti",
    42: "Obelsik Tip of Hatshepsut",
    43: "Offering table of Amenemhat 6",
    44: "Pyramid of Djoser",
    45: "Rhetorical Stela of King Ramesses ll",
    46: "Seated Statue of Amenhotep III",
    47: "Seated Statue of Djoser",
    48: "Seated Statue of God Sekhmet",
    49: "Seated Statue of Ramesses II",
    50: "Seated Statue of Ramesses II and God Ptah",
    51: "Seated Statue of Thutmose III",
    52: "Senwosret III",
    53: "Sphinx",
    54: "Sphinx of Amenmhat III",
    55: "Sphinx of Kings Ramesses ll - Merenptah",
    56: "Standing Statue of King Ramses II",
    57: "Standing Statue of Thutmose III",
    58: "Statue Head of Akhenaten",
    59: "Statue of Amenhotep III and God Re-Horakhty",
    60: "Statue of Amenmhat I",
    61: "Statue of Amun and King",
    62: "Statue of Ankhesenamun",
    63: "Statue of Carcala",
    64: "Statue of God Ptah Ramesses ll Goddess Sekhmet",
    65: "Statue of God Ra-Horakhty",
    66: "Statue of Khafre",
    67: "Statue of Khufu",
    68: "Statue of King Ramesses ll - Goddess Anath",
    69: "Statue of King Ramses II Grand Egyptian Museum",
    70: "Statue of King Ramses II Luxor Temple",
    71: "Statue of King Sety Il Holding Standards",
    72: "Statue of King Zoser",
    73: "Statue of Mentuhotep II",
    74: "Statue of Merenptah as standard Bearer",
    75: "Statue of Osiris",
    76: "Statue of Queen Metnoforet",
    77: "Statue of Ramesses III as standard Bearer",
    78: "Statue of Snefru",
    79: "Statue of Sobekhotep V",
    80: "Statue of Tutankhamun",
    81: "Stela of king Snefero",
    82: "bust of Ramesses II",
    83: "kneeling statue of queen hatshibsut",
}


def ensure_model():
    """Download model if not present locally with proper error handling"""
    if not os.path.exists(MODEL_PATH):
        Path(os.path.dirname(MODEL_PATH) or ".").mkdir(parents=True, exist_ok=True)
        logger.info("📥 Downloading model from Hugging Face...")

        try:
            # Use headers to ensure we get the actual file
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(
                REMOTE_URL, headers=headers, stream=True, timeout=300
            )
            response.raise_for_status()

            # Check if we got HTML instead of binary
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                raise ValueError(
                    f"Received HTML instead of model file. URL might be incorrect: {REMOTE_URL}"
                )

            # Download with progress
            total_size = int(response.headers.get("content-length", 0))
            logger.info(f"Downloading model ({total_size / 1024 / 1024:.2f} MB)...")

            with open(MODEL_PATH, "wb") as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

            logger.info("✅ Download complete!")

            # Verify the file is valid
            file_size = os.path.getsize(MODEL_PATH)
            logger.info(f"Downloaded model size: {file_size} bytes")

            with open(MODEL_PATH, "rb") as f:
                header = f.read(10)
                logger.info(f"File header: {header}")

                # Check if file starts with HTML tags (invalid)
                if header.startswith(b"<html") or header.startswith(b"<!DOCT"):
                    raise ValueError("Downloaded file is HTML, not a valid model file")

            return MODEL_PATH

        except Exception as e:
            # Clean up invalid file
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            logger.error(f"Failed to download model: {e}")
            raise

    return MODEL_PATH


def load_model():
    global model, model_loading, model_error
    model_loading = True

    try:
        from ultralytics import YOLO

        model_path = ensure_model()
        logger.info(f"Loading model from: {model_path}")

        model = YOLO(model_path)
        logger.info("✅ YOLO model loaded successfully!")

        model_loading = False
        model_error = None
        return model

    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(f"❌ {error_msg}")
        model_error = error_msg
        model_loading = False
        return None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup - load model in background"""
    logger.info("🚀 Starting Egyptian Museum Artifact Detection API...")
    logger.info(f"📋 Loaded {len(ARTIFACT_MAPPING)} artifact classes")

    # Try to load model in background
    import threading

    def load_in_background():
        load_model()

    threading.Thread(target=load_in_background, daemon=True).start()
    logger.info("✅ API started - model loading in background")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Egyptian Museum Artifact Detection API",
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "model_error": model_error,
        "model_path": MODEL_PATH,
        "num_artifacts": len(ARTIFACT_MAPPING),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/health")
async def health():
    """Additional health check"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_ready": model is not None,
        "model_loading": model_loading,
        "model_error": model_error,
    }


@app.post("/detect-artifact")
async def detect_artifact(file: UploadFile = File(...)):
    """Detect Egyptian artifacts in uploaded images using YOLO"""

    # Check if model is ready
    current_model = model
    if current_model is None:
        if model_loading:
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Please try again in a few seconds.",
            )
        elif model_error:
            raise HTTPException(
                status_code=503, detail=f"Model failed to load: {model_error}"
            )
        else:
            # Try to load now
            current_model = load_model()
            if current_model is None:
                raise HTTPException(
                    status_code=503,
                    detail="YOLO model not available. Please check server logs.",
                )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image."
        )

    try:
        import cv2
        import numpy as np

        # Read image file
        contents = await file.read()
        logger.info(f"📸 Received image: {file.filename} ({len(contents)} bytes)")

        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image. Please upload a valid image file.",
            )

        logger.info(f"🔍 Running inference on image shape: {image.shape}")

        # Run YOLO inference
        results = current_model(image, conf=CONFIDENCE_THRESHOLD)

        # Process results
        if len(results) == 0 or len(results[0].boxes) == 0:
            logger.info("❌ No artifacts detected in image")
            return {
                "artifact_id": None,
                "confidence": 0.0,
                "message": "No artifact detected. Try a clearer image.",
            }

        # Get the detection with highest confidence
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        logger.info(f"📊 Found {len(boxes)} detection(s)")

        # Find best detection
        best_idx = np.argmax(confidences)
        best_confidence = float(confidences[best_idx])
        best_class = int(classes[best_idx])

        # Map class to artifact name
        artifact_name = ARTIFACT_MAPPING.get(
            best_class, f"Unknown Artifact (Class {best_class})"
        )

        logger.info(f"✅ Detected: {artifact_name} (confidence: {best_confidence:.2%})")

        return {
            "artifact_id": artifact_name,
            "confidence": best_confidence,
            "class_id": best_class,
            "detections_count": len(boxes),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Detection failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {
            "loaded": False,
            "loading": model_loading,
            "error": model_error,
            "message": "Model not loaded yet",
        }

    return {
        "loaded": True,
        "model_path": MODEL_PATH,
        "num_classes": len(ARTIFACT_MAPPING),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total_artifacts": len(ARTIFACT_MAPPING),
    }


@app.get("/artifacts")
async def list_artifacts():
    """Get complete list of all detectable artifacts"""
    artifacts_list = [
        {"class_id": class_id, "artifact_name": artifact_name}
        for class_id, artifact_name in ARTIFACT_MAPPING.items()
    ]
    return {"artifacts": artifacts_list, "total": len(artifacts_list)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
