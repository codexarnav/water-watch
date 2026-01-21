"""
Agent B: Unified Perception Agent

- semantic_bind:
    - Text  -> CLIP ViT-B/32
    - Image -> CLIP ViT-B/32
    - Audio -> CLAP
    - Video -> Keyframes -> CLIP -> mean pool
- lexical_sparse (text only): SPLADE++
"""

from typing import Dict, Optional
import uuid
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import ClapProcessor, ClapModel
import librosa


# =============================
# Device
# =============================

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# CLIP (Text + Image)
# =============================

CLIP_REPO = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO)
clip_model = CLIPModel.from_pretrained(CLIP_REPO).to(device).eval()


# =============================
# CLAP (Audio)
# =============================

CLAP_REPO = "laion/clap-htsat-unfused"

clap_processor = ClapProcessor.from_pretrained(CLAP_REPO)
clap_model = ClapModel.from_pretrained(CLAP_REPO).to(device).eval()


# =============================
# SPLADE++ (Lexical Sparse)
# =============================

SPLADE_REPO = "naver/splade-cocondenser-ensembledistil"

splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_REPO)
splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_REPO).to(device).eval()


# =============================
# Embedding Helpers
# =============================

@torch.no_grad()
def embed_clip_text(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    return clip_model.get_text_features(**inputs).squeeze().cpu().tolist()


@torch.no_grad()
def embed_clip_image(image_uri: str):
    """
    Support local file paths and HTTP(S) URLs for images.
    Downloads remote images into memory before opening with PIL.
    """
    from io import BytesIO

    # Lazy import to avoid adding requests as a global dependency if unused
    if isinstance(image_uri, str) and image_uri.startswith(("http://", "https://")):
        try:
            import requests

            resp = requests.get(image_uri, timeout=15)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to download/open image from URL: {e}")
    else:
        # If the caller passed a bare filename (e.g. "photo.jpg"),
        # prefer the project's `images/` folder. Otherwise try the
        # provided path as-is.
        from pathlib import Path

        p = Path(image_uri)

        if p.is_absolute() and p.exists():
            image = Image.open(p).convert("RGB")
        elif p.exists():
            image = Image.open(p).convert("RGB")
        else:
            images_dir = Path(__file__).parent / "images"
            file_path = images_dir / image_uri
            if file_path.exists():
                image = Image.open(file_path).convert("RGB")
            else:
                raise RuntimeError(
                    f"Image file not found. Tried path '{image_uri}' and '{file_path}'"
                )

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    features = clip_model.get_image_features(**inputs)
    return features.squeeze().cpu().tolist()


@torch.no_grad()
def embed_clap_audio(audio_path: str):
    audio, sr = librosa.load(audio_path, sr=48000)
    inputs = clap_processor(audios=audio, sampling_rate=sr, return_tensors="pt").to(device)
    return clap_model.get_audio_features(**inputs).squeeze().cpu().tolist()


@torch.no_grad()
def embed_video_clip_frames(
    video_path: str,
    frame_interval_sec: int = 2,
    num_segments: int = 4
):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        return None

    frame_step = int(fps * frame_interval_sec)

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        return None

    total_frames = len(frames)
    segment_size = max(1, total_frames // num_segments)

    segment_embeddings = []

    for seg_idx in range(num_segments):
        start = seg_idx * segment_size
        end = start + segment_size

        segment_frames = frames[start:end]

        if len(segment_frames) == 0:
            # pad with zeros if video is too short
            segment_embeddings.append(
                np.zeros(512, dtype=np.float32)
            )
            continue

        emb_list = []

        for img in segment_frames:
            inputs = clip_processor(
                images=img,
                return_tensors="pt"
            ).to(device)

            emb = clip_model.get_image_features(**inputs)
            emb_list.append(emb.squeeze().cpu().numpy())

        # mean-pool *within segment*
        segment_mean = np.mean(emb_list, axis=0)
        segment_embeddings.append(segment_mean)

    video_embedding = np.concatenate(segment_embeddings, axis=0)

    return video_embedding.tolist()



@torch.no_grad()
def embed_lexical_sparse(text: str):
    inputs = splade_tokenizer(text, return_tensors="pt").to(device)
    logits = splade_model(**inputs).logits

    scores = torch.max(torch.log1p(torch.relu(logits)), dim=1).values.squeeze()
    indices = scores.nonzero().squeeze().cpu().tolist()
    values = scores[indices].cpu().tolist()

    return {"indices": indices, "values": values}


# =============================
# Agent B Core
# =============================

def agent_b_perceive(routed_signal: Dict) -> Optional[Dict]:
    modality = routed_signal.get("modality")
    payload = routed_signal.get("payload", {})
    context = routed_signal.get("context", {})

    percept_id = str(uuid.uuid4())

    # ---- TEXT ----
    if modality == "text":
        text = payload.get("text")
        if not text:
            return None

        return {
            "percept_id": percept_id,
            "modality": "text",
            "semantic_bind": embed_clip_text(text),
            "lexical_sparse": embed_lexical_sparse(text),
            "context": context,
            "raw_ref": {"text": text}
        }

    # ---- IMAGE ----
    if modality == "image":
        uri = payload.get("image_uri")
        if not uri:
            return None

        return {
            "percept_id": percept_id,
            "modality": "image",
            "semantic_bind": embed_clip_image(uri),
            "context": context,
            "raw_ref": {"image_uri": uri}
        }

    # ---- AUDIO ----
    if modality == "audio":
        uri = payload.get("audio_uri")
        if not uri:
            return None

        return {
            "percept_id": percept_id,
            "modality": "audio",
            "semantic_bind": embed_clap_audio(uri),
            "context": context,
            "raw_ref": {"audio_uri": uri}
        }

    # ---- VIDEO ----
    if modality == "video":
        uri = payload.get("video_uri")
        if not uri:
            return None

        video_emb = embed_video_clip_frames(uri)
        if video_emb is None:
            return None

        return {
            "percept_id": percept_id,
            "modality": "video",
            "semantic_bind": video_emb,
            "context": context,
            "raw_ref": {"video_uri": uri}
        }

    return None

# =============================
# Demo Run
# =============================

if __name__ == "__main__":

    demo_routed_signal = {
        "modality": "video",
        "payload": {
            "video_uri": "videos\9736660-hd_1920_1080_25fps.mp4"
        },
        "context": {
            "timestamp": "2026-01-21T10:12:00Z",
            "geohash": "ttnwv6d",
            "source": "whatsapp"
        }
    }

    print("Running Agent B perception demo...\n")

    percept = agent_b_perceive(demo_routed_signal)

    if percept is None:
        print("❌ Percept creation failed")
    else:
        print("✅ Percept created successfully\n")

        print("Percept ID:", percept["percept_id"])
        print("Modality:", percept["modality"])
        print("Context:", percept["context"])

        print("\nSemantic Bind Vector Dimension:",
              len(percept["semantic_bind"]))

        if "lexical_sparse" in percept:
            print("Lexical Sparse Terms:",
                  len(percept["lexical_sparse"]["indices"]))

        print("\nRaw Reference:")
        print(percept["raw_ref"])
