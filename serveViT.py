from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
import asyncio

app = FastAPI()

MODEL_PATH = "/workspace/shared/public/mvppgf/models/ViT_pgf"

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load model & processor
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

# Optional optimization
try:
    model = torch.compile(model)
except Exception as e:
    print(f"torch.compile skipped: {e}")

# ✅ Ambil label langsung dari model (lebih aman)
labels = model.config.id2label
print(f"Labels: {labels}")

# ✅ Thread-safe queue
request_queue = asyncio.Queue()

MAX_BATCH_SIZE = 8
BATCH_TIMEOUT = 0.01  # 10 ms


async def batch_worker():
    while True:
        batch = []

        try:
            # ambil minimal 1 request
            item = await request_queue.get()
            batch.append(item)

            # kumpulkan batch tambahan
            try:
                for _ in range(MAX_BATCH_SIZE - 1):
                    item = await asyncio.wait_for(
                        request_queue.get(),
                        timeout=BATCH_TIMEOUT
                    )
                    batch.append(item)
            except asyncio.TimeoutError:
                pass

            images = [item["image"] for item in batch]
            futures = [item["future"] for item in batch]

            print(f"[INFO] Processing batch size: {len(batch)}")

            # preprocess
            # inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            # inference
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = logits.argmax(-1).tolist()

                # 👉 ambil confidence (optional tapi berguna)
                probs = torch.softmax(logits, dim=-1)
                confidences = probs.max(dim=-1).values.tolist()

            # kirim hasil
            for pred, conf, future in zip(preds, confidences, futures):
                if not future.done():
                    future.set_result({
                        "class_id": pred,
                        "label": labels.get(pred, "unknown"),
                        "confidence": round(conf, 4)
                    })

        except Exception as e:
            print(f"[ERROR] Batch processing failed: {e}")

            # jangan sampai request hang
            for item in batch:
                future = item["future"]
                if not future.done():
                    future.set_result({"error": str(e)})


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())


@app.get("/")
def root():
    return {
        "status": "ok",
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    await request_queue.put({
        "image": image,
        "future": future
    })

    result = await future
    return result
