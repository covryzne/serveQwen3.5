from transformers import AutoModelForImageClassification, AutoImageProcessor

MODEL_NAME = "badzmaru/ViT-pornografi_ss-classifier"
SAVE_DIR = "/workspace/shared/public/mvppgf/models/ViT_pgf"

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    cache_dir=SAVE_DIR
)

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=SAVE_DIR
)