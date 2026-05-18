from transformers import ViTConfig, ViTForImageClassification

config = ViTConfig.from_pretrained(
    "/workspace/shared/public/mvppgf/models/ViT_pgf"
)

model = ViTForImageClassification.from_pretrained(
    "/workspace/shared/public/mvppgf/models/ViT_pgf",
    config=config
)

print("OK - model kamu kepakai")