import torch
import onnx

from radar_pipeline.model_cnn_only import CNNTemporal


CHECKPOINT = (
    r"outputs\experiments\cnn_temporal_13feature_v2_stm32"
    r"\fold_1\cnn_temporal_event_centered.pt"
)

OUTPUT = "cnn_temporal_60frame_stm32.onnx"


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

print("Model class:", ckpt["model_class"])
print("Clip length:", ckpt["clip_length_frames"])
print("Features:", ckpt["n_features"])
print("Classes:", ckpt["class_order"])


# --------------------------------------------------
# Rebuild model
# --------------------------------------------------

model = CNNTemporal(
    n_features=int(ckpt["n_features"]),
    n_classes=len(ckpt["class_order"]),
    dropout=float(
        ckpt["train_config"].get("dropout", 0.25)
    ),
)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()


# --------------------------------------------------
# Fixed STM32 input
#
# [batch, frames, features]
# [1, 60, 13]
# --------------------------------------------------

dummy_input = torch.randn(
    1,
    int(ckpt["clip_length_frames"]),
    int(ckpt["n_features"]),
    dtype=torch.float32,
)


# Verify PyTorch
with torch.no_grad():
    output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print("Output:", output)


# --------------------------------------------------
# Export with legacy exporter
# --------------------------------------------------

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT,
    input_names=["radar_input"],
    output_names=["logits"],
    opset_version=17,
    dynamo=False,
    do_constant_folding=True,
)


# --------------------------------------------------
# Validate ONNX
# --------------------------------------------------

onnx_model = onnx.load(OUTPUT)
onnx.checker.check_model(onnx_model)

print("")
print("ONNX validation: OK")
print("Saved:", OUTPUT)