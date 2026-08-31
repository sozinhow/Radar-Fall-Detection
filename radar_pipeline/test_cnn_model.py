import torch
from radar_pipeline.model_cnn_only import CNNTemporal


model=CNNTemporal()

x=torch.randn(1,60,13)

y=model(x)

print(y.shape)