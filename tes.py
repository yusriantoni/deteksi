import torch

# Load YOLOv8 model
helmet_model_path = 'D:/deteksi/104.pt'
helmet_model = torch.load(helmet_model_path)

# Get model classes
model_classes = helmet_model.names

print("Classes in the model:", model_classes)
