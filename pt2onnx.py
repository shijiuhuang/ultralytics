from ultralytics import YOLO

# Load a model

model = YOLO(r'D:\yolov11\ultralytics\ultralytics\yolo11n-pose.pt')  # load a custom trained model

# Export the model
model.export(format="onnx")