from ultralytics import YOLO
from pathlib import Path

current_dir = Path(__file__).parent.parent.parent
print(current_dir)

model =YOLO(current_dir/"models"/"yolov8s.pt")

results =model.train(
    data="data.yaml",
    epochs=40,
    imgsize=416,
    batch=16,
    name='train YOLO first part',
    device=0,
    workers=4,
)