!pip install ultralytics
 
 
!pip install roboflow ultralytics
from roboflow import Roboflow
rf = Roboflow(api_key="xxxxxxxxxxxxxx")
project = rf.workspace("sarans-workspace-6mprp").project("table-tennis-table-is")
version = project.version(1)
dataset = version.download("yolov26")
 
 
from ultralytics import YOLO
model = YOLO('bestPingPongAi.pt')
results = model.train(data="/content/table-tennis-table-IS-1/data.yaml", epochs=100, imgsz=640)