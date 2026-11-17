!pip install roboflow ultralytics
 
from roboflow import Roboflow
rf = Roboflow(api_key="bXyNBuCvb3feAgTVRQNn")
project = rf.workspace("sarans-workspace-6mprp").project("table-tennis-table-is")
version = project.version(1)
dataset = version.download("yolo26")
 
from ultralytics import YOLO
import shutil
 
model = YOLO("yolov8n.pt")
 
yaml_path = f"{dataset.location}/data.yaml"
 
results = model.train(
    data=yaml_path,
    epochs=50,           
    imgsz=640,           
    batch=16,            
    project="runs",      
    name="table_tennis"  
)
 
best_weights_path = "runs/table_tennis/weights/best.pt"
final_model_name = "bestTableTennisTableAi.pt"
 
shutil.copy(best_weights_path, final_model_name)
print(f"✅ ฝึกสอนเสร็จสิ้น! บันทึกไฟล์โมเดลพร้อมใช้งานที่: {final_model_name}")