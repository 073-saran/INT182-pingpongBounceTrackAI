import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
 

 

ball_model = YOLO("/content/bestPingPongAi.pt")
table_model = YOLO("/content/bestTableTennisTableAi.pt")
 
video_path = "/content/One Of The Best Match Points Ever! 😱🔥 - TableTennisDaily (1080p).mp4"
output_path = "output_bounce.mp4"
 
cap = cv2.VideoCapture(video_path)
 

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # ฟอร์แมตไฟล์ .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 

ret, first_frame = cap.read()
if not ret:
    print("ไม่สามารถอ่านวิดีโอได้")
    exit()
 
print("กำลังให้ AI ค้นหาพื้นที่โต๊ะปิงปอง...")
table_results = table_model.predict(first_frame, conf=0.5, verbose=False)
 
table_polygon = None
 

if hasattr(table_results[0], 'masks') and table_results[0].masks is not None:
    mask_points = table_results[0].masks.xy[0]
    contour = np.array(mask_points, dtype=np.int32)
 

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
 

    if len(approx) == 4:
        table_polygon = approx.reshape((-1, 1, 2))
        print("✅ AI ค้นพบพิกัด 4 มุมของโต๊ะอัตโนมัติที่:\n", table_polygon.flatten())
    else:
        print(f"⚠️ AI เจอพื้นที่โต๊ะ แต่แปลงเป็น 4 มุมไม่สำเร็จ (ได้ {len(approx)} มุม) จะใช้พิกัดสำรอง")
 

if table_polygon is None:
    print("⚠️ ใช้พิกัดโต๊ะแบบตั้งค่าเอง (Default)")
    table_polygon = np.array([
        [300, 695],  # มุมบนซ้าย
        [783, 695],  # มุมบนขวา
        [810, 966],  # มุมล่างขวา
        [289, 966]   # มุมล่างซ้าย
    ], np.int32).reshape((-1, 1, 2))
 
 

BALL_CLASS_ID = 0
trajectory = []
bounces = []
 
print("กำลังประมวลผลวิดีโอหาจุดตกลูกปิงปอง... อาจใช้เวลาสักครู่")
 

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 

    results = ball_model.predict(frame, conf=0.3, verbose=False)
 
 
    cv2.polylines(frame, [table_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
 
    for box in results[0].boxes:
        if int(box.cls[0]) == BALL_CLASS_ID:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
 
            trajectory.append((cx, cy))
 

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
 

            if len(trajectory) >= 3:
                y_prev2 = trajectory[-3][1]
                y_prev1 = trajectory[-2][1]
                y_curr  = trajectory[-1][1]
 

                if y_prev1 > y_prev2 and y_prev1 > y_curr:
                    bounce_point = trajectory[-2]
 

                    is_inside_table = cv2.pointPolygonTest(table_polygon, bounce_point, False) >= 0
 
                    if is_inside_table:
 
                        if bounce_point not in bounces:
                            bounces.append(bounce_point)
                            print(f"เจอจุดตกบนโต๊ะที่พิกัด: {bounce_point}")
 

    for bx, by in bounces:
        cv2.circle(frame, (bx, by), 15, (0, 255, 255), 3)
 

    out.write(frame)
 
cap.release()
out.release()
print(f"ประมวลผลวิดีโอเสร็จสิ้น! สามารถดาวน์โหลดไฟล์ {output_path} ได้เลยครับ")
print("-" * 50)
 
 

 
print("กำลังสร้างกราฟ Heatmap...")
total_bounces = len(bounces)
print(f"✅ จำนวนที่ลูกปิงปองตกกระทบโต๊ะทั้งหมด: {total_bounces} ครั้ง")
 
if total_bounces > 0:
    x_coords = [b[0] for b in bounces]
    y_coords = [b[1] for b in bounces]
 
    plt.figure(figsize=(12, 8))
 

    sns.kdeplot(x=x_coords, y=y_coords, cmap="YlOrRd", fill=True, bw_adjust=0.5, alpha=0.7)
 

    table_points = table_polygon.reshape(-1, 2)
    table_x = np.append(table_points[:, 0], table_points[0, 0])
    table_y = np.append(table_points[:, 1], table_points[0, 1])
    plt.plot(table_x, table_y, color='blue', linewidth=3, linestyle='--', label="Table Boundary")