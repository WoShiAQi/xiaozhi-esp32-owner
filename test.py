from ultralytics import YOLO

model = YOLO("best.pt")
results = model("image.png")  # 假设您有一张名为 "image.jpg" 的图像
results=model.predict(source="image.png",save=True,show=True)
# results=model.predict(source="123.mp4",show=True)
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        print(f"检测到的类别: {class_id}, 置信度: {confidence:.2f}, 坐标: ({x1}, {y1}), ({x2}, {y2})") # 打印检测结果