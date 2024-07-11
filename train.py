from ultralytics import YOLOv10

model = YOLOv10('weight/yolov10x.pt')
results = model.train(data="/mnt/d/docker_volume/room_of_nhanhuynh/ai_butler/CAD-ShibaSangyo/full_build_from_scratch_nhanhuynh/CORE-2/data.yaml", epochs=2000, imgsz=1280,  batch=2)