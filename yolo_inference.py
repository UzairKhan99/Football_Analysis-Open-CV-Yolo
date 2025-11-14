from ultralytics import YOLO
model = YOLO('models/best.pt')
results = model.predict(
    "Input_Videos/08fd33_4.mp4", save=True)

print(results[0])


for i in results[0].boxes:
    print(i)
