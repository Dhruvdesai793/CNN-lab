INPUT_VIDEO = "assets/drive.mp4"
OUTPUT_VIDEO = "results/output.mp4"

model = load_model()

cap = cv2.VideoCapture(INPUT_VIDEO)

writer = ...

while True:

    ret, frame = cap.read()

    if not ret:
        break

    prediction = predict_frame(model, frame)

    overlay = overlay_mask(frame, prediction)

    writer.write(overlay)

cap.release()
writer.release()