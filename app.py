from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

MODEL_PATH = "best.pt"
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

CONFIDENCE = 0.25
IMG_SIZE = 640

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

print("Các nhãn của model:")
print(model.names)


@app.route("/", methods=["GET", "POST"])
def index():

    original_image = None
    result_image = None
    detections = []
    message = None

    if request.method == "POST":

        if "image" not in request.files:
            message = "Chưa chọn ảnh."
            return render_template(
                "index.html",
                message=message
            )

        file = request.files["image"]

        if file.filename == "":
            message = "Chưa chọn ảnh."
            return render_template(
                "index.html",
                message=message
            )

        extension = os.path.splitext(file.filename)[1].lower()

        allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]

        if extension not in allowed_extensions:
            message = "Định dạng ảnh không được hỗ trợ."
            return render_template(
                "index.html",
                message=message
            )

        # Tạo tên file mới
        filename = str(uuid.uuid4()) + extension

        upload_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(upload_path)

        original_image = "/" + upload_path.replace("\\", "/")

        # Đọc ảnh
        img = cv2.imread(upload_path)

        if img is None:
            message = "Không đọc được ảnh."
            return render_template(
                "index.html",
                message=message
            )

        # YOLO11 nhận diện
        results = model.predict(
            source=img,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            verbose=False
        )

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                original_name = model.names[class_id]

                # Tên hiển thị
                class_name = "U Nao"

                detections.append({
                    "class": class_name,
                    "original_class": original_name,
                    "confidence": round(
                        confidence * 100,
                        2
                    )
                })

                # Vẽ bounding box
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                label = (
                    f"{class_name} "
                    f"{confidence * 100:.1f}%"
                )

                (text_width, text_height), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )

                text_y = max(
                    y1 - 10,
                    text_height + 10
                )

                # Nền đỏ
                cv2.rectangle(
                    img,
                    (x1, text_y - text_height - 10),
                    (x1 + text_width + 10, text_y + 5),
                    (0, 0, 255),
                    -1
                )

                # Chữ trắng
                cv2.putText(
                    img,
                    label,
                    (x1 + 5, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        # Lưu ảnh kết quả
        result_filename = "result_" + filename

        result_path = os.path.join(
            RESULT_FOLDER,
            result_filename
        )

        cv2.imwrite(
            result_path,
            img
        )

        result_image = "/" + result_path.replace("\\", "/")

    return render_template(
        "index.html",
        original_image=original_image,
        result_image=result_image,
        detections=detections,
        message=message
    )


if __name__ == "__main__":

    print("================================")
    print("AI NHẬN DIỆN U NÃO - YOLO11")
    print("================================")
    print("Web: http://127.0.0.1:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )