from ultralytics import YOLO
import cv2
import os

# =========================
# CẤU HÌNH
# =========================
MODEL_PATH = "best(2).pt"
IMAGE_PATH = "123.jpg"

CONFIDENCE = 0.05

# =========================
# KIỂM TRA FILE
# =========================
if not os.path.exists(MODEL_PATH):
    print(f"Không tìm thấy model: {MODEL_PATH}")
    exit()

if not os.path.exists(IMAGE_PATH):
    print(f"Không tìm thấy ảnh: {IMAGE_PATH}")
    exit()

# =========================
# LOAD MODEL
# =========================
print("Đang tải model...")

model = YOLO(MODEL_PATH)

print("Nhãn trong model:")
print(model.names)

# =========================
# ĐỌC ẢNH
# =========================
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Không đọc được ảnh.")
    exit()

# =========================
# NHẬN DIỆN
# =========================
results = model.predict(
    source=img,
    conf=CONFIDENCE,
    imgsz=640,
    verbose=False
)

count = 0

# =========================
# XỬ LÝ KẾT QUẢ
# =========================
for result in results:

    if result.boxes is None or len(result.boxes) == 0:
        continue

    for box in result.boxes:

        # Tọa độ vùng nhận diện
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Độ tin cậy
        confidence = float(box.conf[0])

        # ID lớp
        class_id = int(box.cls[0])

        # Tên nhãn gốc
        original_name = model.names[class_id]

        # Tên hiển thị
        class_name = "U Nao"

        count += 1

        print(
            f"Phát hiện {count}: "
            f"{original_name} | "
            f"Confidence = {confidence * 100:.2f}% | "
            f"Box = ({x1}, {y1}, {x2}, {y2})"
        )

        # =========================
        # VẼ KHUNG
        # =========================
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        label = f"{class_name} {confidence * 100:.1f}%"

        # =========================
        # NỀN NHÃN
        # =========================
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        label_y = max(y1, text_h + 10)

        cv2.rectangle(
            img,
            (x1, label_y - text_h - 10),
            (x1 + text_w + 10, label_y),
            (0, 0, 255),
            -1
        )

        # =========================
        # CHỮ
        # =========================
        cv2.putText(
            img,
            label,
            (x1 + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

# =========================
# KẾT QUẢ
# =========================
if count > 0:
    print()
    print(f"Phát hiện {count} vùng nghi ngờ U Nao.")
else:
    print()
    print("Không có detection vượt ngưỡng.")

# =========================
# HIỂN THỊ ẢNH
# =========================
cv2.imshow("Ket qua nhan dien U Nao", img)

print("Nhấn phím bất kỳ để đóng.")

cv2.waitKey(0)
cv2.destroyAllWindows()
/* mô hình nhận diện u não*/