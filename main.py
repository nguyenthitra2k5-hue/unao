from ultralytics import YOLO
import cv2
import os

# =========================
# CẤU HÌNH
# =========================
MODEL_PATH = "best.pt"
IMAGE_PATH = "123.jpg"
OUTPUT_PATH = "ket_qua_u_nao.jpg"

CONFIDENCE = 0.4
IMG_SIZE = 640

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

print("Các nhãn của model:")
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
    imgsz=IMG_SIZE,
    verbose=False
)

detection_count = 0

# =========================
# XỬ LÝ KẾT QUẢ
# =========================
for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        detection_count += 1

        # Tọa độ bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Confidence
        confidence = float(box.conf[0])

        # Class ID
        class_id = int(box.cls[0])

        # Class gốc của model
        original_class = model.names[class_id]

        # Tên hiển thị
        class_name = "U Nao"

        # =========================
        # VẼ KHUNG
        # =========================
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        # =========================
        # LABEL
        # =========================
        label = f"{class_name} {confidence * 100:.1f}%"

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

        # Nền đỏ cho label
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

        print(
            f"Phát hiện: {original_class} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

# =========================
# TRẠNG THÁI
# =========================
if detection_count > 0:

    status = f"PHAT HIEN {detection_count} VUNG NGHI NGO U NAO"

    cv2.putText(
        img,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

else:

    status = "KHONG PHAT HIEN VUNG NGHI NGO"

    cv2.putText(
        img,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# =========================
# LƯU ẢNH
# =========================
cv2.imwrite(
    OUTPUT_PATH,
    img
)

print()
print("==============================")
print("KẾT QUẢ")
print("==============================")
print("Số vùng phát hiện:", detection_count)
print("Ảnh kết quả:", OUTPUT_PATH)

# =========================
# HIỂN THỊ
# =======================
cv2.imshow(
    "AI Nhan Dien U Nao",
    img
)

print("Nhấn phím bất kỳ để thoát.")

cv2.waitKey(0)
cv2.destroyAllWindows()
