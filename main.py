from ultralytics import YOLO
import cv2
import os

# =========================
# CẤU HÌNH
# =========================
MODEL_PATH = "best (2).pt"
IMAGE_PATH = "images (1).jpg"
OUTPUT_PATH = "ket_qua_u_nao.jpg"

# Ngưỡng confidence
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
    imgsz=640,
    verbose=False
)

detections = 0

# =========================
# XỬ LÝ KẾT QUẢ
# =========================
for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        # Tọa độ bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Độ tin cậy
        confidence = float(box.conf[0])

        # Class ID
        class_id = int(box.cls[0])

        # Vì model của bạn chỉ có class tumor
        class_name = "U Nao"

        detections += 1

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

        # Nội dung nhãn
        label = f"{class_name} {confidence * 100:.1f}%"

        # Tính kích thước chữ
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )

        # Vị trí chữ
        text_y = max(y1 - 10, text_h + 10)

        # Nền đỏ
        cv2.rectangle(
            img,
            (x1, text_y - text_h - 10),
            (x1 + text_w + 10, text_y + 5),
            (0, 0, 255),
            -1
        )

        # Chữ trắng
        cv2.putText(
            img,
            label,
            (x1 + 5, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # In ra terminal
        print(
            f"Phát hiện {class_name} | "
            f"Confidence: {confidence * 100:.2f}% | "
            f"Box: ({x1}, {y1}, {x2}, {y2})"
        )

# =========================
# KẾT LUẬN
# =========================
if detections > 0:

    print(f"\nPhát hiện {detections} vùng nghi ngờ u não.")

    cv2.putText(
        img,
        f"PHAT HIEN: {detections} VUNG U NAO",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

else:

    print("\nKhông phát hiện vùng nghi ngờ.")

    cv2.putText(
        img,
        "KHONG CO DETECTION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

# =========================
# LƯU KẾT QUẢ
# =========================
cv2.imwrite(OUTPUT_PATH, img)

print(f"Đã lưu ảnh kết quả: {OUTPUT_PATH}")

# =========================
# HIỂN THỊ
# =========================
cv2.imshow("Nhan dien U Nao", img)

print("Nhan phim bat ky de dong cua so...")

cv2.waitKey(0)
cv2.destroyAllWindows()