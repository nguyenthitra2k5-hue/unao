from ultralytics import YOLO
import cv2
import os

# =========================
# CẤU HÌNH
# =========================
MODEL_PATH = "best.pt"

# Camera mặc định
CAMERA_ID = 0

# Ngưỡng confidence
CONFIDENCE = 0.25

# Kích thước ảnh YOLO
IMG_SIZE = 640

# Độ phân giải camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# =========================
# KIỂM TRA MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    print(f"Không tìm thấy model: {MODEL_PATH}")
    exit()

# =========================
# LOAD MODEL
# =========================
print("Đang tải model...")

model = YOLO(MODEL_PATH)

print("Các nhãn của model:")
print(model.names)

# =========================
# MỞ CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("Không mở được camera.")
    print("Thử đổi CAMERA_ID = 1 hoặc 2")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Camera đã mở.")
print("Nhấn Q để thoát.")
print("Nhấn S để lưu ảnh kết quả.")

# =========================
# VÒNG LẶP CAMERA
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Không đọc được frame từ camera.")
        break

    # =========================
    # YOLO NHẬN DIỆN
    # =========================
    results = model.predict(
        source=frame,
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

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            original_name = model.names[class_id]

            # Đổi tên hiển thị
            class_name = "U Nao"

            detection_count += 1

            # =========================
            # VẼ KHUNG
            # =========================
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            # =========================
            # NHÃN
            # =========================
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
frame,
                (
                    x1,
                    text_y - text_height - 10
                ),
                (
                    x1 + text_width + 10,
                    text_y + 5
                ),
                (0, 0, 255),
                -1
            )

            # Chữ trắng
            cv2.putText(
                frame,
                label,
                (
                    x1 + 5,
                    text_y - 3
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # =========================
    # HIỂN THỊ TRẠNG THÁI
    # =========================
    if detection_count > 0:

        status = (
            f"PHAT HIEN "
            f"{detection_count} "
            f"VUNG U NAO"
        )

        color = (0, 0, 255)

    else:

        status = "KHONG CO DETECTION"
        color = (0, 255, 0)

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    # =========================
    # HIỂN THỊ CAMERA
    # =========================
    cv2.imshow(
        "Nhan dien U Nao - Camera",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Q = thoát
    if key == ord("q"):
        break

    # S = lưu ảnh
    elif key == ord("s"):

        cv2.imwrite(
            "ket_qua_camera.jpg",
            frame
        )

        print("Đã lưu: ket_qua_camera.jpg")

# =========================
# GIẢI PHÓNG CAMERA
# =========================
cap.release()
cv2.destroyAllWindows()

print("Đã đóng camera.")