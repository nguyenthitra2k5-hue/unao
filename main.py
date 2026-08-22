from ultralytics import YOLO
import cv2
import os

# =========================
# CẤU HÌNH
# =========================
MODEL_PATH = "best.pt"
VIDEO_PATH = "Phẫu thuật thành công khối u não kích thước lớn.mp4"

CONFIDENCE = 0.7
IMG_SIZE = 640

OUTPUT_PATH = "ket_qua_video_u_nao.mp4"

# =========================
# KIỂM TRA FILE
# =========================
if not os.path.exists(MODEL_PATH):
    print(f"Không tìm thấy model: {MODEL_PATH}")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"Không tìm thấy video: {VIDEO_PATH}")
    exit()

# =========================
# LOAD MODEL
# =========================
print("Đang tải model...")

model = YOLO(MODEL_PATH)

print("Các nhãn:")
print(model.names)

# =========================
# MỞ VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Không mở được video.")
    exit()

# Lấy thông tin video
fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("FPS:", fps)
print("Kích thước:", width, "x", height)
print("Tổng frame:", total_frames)

# =========================
# TẠO VIDEO OUTPUT
# =========================
fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0

# =========================
# XỬ LÝ VIDEO
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

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
                (x1, text_y - text_height - 10),
                (x1 + text_width + 10, text_y + 5),
                (0, 0, 255),
                -1
            )

            # Chữ trắng
            cv2.putText(
                frame,
                label,
                (x1 + 5, text_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # =========================
    # TRẠNG THÁI
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
    # GHI VIDEO
    # =========================
    writer.write(frame)

    # =========================
    # HIỂN THỊ
    # =========================
    cv2.imshow(
        "Nhan dien U Nao - Video",
        frame
    )

    print(
        f"\rĐang xử lý frame "
        f"{frame_count}/{total_frames}",
        end=""
    )

    # Nhấn Q để dừng
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# GIẢI PHÓNG
# =========================
cap.release()
writer.release()
cv2.destroyAllWindows()

print()
print("Hoàn thành.")
print("Video kết quả:", OUTPUT_PATH)