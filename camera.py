import cv2
from ultralytics import YOLO

# Load YOLOv8 models
helmet_model_path = 'D:/deteksi/104.pt'
license_plate_model_path = 'D:/deteksi/best.pt'
helmet_model = YOLO(helmet_model_path)
license_plate_model = YOLO(license_plate_model_path)

class VideoCamera(object):
    def __init__(self, index=1):
        self.video = cv2.VideoCapture(index)
        if not self.video.isOpened():
            raise Exception(f"Cannot open camera with index {index}")
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None
        
        results = helmet_model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = helmet_model.names[int(box.cls[0])]
                
                if class_name == 'bike':
                    bike_roi = frame[y1:y2, x1:x2]

                    helmet_results = helmet_model(bike_roi)
                    no_helmet_detected = False
                    for helmet_result in helmet_results:
                        helmet_boxes = helmet_result.boxes
                        for helmet_box in helmet_boxes:
                            hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0])
                            helmet_class_name = helmet_model.names[int(helmet_box.cls[0])]

                            if helmet_class_name in ['helmet', 'No-helmet']:
                                color = (0, 255, 0) if helmet_class_name == 'helmet' else (0, 0, 255)
                                cv2.rectangle(bike_roi, (hx1, hy1), (hx2, hy2), color, 2)
                                draw_label(bike_roi, f'{helmet_class_name}', hx1, hy1, color)

                                if helmet_class_name == 'No-helmet':
                                    no_helmet_detected = True

                    if no_helmet_detected:
                        license_results = license_plate_model(bike_roi)
                        for license_result in license_results:
                            license_boxes = license_result.boxes
                            for license_box in license_boxes:
                                lx1, ly1, lx2, ly2 = map(int, license_box.xyxy[0])
                                l_class_name = license_plate_model.names[int(license_box.cls[0])]
                                if l_class_name == 'License Plate':
                                    l_color = (255, 255, 0)
                                    cv2.rectangle(bike_roi, (lx1, ly1), (lx2, ly2), l_color, 2)
                                    draw_label(bike_roi, f'{l_class_name}', lx1, ly1, l_color)

                    frame[y1:y2, x1:x2] = bike_roi
                else:
                    color = (0, 255, 0) if class_name == 'helmet' else (0, 0, 255) if class_name == 'No-helmet' else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label(frame, class_name, x1, y1, color)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def draw_label(image, text, x, y, color):
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), color, cv2.FILLED)
    cv2.putText(image, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
