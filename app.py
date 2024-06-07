import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from camera import VideoCamera, gen, draw_label

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load YOLOv8 models
helmet_model_path = 'D:/deteksi/104.pt'
license_plate_model_path = 'D:/deteksi/best.pt'
helmet_model = YOLO(helmet_model_path)
license_plate_model = YOLO(license_plate_model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/use_webcam')
def use_webcam():
    return render_template('livestream.html')

@app.route('/upload_files', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            filetype = 'image' if file.filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'} else 'video'
            if filetype == 'video':
                detect_objects_in_video(file_path)
            else:
                detect_objects_in_image(file_path)
            return render_template('upload.html', filename=filename, filetype=filetype)
    return render_template('upload.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(index=1)), mimetype='multipart/x-mixed-replace; boundary=frame')

def draw_label(image, text, x, y, color):
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), color, cv2.FILLED)
    cv2.putText(image, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)

def detect_objects_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return

    results = helmet_model(image)
    print(f"Total objects detected by helmet model: {len(results)}")

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = helmet_model.names[int(box.cls[0])]
            print(f"Detected {class_name} at ({x1}, {y1}), ({x2}, {y2})")

            if class_name == 'bike':
                bike_roi = image[y1:y2, x1:x2]
                print(f"Cropping ROI for bike: ({x1}, {y1}), ({x2}, {y2})")

                # Deteksi helm dalam bounding box sepeda motor
                helmet_results = helmet_model(bike_roi)
                no_helmet_detected = False
                for helmet_result in helmet_results:
                    helmet_boxes = helmet_result.boxes
                    for helmet_box in helmet_boxes:
                        hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0])
                        helmet_class_name = helmet_model.names[int(helmet_box.cls[0])]
                        print(f"Detected {helmet_class_name} at ({hx1}, {hy1}), ({hx2}, {hy2}) in bike ROI")

                        if helmet_class_name in ['helmet', 'No-helmet']:
                            color = (0, 255, 0) if helmet_class_name == 'helmet' else (0, 0, 255)
                            cv2.rectangle(bike_roi, (hx1, hy1), (hx2, hy2), color, 2)
                            draw_label(bike_roi, f'{helmet_class_name}', hx1, hy1, color)
                            print(f"Labeled {helmet_class_name}")

                            if helmet_class_name == 'No-helmet':
                                no_helmet_detected = True

                # Hanya lakukan deteksi plat nomor jika ada pelanggaran helm
                if no_helmet_detected:
                    license_results = license_plate_model(bike_roi)
                    print(f"Total objects detected by license plate model: {len(license_results)}")
                    for license_result in license_results:
                        license_boxes = license_result.boxes
                        for license_box in license_boxes:
                            lx1, ly1, lx2, ly2 = map(int, license_box.xyxy[0])
                            l_class_name = license_plate_model.names[int(license_box.cls[0])]
                            if l_class_name == 'License Plate':
                                l_color = (255, 255, 0)
                                print(f"Detected License Plate at ({lx1}, {ly1}), ({lx2}, {ly2})")
                                cv2.rectangle(bike_roi, (lx1, ly1), (lx2, ly2), l_color, 2)
                                draw_label(bike_roi, f'{l_class_name}', lx1, ly1, l_color)

                # Replace the original bike ROI with the updated one
                image[y1:y2, x1:x2] = bike_roi
            else:
                color = (0, 255, 0) if class_name == 'helmet' else (0, 0, 255) if class_name == 'No-helmet' else (255, 0, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                draw_label(image, class_name, x1, y1, color)

    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
    cv2.imwrite(output_image_path, image)
    print(f"Saved output image to {output_image_path}")

def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('static/uploads/output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = helmet_model(frame)
        print(f"Total objects detected by helmet model: {len(results)} in frame {frame_count}")

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = helmet_model.names[int(box.cls[0])]
                print(f"Detected {class_name} at ({x1}, {y1}), ({x2}, {y2}) in frame {frame_count}")

                if class_name == 'bike':
                    bike_roi = frame[y1:y2, x1:x2]
                    print(f"Cropping ROI for bike: ({x1}, {y1}), ({x2}, {y2}) in frame {frame_count}")

                    # Deteksi helm dalam bounding box sepeda motor
                    helmet_results = helmet_model(bike_roi)
                    no_helmet_detected = False
                    for helmet_result in helmet_results:
                        helmet_boxes = helmet_result.boxes
                        for helmet_box in helmet_boxes:
                            hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0])
                            helmet_class_name = helmet_model.names[int(helmet_box.cls[0])]
                            print(f"Detected {helmet_class_name} at ({hx1}, {hy1}), ({hx2}, {hy2}) in bike ROI")

                            if helmet_class_name in ['helmet', 'No-helmet']:
                                color = (0, 255, 0) if helmet_class_name == 'helmet' else (0, 0, 255)
                                cv2.rectangle(bike_roi, (hx1, hy1), (hx2, hy2), color, 2)
                                draw_label(bike_roi, f'{helmet_class_name}', hx1, hy1, color)
                                print(f"Labeled {helmet_class_name}")

                                if helmet_class_name == 'No-helmet':
                                    no_helmet_detected = True

                    # Hanya lakukan deteksi plat nomor jika ada pelanggaran helm
                    if no_helmet_detected:
                        license_results = license_plate_model(bike_roi)
                        print(f"Total objects detected by license plate model: {len(license_results)} in frame {frame_count}")
                        for license_result in license_results:
                            license_boxes = license_result.boxes
                            for license_box in license_boxes:
                                lx1, ly1, lx2, ly2 = map(int, license_box.xyxy[0])
                                l_class_name = license_plate_model.names[int(license_box.cls[0])]
                                if l_class_name == 'License Plate':
                                    l_color = (255, 255, 0)
                                    print(f"Detected License Plate at ({lx1}, {ly1}), ({lx2}, {ly2}) in frame {frame_count}")
                                    cv2.rectangle(bike_roi, (lx1, ly1), (lx2, lx2), l_color, 2)
                                    draw_label(bike_roi, f'{l_class_name}', lx1, ly1, l_color)

                    # Replace the original bike ROI with the updated one
                    frame[y1:y2, x1:x2] = bike_roi
                else:
                    color = (0, 255, 0) if class_name == 'helmet' else (0, 0, 255) if class_name == 'No-helmet' else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label(frame, class_name, x1, y1, color)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Saved video output to static/uploads/output.mp4")

if __name__ == '__main__':
    app.run(debug=True)
