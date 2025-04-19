from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO
from collections import defaultdict
import time
import threading
import queue
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
passed_count = 0
rejected_count = 0
vehicle_count = 0
tracked_objects = defaultdict(lambda: {'crossed': False, 'center': None, 'direction': None})
VEHICLE_CLASSES = ['truck']

# Frame buffer for processing
frame_queue = queue.Queue(maxsize=10)
processed_frame_queue = queue.Queue(maxsize=10)

# Configure camera settings
RTSP_URL = "rtsp://192.168.1.7/live/ch00_0"
RTSP_OPTIONS = {
    cv2.CAP_PROP_BUFFERSIZE: 2,  # Reduce buffer size
    cv2.CAP_PROP_FRAME_WIDTH: 640,
    cv2.CAP_PROP_FRAME_HEIGHT: 480,
    cv2.CAP_PROP_FPS: 15  # Limit FPS to reduce processing load
}

def get_box_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def camera_reader():
    """Thread function to continuously read frames from camera"""
    # Use a different approach for RTSP transport mode
    cap = cv2.VideoCapture(RTSP_URL + "?tcp", cv2.CAP_FFMPEG)  # Add tcp flag to URL
    
    if not cap.isOpened():
        print("❌ Camera not found")
        return
    
    print("✅ Camera opened successfully")
    
    # Apply camera settings
    for prop, value in RTSP_OPTIONS.items():
        cap.set(prop, value)
    
    # Set H264 codec if available
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    except Exception as e:
        print(f"Note: H264 codec setting failed: {e}")
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read frame, reconnecting...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL + "?tcp", cv2.CAP_FFMPEG)
            for prop, value in RTSP_OPTIONS.items():
                cap.set(prop, value)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            except:
                pass
            continue
        
        frame_count += 1
        # Process every 2nd frame to reduce load
        if frame_count % 2 == 0:
            try:
                # Resize early to reduce processing load
                frame = cv2.resize(frame, (400, 300))
                # Add frame to queue, replace oldest if full
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full

def process_frames():
    """Thread function to process frames with YOLO"""
    global vehicle_count, passed_count, rejected_count, tracked_objects
    
    # Load model only once
    model = YOLO('yolov8n.pt')
    print("✅ Model loaded")
    
    # Use a lower resolution model for faster processing
    line_x = 200  # Define line position
    
    # Track last emission time to avoid flooding socket
    last_emit_time = 0
    emit_interval = 0.2  # seconds
    
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 0), 2)
            
            # Run detection with confidence threshold to filter weak detections
            # Check if CUDA is available and recognized by our OpenCV version
            use_gpu = False
            try:
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    use_gpu = True
            except:
                pass
            
            results = model.track(
                source=frame,
                conf=0.4,  # Increased confidence threshold
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False,  # Disable verbose output to speed up processing
                device='cuda' if use_gpu else 'cpu'  # Use GPU if available
            )
            
            changes_made = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls].lower()
                    
                    if class_name in VEHICLE_CLASSES:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        track_id = int(box.id[0]) if box.id is not None else None
                        
                        if track_id is not None:
                            center = get_box_center(x1, y1, x2, y2)
                            
                            if track_id not in tracked_objects:
                                tracked_objects[track_id] = {'crossed': False, 'center': center, 'direction': None}
                                vehicle_count += 1
                                changes_made = True
                            
                            # Determine direction
                            prev_center = tracked_objects[track_id]['center']
                            
                            # Calculate meaningful direction change (only if moved at least 5 pixels)
                            if abs(center[0] - prev_center[0]) > 5:
                                direction = 'forward' if center[0] > prev_center[0] else 'backward'
                                tracked_objects[track_id]['direction'] = direction
                            else:
                                direction = tracked_objects[track_id]['direction']  # Keep previous direction
                            
                            tracked_objects[track_id]['center'] = center  # Update center
                            
                            if direction and not tracked_objects[track_id]['crossed']:
                                if center[0] > line_x and direction == 'forward':
                                    tracked_objects[track_id]['crossed'] = True
                                    passed_count += 1
                                    changes_made = True
                                
                                elif center[0] < line_x and direction == 'backward':
                                    tracked_objects[track_id]['crossed'] = True
                                    rejected_count += 1
                                    changes_made = True
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name}: {conf:.2f}", 
                                      (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (0, 255, 0), 2)
            
            # Send updates only when there are changes to avoid network congestion
            current_time = time.time()
            if changes_made and current_time - last_emit_time > emit_interval:
                socketio.emit('count_update', {
                    'passed': passed_count,
                    'rejected': rejected_count,
                    'total': vehicle_count
                })
                last_emit_time = current_time
            
            cv2.putText(frame, f"Count: {vehicle_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Put processed frame in output queue
            if processed_frame_queue.full():
                try:
                    processed_frame_queue.get_nowait()
                except queue.Empty:
                    pass
            processed_frame_queue.put(frame, block=False)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Processing error: {e}")
            continue

def generate_frames():
    """Generator function for the video feed"""
    import numpy as np
    
    while True:
        try:
            frame = processed_frame_queue.get(timeout=1)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Lower quality for speed
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            # Return a blank frame if queue is empty to keep connection alive
            blank_frame = np.ones((300, 400, 3), dtype=np.uint8) * 255  # Create white image
            cv2.putText(blank_frame, "Connecting...", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            time.sleep(0.5)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return jsonify({
        "passed": passed_count,
        "rejected": rejected_count,
        "total": vehicle_count
    })

@app.route('/update_passed')
def update_passed():
    global passed_count
    passed_count += 1
    socketio.emit('count_update', {
        'passed': passed_count,
        'rejected': rejected_count,
        'total': vehicle_count
    })
    return jsonify({"status": "success"})

@app.route('/update_rejected')
def update_rejected():
    global rejected_count
    rejected_count += 1
    socketio.emit('count_update', {
        'passed': passed_count,
        'rejected': rejected_count,
        'total': vehicle_count
    })
    return jsonify({"status": "success"})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send current counts immediately upon connection
    emit('count_update', {
        'passed': passed_count,
        'rejected': rejected_count,
        'total': vehicle_count
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    import numpy as np
    
    # Start camera reader thread
    camera_thread = threading.Thread(target=camera_reader, daemon=True)
    camera_thread.start()
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
    
    print("Starting server on port 5000...")
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)