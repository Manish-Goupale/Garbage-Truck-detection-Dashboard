from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pytesseract
from datetime import datetime
import mysql.connector
import time
import os
import threading
from queue import Queue, Empty
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'user_database',
    'pool_size': 5  # Connection pooling
}

# Global variables for producer-consumer pattern
frame_queue = Queue(maxsize=5)  # Limit queue size to prevent memory issues
result_queue = Queue(maxsize=10)
latest_processed_frame = None
latest_plate_info = None
processing_active = True
detected_plates = []  # Store recent plate detections with their coordinates

def initialize_database():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS license_plates (
                id INT AUTO_INCREMENT PRIMARY KEY,
                plate_number VARCHAR(20),
                timestamp DATETIME,
                confidence FLOAT,
                INDEX (timestamp)
            )
        ''')
        conn.commit()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def enhance_image(image):
    """Optimized image enhancement"""
    # Apply quick contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))  # Reduced parameters
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def fast_preprocess(frame):
    """Faster preprocessing with fewer steps"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply quick blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Quick adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Simple morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed

def save_to_database(plate_number, confidence):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Check if the same plate was detected in the last minute to prevent duplicates
        query = """
        SELECT id FROM license_plates 
        WHERE plate_number = %s AND timestamp > DATE_SUB(NOW(), INTERVAL 1 MINUTE)
        LIMIT 1
        """
        cursor.execute(query, (plate_number,))
        if cursor.fetchone():
            logger.info(f"Skipping duplicate plate {plate_number}")
            return
            
        query = """
        INSERT INTO license_plates (plate_number, timestamp, confidence)
        VALUES (%s, %s, %s)
        """
        values = (plate_number, datetime.now(), confidence)
        
        cursor.execute(query, values)
        conn.commit()
        logger.info(f"✅ Saved plate {plate_number} to database (confidence: {confidence:.2f})")
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def detect_plate(frame):
    """Improved plate detection with more visible bounding boxes"""
    global detected_plates
    
    if frame is None:
        return frame, None, 0
        
    original = frame.copy()
    height, width = frame.shape[:2]
    
    processed = fast_preprocess(frame)
    
    # Find contours with less precision
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Limit the number of contours to analyze
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Increased from 8 to 10
    
    plate_text = None
    confidence = 0
    best_score = 0
    detected_plates = []  # Reset plate detections for this frame
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Quick filter on aspect ratio and size
        if 1.5 <= aspect_ratio <= 7.0 and w > 70 and h > 20:  # Relaxed size constraint
            roi_score = (w * h) / (width * height)
            
            if 0.0005 <= roi_score <= 0.07:  # Relaxed area constraint
                # Extract region with minimal padding
                padding = int(h * 0.15)  # Increased padding
                plate_img = original[max(y-padding,0):min(y+h+padding,height), 
                                  max(x-padding,0):min(x+w+padding,width)]
                
                if plate_img.size == 0:
                    continue
                
                # Quick enhancement
                plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                plate_gray = cv2.equalizeHist(plate_gray)
                
                try:
                    # Use optimized OCR config for license plates
                    config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(plate_gray, config=config)
                    text = ''.join(e for e in text if e.isalnum())
                    
                    if len(text) >= 4 and len(text) <= 15:  # Relaxed length constraints
                        current_score = roi_score * (len(text) / 12.0)
                        if current_score > best_score:
                            best_score = current_score
                            plate_text = text
                            confidence = 0.7 + (current_score * 0.3)
                            
                            # Store the detection for this frame
                            detected_plates.append({
                                'text': text,
                                'confidence': confidence,
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h
                            })
                                
                except Exception as e:
                    logger.error(f"OCR Error: {e}")
                    continue
    
    # Draw bounding boxes with much thicker lines and contrasting colors
    for plate in detected_plates:
        # Draw a thicker rectangle
        cv2.rectangle(frame, 
                     (plate['x'], plate['y']), 
                     (plate['x'] + plate['w'], plate['y'] + plate['h']), 
                     (0, 255, 0), 4)  # Increased thickness to 4
        
        # Add contrasting background for text
        text_bg = np.zeros((30, 200, 3), dtype=np.uint8)
        text_bg[:] = (0, 0, 0)  # Black background
        
        # Draw plate text with larger font and better position
        cv2.putText(frame, plate['text'], 
                   (plate['x'], max(plate['y'] - 15, 30)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0,  # Larger font
                   (255, 255, 0), 3)  # Yellow text with increased thickness
    
    return frame, plate_text, confidence

def frame_producer():
    """Thread function to capture frames from camera"""
    global processing_active
    cap = None
    
    try:
        # Try to open RTSP stream directly with optimized parameters
        cap = cv2.VideoCapture("rtsp://192.168.1.7/live/ch00_0")
        
        # Set buffer size to minimum
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set preferred resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set RTSP transport
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        if not cap.isOpened():
            # Fallback options
            sources = [0, 1, 2, 'http://localhost:8080/video']
            for source in sources:
                try:
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"✅ Camera opened successfully: {source}")
                            break
                        else:
                            cap.release()
                except Exception as e:
                    logger.error(f"❌ Failed to open camera source {source}: {e}")
                    if cap:
                        cap.release()
        
        if cap is None or not cap.isOpened():
            logger.error("❌ No camera found")
            processing_active = False
            return
            
        logger.info("✅ Camera initialized successfully")
        
        frame_count = 0
        skip_frames = 1  # Process every 2nd frame (reduced from 3rd)
        
        while processing_active:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                logger.warning("❌ Failed to capture frame")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            if frame_count % (skip_frames + 1) != 0:
                continue  # Skip frame
                
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            
            # Put frame in queue, non-blocking
            try:
                frame_queue.put(frame, block=False)
            except:
                # If queue is full, skip this frame
                pass
                
    except Exception as e:
        logger.error(f"❌ Camera error: {e}")
    finally:
        if cap:
            cap.release()
        processing_active = False

def frame_processor():
    """Thread function to process frames"""
    global processing_active, latest_processed_frame, latest_plate_info
    
    while processing_active:
        try:
            # Get frame from queue with timeout
            frame = frame_queue.get(timeout=1.0)
            
            # Process frame
            processed_frame, plate_text, confidence = detect_plate(frame)
            
            # Add timestamp and processing indicator
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, timestamp, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add "PROCESSING" indicator
            cv2.putText(processed_frame, "PROCESSING", (processed_frame.shape[1]-150, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update latest processed frame
            latest_processed_frame = processed_frame
            
            # Save plate info if confidence is high enough
            if plate_text and confidence > 0.65:  # Reduced threshold slightly
                plate_info = {
                    "plate": plate_text,
                    "confidence": confidence,
                    "timestamp": timestamp
                }
                latest_plate_info = plate_info
                
                # Save to database in a separate thread
                threading.Thread(target=save_to_database, args=(plate_text, confidence)).start()
            
            frame_queue.task_done()
            
        except Empty:
            # Queue is empty, just continue
            pass
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            time.sleep(0.1)

def generate_frames():
    """Generator function for streaming processed frames"""
    global latest_processed_frame, processing_active
    
    while processing_active:
        if latest_processed_frame is not None:
            try:
                # Use higher quality JPEG encoding to ensure bounding boxes are visible
                ret, buffer = cv2.imencode('.jpg', latest_processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"❌ Frame encoding error: {e}")
        
        # Shorter sleep to improve responsiveness
        time.sleep(0.03)

@app.route('/plate_feed')
def plate_feed():
    """Endpoint for streaming the processed video feed"""
    return Response(generate_frames(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recent_plates')
def get_recent_plates():
    """Endpoint to get recently detected license plates"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT plate_number, timestamp, confidence 
            FROM license_plates 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        plates = cursor.fetchall()
        # Convert datetime objects to strings for JSON serialization
        for plate in plates:
            plate['timestamp'] = plate['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            plate['confidence'] = float(plate['confidence'])
            
        return jsonify(plates)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "queue_size": frame_queue.qsize(),
        "processing_active": processing_active,
        "latest_plate": latest_plate_info,
        "detected_plates_count": len(detected_plates)
    })

def start_processing_threads():
    """Start the producer and consumer threads"""
    producer = threading.Thread(target=frame_producer)
    producer.daemon = True
    producer.start()
    
    processor = threading.Thread(target=frame_processor)
    processor.daemon = True
    processor.start()
    
    logger.info("✅ Processing threads started")

if __name__ == '__main__':
    if not os.path.exists('debug_images'):
        os.makedirs('debug_images')
    
    # Set Tesseract path based on OS
    if os.name == 'nt':  # Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:  # Linux/Mac
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    
    initialize_database()
    start_processing_threads()
    
    logger.info("Starting license plate detection server on port 5001...")
    # Run with threaded=True but debug=False to avoid the reloader causing duplicate threads
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)