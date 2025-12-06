"""
Flask web interface for the surveillance system
"""
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import cv2
import threading
import time
import numpy as np
from datetime import datetime
import json
import gc
from werkzeug.utils import secure_filename
from video_processor import ThreadedVideoCapture, FrameProcessor
from captioner import OptimizedCaptioner
from caption_classifier import OptimizedCaptionClassifier
from alert_manager import AlertManager
import config

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALERTS_FOLDER'] = 'alerts'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Ensure upload and alerts folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ALERTS_FOLDER'], exist_ok=True)

# Initialize models and processors
print("Initializing models...")
captioner = OptimizedCaptioner()
classifier = OptimizedCaptionClassifier()
alert_manager = AlertManager()
frame_processor = FrameProcessor(captioner, classifier, alert_manager)
print("Models initialized successfully!")

# Global variables
camera = None
camera_lock = threading.Lock()

def release_camera():
    """Completely release the camera and clean up resources"""
    global camera
    with camera_lock:
        if camera is not None:
            print("Releasing camera...")
            try:
                if hasattr(camera, 'stop'):
                    camera.stop()
                if hasattr(camera, 'release'):
                    camera.release()
                # Add a small delay to ensure resources are freed
                time.sleep(1)
            except Exception as e:
                print(f"Error releasing camera: {e}")
            finally:
                camera = None
                # Force garbage collection
                gc.collect()
                print("Camera resources released")

def get_camera(lock_required=True):
    """Get or create camera instance - simplified like main_preview.py
    Args:
        lock_required: If False, assumes lock is already held by caller
    """
    global camera
    
    # If lock is not required, we assume caller already has it
    if lock_required:
        lock = camera_lock
    else:
        # Use a dummy lock that does nothing
        class DummyLock:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        lock = DummyLock()
    
    with lock:
        try:
            print("\n=== Initializing Camera ===")
            
            # Release existing camera if any
            if camera is not None:
                try:
                    print("Releasing existing camera instance...")
                    if hasattr(camera, 'release'):
                        camera.release()
                    elif hasattr(camera, 'stop'):
                        camera.stop()
                    camera = None
                    time.sleep(0.5)  # Give camera time to release
                except Exception as e:
                    print(f"Error releasing existing camera: {e}")
                    camera = None
                    time.sleep(0.5)

            # Create camera instance - same as main_preview.py
            # ThreadedVideoCapture will handle backend selection internally
            print(f"Starting video capture (source: {config.DEFAULT_CAMERA})...")
            try:
                cam = ThreadedVideoCapture(config.DEFAULT_CAMERA)
            except Exception as e:
                import traceback
                print(f"ERROR: Failed to create ThreadedVideoCapture: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                return None
            
            # Give it a moment to fully initialize
            time.sleep(0.5)
            
            # Check if camera opened - same as main_preview.py
            if not cam.isOpened():
                print("ERROR: Could not open camera!")
                try:
                    if hasattr(cam, 'release'):
                        cam.release()
                except:
                    pass
                return None
            
            # Verify we can actually read a frame
            ret, test_frame = cam.read()
            if not ret or test_frame is None:
                print("WARNING: Camera opened but cannot read frames yet, will retry in stream...")
                # Don't return None, let it try in the stream
            
            print("Camera initialized successfully")
            return cam
                
        except Exception as e:
            import traceback
            print(f"Error in get_camera: {str(e)}\n{traceback.format_exc()}")
            return None

def generate_frames():
    """Generate video frames for streaming with robust error handling"""
    global camera, frame_processor
    
    print("[generate_frames] Starting frame generation...")
    
    # Initialize frame processor if not already done
    if frame_processor is None:
        print("Initializing frame processor...")
        frame_processor = FrameProcessor(
            captioner,
            classifier,
            alert_manager,
            process_every_n=config.PROCESS_EVERY_N_FRAMES
        )
    
    # Performance tracking
    latest_caption = ""
    latest_prediction = None
    frame_count = 0
    last_processed_time = time.time()
    last_frame_time = time.time()
    fps = 0
    frame_times = []
    
    # Error handling
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5
    RECOVERY_ATTEMPTS = 3
    recovery_attempts = 0
    
    # Performance settings
    TARGET_FPS = 30  # Target frames per second
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    last_frame_time = time.time()
    
    # Skip frames to maintain target FPS
    skip_frames = 0
    SKIP_FRAME_AFTER_PROCESS = 1  # Skip N frames after processing
    
    # Pre-allocate frame buffer
    frame_buffer = None
    
    while True:
        try:
            # Get a frame from the camera - simplified check like main_preview.py
            # Use lock to prevent race condition with /start_camera route
            with camera_lock:
                camera_ready = False
                if camera is None:
                    print("[generate_frames] Camera is None, initializing...")
                elif not hasattr(camera, 'isOpened'):
                    print("[generate_frames] Camera object has no isOpened method")
                else:
                    try:
                        camera_ready = camera.isOpened()
                        if not camera_ready:
                            print("[generate_frames] Camera is not opened")
                    except Exception as e:
                        print(f"[generate_frames] Error checking camera state: {e}")
                        camera_ready = False
                
                if not camera_ready:
                    print("[generate_frames] Camera not ready, attempting to initialize...")
                    # Don't acquire lock again since we already have it
                    camera = get_camera(lock_required=False)
                    # If camera was just initialized, give it time to start capturing
                    if camera is not None:
                        print("[generate_frames] Camera initialized, waiting for first frame...")
                        time.sleep(1.0)  # Give ThreadedVideoCapture time to fill the queue
            
            # Check again outside lock
            if camera is None or not hasattr(camera, 'isOpened') or not camera.isOpened():
                # Create a blank frame with error message
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera initialization failed", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
                continue
            
            # Calculate time since last frame and sleep if needed to maintain target FPS
            current_time = time.time()
            time_since_last = current_time - last_frame_time
            
            # Skip this frame if we're ahead of schedule
            if time_since_last < FRAME_INTERVAL and frame_count > 0:
                time.sleep(max(0, FRAME_INTERVAL - time_since_last - 0.001))  # Small buffer
                continue
                
            last_frame_time = time.time()
            
            # Skip frames if we're processing too slowly
            if skip_frames > 0:
                skip_frames -= 1
                # Still read the frame to keep the buffer fresh, but don't process it
                try:
                    ret, _ = camera.read()
                    if not ret:
                        consecutive_errors += 1
                except:
                    consecutive_errors += 1
                continue
                
            # Read frame from camera
            ret, frame = camera.read()
            
            # Handle frame read errors
            if not ret or frame is None:
                consecutive_errors += 1
                if consecutive_errors == 1:  # Only log first time
                    print("Waiting for frames from camera...")
                
                # Try to recover by skipping a few frames
                if consecutive_errors % 10 == 0:  # Every 10 errors
                    print(f"Still waiting for frames... (attempt {consecutive_errors})")
                    
                time.sleep(0.05)  # Small delay to prevent busy-waiting
                continue
                
            # Reset error counter on successful frame read
            consecutive_errors = 0
            
            if not ret or frame is None:
                consecutive_errors += 1
                print(f"Failed to read frame from camera (attempt {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})")
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print("Max consecutive errors reached, attempting to recover...")
                    recovery_attempts += 1
                    
                    if recovery_attempts > RECOVERY_ATTEMPTS:
                        print("Max recovery attempts reached, giving up...")
                        break
                        
                    print(f"Recovery attempt {recovery_attempts}/{RECOVERY_ATTEMPTS}")
                    
                    # Release existing camera
                    with camera_lock:
                        if camera is not None:
                            try:
                                if hasattr(camera, 'release'):
                                    camera.release()
                                elif hasattr(camera, 'stop'):
                                    camera.stop()
                            except Exception as e:
                                print(f"Error releasing camera: {e}")
                            camera = None
                    
                    # Wait before reinitializing
                    time.sleep(1.0)
                    
                    # Reinitialize camera
                    with camera_lock:
                        camera = get_camera(lock_required=False)
                        if camera is not None and camera.isOpened():
                            print("Camera reinitialized successfully")
                            consecutive_errors = 0
                            continue
                        
                    # If we get here, reinitialization failed
                    print("Failed to reinitialize camera, waiting before next attempt...")
                    time.sleep(2.0)
                    continue
                
                # Return an error frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                error_msg = f"Camera error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})"
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    error_msg = "Reinitializing camera..."
                cv2.putText(frame, error_msg, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.3)
                continue
            
            # Reset error counter on successful frame read
            consecutive_errors = 0
            frame_count += 1
            
            # Process the frame (every N frames for performance)
            current_time = time.time()
            should_process = (frame_count % config.PROCESS_EVERY_N_FRAMES == 0)
            
            # Process frame if needed, but don't block
            if should_process and frame_processor is not None:
                try:
                    # Non-blocking frame processing
                    frame_processor.process_frame(frame)
                    result = frame_processor.get_result()
                    
                    if result:
                        latest_caption = result.get('caption', '')
                        latest_prediction = result.get('prediction')
                        confidence = result.get('confidence', 0)
                        
                        # Log detection for debugging
                        if latest_prediction and latest_prediction != 'NORMAL':
                            print(f"ALERT: {latest_prediction} detected! (Confidence: {confidence:.2f})")
                            
                            # Define alert types and their properties
                            alert_types = {
                                'fire': {
                                    'severity': 'CRITICAL',
                                    'color': (0, 0, 255),  # Red
                                    'sound_freq': 2000,    # High pitch
                                    'beep_pattern': [(200, 200), (100, 0), (200, 200), (100, 0), (200, 200)],  # Three beeps with pauses
                                    'prefix': '🚨 FIRE DETECTED! ',
                                    'priority': 1
                                },
                                'gun': {
                                    'severity': 'CRITICAL',
                                    'color': (0, 0, 255),  # Red
                                    'sound_freq': 1800,
                                    'beep_pattern': [(100, 50), (100, 50), (100, 50), (100, 50)],  # Rapid beeps
                                    'prefix': '🔫 WEAPON DETECTED! ',
                                    'priority': 1
                                },
                                'knife': {
                                    'severity': 'HIGH',
                                    'color': (0, 69, 255),  # Orange-red
                                    'sound_freq': 1500,
                                    'beep_pattern': [(300, 100), (300, 100)],  # Two longer beeps
                                    'prefix': '🔪 KNIFE DETECTED! ',
                                    'priority': 2
                                },
                                'robbery': {
                                    'severity': 'HIGH',
                                    'color': (0, 120, 255),  # Orange
                                    'sound_freq': 1200,
                                    'beep_pattern': [(200, 100), (200, 100), (200, 100)],  # Three beeps
                                    'prefix': '👤 SUSPICIOUS ACTIVITY! ',
                                    'priority': 2
                                },
                                'default': {
                                    'severity': 'MEDIUM',
                                    'color': (0, 255, 255),  # Yellow
                                    'sound_freq': 1000,
                                    'beep_pattern': [(200, 0)],  # Single beep
                                    'prefix': '⚠️ ANOMALY DETECTED: ',
                                    'priority': 3
                                }
                            }
                            
                            # Find matching alert type or use default
                            alert_type = 'default'
                            for key in alert_types:
                                if key in latest_prediction.lower():
                                    if alert_type == 'default' or alert_types[key]['priority'] < alert_types[alert_type]['priority']:
                                        alert_type = key
                            
                            alert_config = alert_types[alert_type]
                            
                            # Only save alert for non-NORMAL predictions
                            if latest_prediction != 'NORMAL':
                                alert_data = {
                                    'event': latest_prediction,
                                    'type': alert_type,
                                    'severity': alert_config['severity'],
                                    'confidence': float(confidence),
                                    'caption': latest_caption,  # Include AI-generated caption
                                    'prediction': latest_prediction,  # Add prediction field for filtering
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                save_alert(alert_data)
                            
                            # Play alert sound based on alert type (non-blocking)
                            try:
                                import winsound
                                
                                def play_alert_sound():
                                    try:
                                        for duration, pause in alert_config['beep_pattern']:
                                            winsound.Beep(alert_config['sound_freq'], duration)
                                            if pause > 0:
                                                time.sleep(pause/1000)
                                    except Exception as e:
                                        print(f"Error in alert sound: {e}")
                                
                                alert_thread = threading.Thread(target=play_alert_sound, daemon=True)
                                alert_thread.start()
                                
                            except Exception as e:
                                print(f"Could not play alert sound: {e}")
                            
                            # Set alert text with emoji prefix
                            alert_text = f"{alert_config['prefix']}{latest_prediction.replace('_', ' ').upper()} ({confidence:.0%})"
                except Exception as e:
                    print(f"Error processing frame: {e}")
                
                # Skip next frame to maintain performance
                skip_frames = 1
            
            # Create display frame (always create a new one to avoid reference issues)
            display_frame = frame.copy()
            
            # Draw captions and alerts on the frame
            if latest_prediction and isinstance(latest_prediction, str):
                try:
                    # Get severity and color
                    confidence = result.get('confidence', 0)
                    
                    # Get alert configuration based on detection type
                    alert_type = 'default'
                    for key in ['fire', 'gun', 'knife', 'robbery']:
                        if key in latest_prediction.lower():
                            alert_type = key
                            break
                    
                    # Always show caption if available, even for NORMAL predictions
                    if latest_caption:
                        # Draw caption at the bottom of the frame
                        caption_y = display_frame.shape[0] - 20
                        cv2.putText(display_frame, f"AI: {latest_caption}", 
                                  (10, caption_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"AI: {latest_caption}", 
                                  (12, caption_y+2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # Only show alert box for non-NORMAL predictions
                    if latest_prediction != 'NORMAL':
                        alert_config = {
                        'fire': {'severity': 'CRITICAL', 'color': (0, 0, 255), 'prefix': '🚨 FIRE DETECTED! '},
                        'gun': {'severity': 'CRITICAL', 'color': (0, 0, 255), 'prefix': '🔫 WEAPON DETECTED! '},
                        'knife': {'severity': 'HIGH', 'color': (0, 69, 255), 'prefix': '🔪 KNIFE DETECTED! '},
                        'robbery': {'severity': 'HIGH', 'color': (0, 120, 255), 'prefix': '👤 SUSPICIOUS ACTIVITY! '},
                        'default': {'severity': 'MEDIUM', 'color': (0, 255, 255), 'prefix': '⚠️ ANOMALY DETECTED: '}
                    }[alert_type]
                    
                    # Adjust color based on confidence
                    if confidence < 0.4:
                        alert_config['color'] = (0, 255, 255)  # Yellow for low confidence
                    elif confidence < 0.7 and alert_type == 'default':
                        alert_config['color'] = (0, 165, 255)  # Orange for medium confidence (default only)
                    
                    severity = alert_config['severity']
                    color = alert_config['color']
                    
                    # Set alert text with emoji prefix and confidence
                    alert_text = f"{alert_config['prefix']}{latest_prediction.replace('_', ' ').upper()} ({confidence:.0%})"
                    
                    # Draw alert box with border and shadow
                    (text_width, text_height), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    # Calculate box dimensions with padding
                    padding = 15
                    text_x = 15
                    text_y = 35
                    box_w = text_width + 2 * padding
                    box_h = text_height + 2 * padding
                    
                    # Draw shadow (slightly offset)
                    shadow_offset = 3
                    cv2.rectangle(display_frame, 
                                (5 + shadow_offset, 5 + shadow_offset), 
                                (5 + box_w + shadow_offset, 5 + box_h + shadow_offset), 
                                (0, 0, 0), -1)
                    
                    # Draw main alert box
                    cv2.rectangle(display_frame, 
                                (5, 5), 
                                (5 + box_w, 5 + box_h), 
                                color, -1)  # Filled with alert color
                    
                    # Draw border
                    border_color = tuple(min(c + 100, 255) for c in color)  # Lighter border
                    cv2.rectangle(display_frame, 
                                (5, 5), 
                                (5 + box_w, 5 + box_h), 
                                border_color, 2)
                    
                    # Draw text with shadow for better readability
                    text_color = (255, 255, 255)  # White text
                    
                    # Draw text shadow (slightly offset)
                    cv2.putText(display_frame, alert_text, 
                              (text_x + 1, text_y + 1), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Draw main text
                    cv2.putText(display_frame, alert_text, 
                              (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    
                    # Log to console for debugging
                    print(f"ALERT DISPLAYED: {alert_text}")
                    
                except Exception as e:
                    print(f"Error drawing overlay: {e}")
            
            # Calculate and display FPS (less frequently to reduce overhead)
            frame_times.append(current_time)
            frame_times = [t for t in frame_times if current_time - t < 2.0]  # Keep last 2 seconds
            
            if len(frame_times) > 1:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                
                # Only update FPS display every 10 frames to reduce overhead
                if frame_count % 10 == 0:
                    # Draw FPS in top-right corner with background for better visibility
                    fps_text = f"FPS: {fps:.1f}"
                    (fps_width, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw background
                    cv2.rectangle(display_frame, 
                                (display_frame.shape[1] - fps_width - 20, 10),
                                (display_frame.shape[1] - 10, 40),
                                (0, 0, 0), -1)
                    
                    # Draw FPS text
                    cv2.putText(display_frame, fps_text,
                               (display_frame.shape[1] - fps_width - 10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update frame counter
            frame_count += 1
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            import traceback
            traceback.print_exc()
            
            # Return an error frame
            try:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Error: {str(e)}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except:
                pass
                
            time.sleep(0.5)

# Update the process_frame function
def process_frame(frame):
    """Process a single frame"""
    try:
        # Check if frame is valid
        if frame is None:
            print("Error: Received None frame")
            return None

        # Generate caption
        caption = captioner.generate_caption(frame)
        
        # Classify the caption
        if hasattr(classifier, 'classify'):
            prediction = classifier.classify(caption)
        else:
            print("Error: No suitable prediction method found in classifier")
            return None
        
        # Determine severity
        severity = "NORMAL"
        if prediction['label'] != 'NORMAL':
            if prediction['confidence'] > 0.7:
                severity = "HIGH"
            elif prediction['confidence'] > 0.4:
                severity = "MEDIUM"
            else:
                severity = "LOW"
        
        # Prepare result
        result = {
            'caption': caption,
            'prediction': prediction['label'],
            'confidence': float(prediction['confidence']),  # Convert to float for JSON serialization
            'severity': severity,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save alert if not normal
        if prediction['label'] != 'NORMAL':
            save_alert(result)
        
        return result
        
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return None
def draw_results(frame, result):
    """Draw detection results on frame"""
    # Add your drawing logic here
    # Example: draw bounding boxes, labels, etc.
    return frame

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    print("[video_feed] Video feed route accessed")
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['GET', 'POST'])
def start_camera():
    """Start camera feed with improved error handling"""
    global camera
    
    try:
        with camera_lock:
            # Check if camera is already working
            if camera is not None:
                try:
                    if hasattr(camera, 'isOpened') and camera.isOpened():
                        # Test if we can read a frame
                        ret, test_frame = camera.read()
                        if ret and test_frame is not None:
                            print("Camera is already initialized and working")
                            return jsonify({
                                "status": "success", 
                                "message": "Camera is already running"
                            })
                        else:
                            print("Camera exists but cannot read frames, reinitializing...")
                    else:
                        print("Camera exists but is not opened, reinitializing...")
                except Exception as e:
                    print(f"Error checking existing camera: {e}, reinitializing...")
            
            # Release existing camera if any (only if it's not working)
            if camera is not None:
                try:
                    print("Releasing existing camera...")
                    if hasattr(camera, 'release'):
                        camera.release()
                    elif hasattr(camera, 'stop'):
                        camera.stop()
                    camera = None
                    time.sleep(0.5)
                    print("Existing camera released.")
                except Exception as e:
                    print(f"Error releasing camera: {e}")
                    camera = None
                    time.sleep(0.5)
            
            # Initialize new camera
            print("Starting video capture...")
            camera = get_camera(lock_required=False)  # Lock already held
            
            if camera is None:
                print("ERROR: get_camera() returned None")
                return jsonify({
                    "status": "error",
                    "message": "Failed to initialize camera. Please check if it's connected and not in use by another application."
                }), 500
            
            if not hasattr(camera, 'isOpened'):
                print("ERROR: Camera object has no isOpened method")
                return jsonify({
                    "status": "error",
                    "message": "Invalid camera object."
                }), 500
            
            try:
                is_opened = camera.isOpened()
                if not is_opened:
                    print("ERROR: Camera is not opened")
                    return jsonify({
                        "status": "error",
                        "message": "Camera initialized but is not opened. Please check if camera is available and not in use by another application."
                    }), 500
            except Exception as e:
                print(f"ERROR: Exception checking camera.isOpened(): {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Error checking camera status: {str(e)}"
                }), 500
            
            # Test frame read (but don't fail if it's not ready yet - let generate_frames handle it)
            try:
                ret, test_frame = camera.read()
                if not ret or test_frame is None:
                    print("WARNING: Camera opened but cannot read frame yet, will retry in stream...")
                    # Don't fail here, let generate_frames handle it
            except Exception as e:
                print(f"WARNING: Exception testing frame read: {e}, will retry in stream...")
                # Don't fail here, let generate_frames handle it
        
        print("Camera started successfully")
        return jsonify({
            "status": "success",
            "message": "Camera started successfully"
        })
        
    except Exception as e:
        import traceback
        print(f"Error in start_camera: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": f"Failed to start camera: {str(e)}"
        }), 500
@app.route('/test_camera', methods=['GET', 'POST'])
def test_camera():
    """Test camera access directly"""
    try:
        camera_index = config.DEFAULT_CAMERA
        print(f"\n=== Testing Camera {camera_index} ===")
        
        # Try different backends in order of preference for Windows
        backends_to_try = [
            (cv2.CAP_MSMF, "MSMF (Media Foundation)"),
            (None, "Default (Auto-detect)"),
            (cv2.CAP_DSHOW, "DSHOW (DirectShow)")
        ]
        
        results = []
        
        for backend, backend_name in backends_to_try:
            print(f"Test: Trying {backend_name} backend...")
            cap = None
            try:
                if backend is not None:
                    cap = cv2.VideoCapture(camera_index, backend)
                else:
                    cap = cv2.VideoCapture(camera_index)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    result = {
                        "backend": backend_name,
                        "opened": True,
                        "frame_read": ret and frame is not None,
                        "frame_shape": list(frame.shape) if ret and frame is not None else None,
                        "error": None
                    }
                    results.append(result)
                    print(f"  {backend_name}: Success")
                    
                    # If this backend works, no need to try others
                    if result["frame_read"]:
                        break
                else:
                    results.append({
                        "backend": backend_name,
                        "opened": False,
                        "frame_read": False,
                        "frame_shape": None,
                        "error": "Camera opened but isOpened() returned False"
                    })
                    print(f"  {backend_name}: Failed to open")
                    
            except Exception as e:
                error_msg = str(e)
                results.append({
                    "backend": backend_name,
                    "opened": False,
                    "frame_read": False,
                    "frame_shape": None,
                    "error": error_msg
                })
                print(f"  {backend_name}: Error - {error_msg}")
                
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                time.sleep(0.5)  # Small delay between tests
        
        # Check if any backend worked
        working_backends = [r for r in results if r.get('frame_read')]
        
        if working_backends:
            best_backend = working_backends[0]
            return jsonify({
                "status": "success",
                "message": f"Camera {camera_index} is accessible via {best_backend['backend']}",
                "best_backend": best_backend,
                "all_results": results
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Camera {camera_index} is not accessible with any backend",
                "all_results": results
            }), 500
            
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": f"Error testing camera: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/camera_status')
def camera_status():
    """Check camera status and return detailed information"""
    status = {
        "camera_initialized": camera is not None,
        "camera_opened": False,
        "backend": None,
        "frame_size": None,
        "fps": None,
        "error": None
    }
    
    if camera is not None:
        try:
            status["camera_opened"] = camera.isOpened()
            
            if hasattr(camera, 'backend'):
                status["backend"] = str(camera.backend)
                
            if status["camera_opened"]:
                ret, frame = camera.read()
                if ret and frame is not None:
                    status["frame_size"] = {
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "channels": frame.shape[2] if len(frame.shape) > 2 else 1
                    }
                    
                if hasattr(camera, 'get_fps'):
                    status["fps"] = camera.get_fps()
        
        except Exception as e:
            status["error"] = str(e)
    
    return jsonify(status)

@app.route('/stop_camera', methods=['GET', 'POST'])
def stop_camera():
    """Stop camera feed and release all resources"""
    global camera
    try:
        release_camera()
        return jsonify({
            'status': 'success',
            'message': 'Camera stopped and resources released successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error stopping camera: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload (image or video)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check if it's a video
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                return process_video(filepath, filename)
            else:
                return process_image(filepath, filename)
                
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Unknown error'}), 500

def process_image(filepath, filename):
    """Process a single image"""
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    # Process the image
    result = process_frame(image)
    result['image_url'] = f"/static/uploads/{filename}"
    return jsonify(result)

def process_video(filepath, filename):
    """Process video file"""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({'error': 'Could not open video'}), 400
        
    # Process first frame
    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'Could not read video frame'}), 400
        
    result = process_frame(frame)
    result['video_url'] = f"/static/uploads/{filename}"
    return jsonify(result)



def save_alert(alert_data):
    """Save alert to the alerts log, but only for non-NORMAL predictions"""
    # Only save if it's not a NORMAL prediction
    if alert_data.get('prediction', '').upper() == 'NORMAL':
        return
        
    alerts_log = os.path.join(app.config['ALERTS_FOLDER'], 'alerts_log.json')
    try:
        if os.path.exists(alerts_log):
            with open(alerts_log, 'r') as f:
                try:
                    alerts = json.load(f)
                    # Keep only the last 100 alerts to prevent the file from growing too large
                    alerts = alerts[-99:]
                except json.JSONDecodeError:
                    alerts = []
        else:
            alerts = []
        
        # Add timestamp if not present
        if 'timestamp' not in alert_data:
            alert_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Only add if not already in the list (avoid duplicates)
        if not any(a.get('event') == alert_data.get('event') and 
                  abs((datetime.strptime(a.get('timestamp'), '%Y-%m-%d %H:%M:%S') - 
                       datetime.strptime(alert_data['timestamp'], '%Y-%m-%d %H:%M:%S')).total_seconds()) < 5 
                  for a in alerts):
            alerts.append(alert_data)
            print(f"ALERT SAVED: {alert_data.get('event')} at {alert_data['timestamp']}")
        
        with open(alerts_log, 'w') as f:
            json.dump(alerts, f, indent=2)
            
    except Exception as e:
        print(f"Error saving alert: {e}")
        import traceback
        traceback.print_exc()

@app.route('/recent_alerts')
def get_recent_alerts():
    """Get recent alerts (last 10)"""
    try:
        alerts = get_alerts()
        # Return the 10 most recent alerts
        return jsonify(alerts[-10:][::-1])  # Return in reverse chronological order
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_alerts():
    """Get all alerts from the log file, filtered for suspicious activities"""
    alerts_log = os.path.join(app.config['ALERTS_FOLDER'], 'alerts_log.json')
    suspicious_keywords = ['fire', 'gun', 'knife', 'robbery', 'weapon', 'fight', 'intruder', 'suspicious']
    
    try:
        if os.path.exists(alerts_log):
            with open(alerts_log, 'r') as f:
                try:
                    all_alerts = json.load(f)
                    # Filter for suspicious activities
                    filtered_alerts = [
                        alert for alert in all_alerts 
                        if any(keyword in alert.get('event', '').lower() 
                             for keyword in suspicious_keywords)
                    ]
                    return filtered_alerts
                except json.JSONDecodeError:
                    print("Error: alerts_log.json is corrupted, starting fresh")
                    return []
        return []
    except Exception as e:
        print(f"Error reading alerts: {e}")
        return []
        return []

@app.route('/alerts')
def get_all_alerts():
    """API endpoint to get all alerts"""
    try:
        alerts = get_alerts()
        return jsonify(alerts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

def cleanup():
    """Clean up resources"""
    global camera, frame_processor
    
    print("\nCleaning up resources...")
    
    # Stop camera if it's running
    if camera is not None:
        print("Stopping camera...")
        try:
            if hasattr(camera, 'stop') and callable(camera.stop):
                camera.stop()
            if hasattr(camera, 'release') and callable(camera.release):
                camera.release()
            print("Camera resources released.")
        except Exception as e:
            print(f"Error cleaning up camera: {e}")
    
    # Stop frame processor
    if frame_processor is not None:
        print("Stopping frame processor...")
        try:
            if hasattr(frame_processor, 'stop') and callable(frame_processor.stop):
                frame_processor.stop()
            print("Frame processor stopped.")
        except Exception as e:
            print(f"Error stopping frame processor: {e}")
    
    print("Cleanup complete.")

if __name__ == '__main__':
    import atexit
    
    # Register cleanup function
    atexit.register(cleanup)
    
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('alerts', exist_ok=True)
    
    # Run the app with better error handling
    try:
        print("Starting application...")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        cleanup()
