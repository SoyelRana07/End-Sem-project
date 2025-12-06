"""
Multi-threaded video capture and processing
"""
import cv2
import threading
import queue
import time
from collections import deque
from datetime import datetime
import numpy as np

def open_camera_with_timeout(src, backend=None, timeout=5.0):
    """Open camera with timeout to prevent hanging"""
    result = [None]
    exception = [None]
    
    def _open():
        try:
            if backend is not None:
                cap = cv2.VideoCapture(src, backend)
            else:
                cap = cv2.VideoCapture(src)
            result[0] = cap
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=_open, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return None, "Timeout opening camera (operation took longer than 5 seconds)"
    
    if exception[0] is not None:
        return None, str(exception[0])
    
    return result[0], None

class ThreadedVideoCapture:
    """
    Threaded video capture for better performance
    """
    def __init__(self, src=0):
        """Initialize video capture in separate thread with better error handling"""
        self.cap = None
        self.src = src
        
        # Initialize frame queue and other properties first
        self.q = queue.Queue(maxsize=5)  # Increased queue size to prevent empty reads
        self.stopped = False
        self.lock = threading.Lock()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize camera - same approach as main_preview.py
        try:
            print(f"[ThreadedVideoCapture.__init__] Initializing camera {src}...")
            self.initialize_camera()
            # Verify camera is actually opened after initialization
            if self.cap is not None:
                if self.cap.isOpened():
                    print(f"[ThreadedVideoCapture.__init__] Camera opened successfully, testing read...")
                    # Try a test read to ensure it's working
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"[ThreadedVideoCapture.__init__] Camera test read successful")
                    else:
                        print(f"[ThreadedVideoCapture.__init__] Warning: Camera opened but test read failed (ret={ret})")
                else:
                    print(f"[ThreadedVideoCapture.__init__] WARNING: Camera object created but isOpened() returns False")
            else:
                print(f"[ThreadedVideoCapture.__init__] WARNING: Camera initialization completed but self.cap is None")
        except Exception as e:
            import traceback
            print(f"[ThreadedVideoCapture.__init__] ERROR: Failed to initialize camera: {e}")
            print(f"[ThreadedVideoCapture.__init__] Traceback: {traceback.format_exc()}")
            # Camera will be None, but we'll still start the thread to handle reinitialization
        
        # Start capture thread
        self.thread = threading.Thread(target=self._reader, daemon=True, name="CameraReader")
        self.thread.start()
        print(f"[ThreadedVideoCapture] Background thread started: {self.thread.name}, alive: {self.thread.is_alive()}")
        
        # Give the thread a moment to start and potentially get first frame
        # Wait longer to ensure queue has frames
        for wait_iteration in range(10):  # Wait up to 2 seconds (10 * 0.2s)
            time.sleep(0.2)
            queue_size = self.q.qsize()
            if queue_size > 0:
                print(f"[ThreadedVideoCapture] Queue has {queue_size} frame(s) after {wait_iteration * 0.2:.1f}s")
                break
        else:
            # Check if thread is still alive
            if not self.thread.is_alive():
                print(f"[ThreadedVideoCapture] WARNING: Background thread died!")
            else:
                queue_size = self.q.qsize()
                print(f"[ThreadedVideoCapture] WARNING: Queue still empty after 2s, size: {queue_size}")
        
    def initialize_camera(self):
        """Initialize camera with retry mechanism and timeout"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Release existing capture if any
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                
                print(f"Attempting to open camera {self.src} (attempt {attempt + 1}/{max_attempts})...")
                
                # Try different backends - MSMF is better for Windows 10+, then default, then DSHOW
                cap_opened = False
                backends_to_try = [
                    (cv2.CAP_MSMF, "MSMF (Media Foundation)"),
                    (None, "Default (Auto-detect)"),
                    (cv2.CAP_DSHOW, "DSHOW (DirectShow)")
                ]
                
                for backend, backend_name in backends_to_try:
                    if cap_opened:
                        break
                    
                    print(f"  Trying {backend_name} backend...")
                    try:
                        if backend is not None:
                            self.cap = cv2.VideoCapture(self.src, backend)
                        else:
                            self.cap = cv2.VideoCapture(self.src)
                        
                        if self.cap.isOpened():
                            # Try to read a frame to verify it actually works
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                cap_opened = True
                                print(f"  {backend_name} backend successful")
                            else:
                                print(f"  {backend_name} opened but cannot read frames")
                                self.cap.release()
                                self.cap = None
                        else:
                            if self.cap is not None:
                                self.cap.release()
                            self.cap = None
                    except Exception as e:
                        print(f"  {backend_name} backend exception: {e}")
                        if self.cap is not None:
                            try:
                                self.cap.release()
                            except:
                                pass
                            self.cap = None
                
                if not cap_opened or self.cap is None:
                    raise RuntimeError(f"Failed to open camera {self.src} with any backend")
                
                # Set buffer size to 1 to minimize latency
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except:
                    pass
                
                # Set resolution
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                except:
                    pass
                
                # Set FPS if possible
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                except:
                    pass
                
                # Try to set auto focus (works for some cameras)
                try:
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                except:
                    pass
                
                # Check if camera opened successfully
                if not self.cap.isOpened():
                    raise RuntimeError(f"Camera {self.src} opened but isOpened() returned False")
                
                print(f"  Camera {self.src} opened successfully, testing frame read...")
                
                # Test read - try a few times to ensure camera is ready
                ret = False
                test_frame = None
                for test_attempt in range(5):
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"  Successfully read test frame (attempt {test_attempt + 1})")
                        break
                    time.sleep(0.2)
                
                if not ret or test_frame is None:
                    raise RuntimeError(f"Failed to read from camera {self.src} after 5 attempts")
                
                # Give camera a moment to stabilize
                time.sleep(0.2)
                
                print(f"Successfully initialized camera {self.src}")
                return
                
            except Exception as e:
                import traceback
                print(f"Camera initialization attempt {attempt + 1} failed: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                
                # Clean up on failure
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                
                if attempt == max_attempts - 1:
                    error_msg = f"Failed to initialize camera after {max_attempts} attempts: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    raise RuntimeError(error_msg)
                
                # Wait before retrying with increasing delay
                wait_time = 1.0 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    def _reader(self):
        """Read frames in background thread with better error handling"""
        print(f"[_reader] Thread started, camera cap is None: {self.cap is None}")
        consecutive_errors = 0
        max_errors = 5
        frames_read = 0
        
        while not self.stopped:
            try:
                # Check if camera is opened
                if self.cap is None:
                    print("[_reader] self.cap is None, waiting...")
                    time.sleep(1)
                    continue
                    
                if not self.cap.isOpened():
                    print("Camera not open, attempting to reinitialize...")
                    try:
                        # Reinitialize without starting a new thread
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            try:
                                # Release existing capture if any
                                if self.cap is not None:
                                    try:
                                        self.cap.release()
                                    except:
                                        pass
                                
                                # Try to open the camera
                                print(f"  Reinitializing camera {self.src} (attempt {attempt + 1})...")
                                
                                # Try different backends in order
                                cap_opened = False
                                backends_to_try = [
                                    (cv2.CAP_MSMF, "MSMF"),
                                    (None, "Default"),
                                    (cv2.CAP_DSHOW, "DSHOW")
                                ]
                                
                                for backend, backend_name in backends_to_try:
                                    if cap_opened:
                                        break
                                    try:
                                        if backend is not None:
                                            self.cap = cv2.VideoCapture(self.src, backend)
                                        else:
                                            self.cap = cv2.VideoCapture(self.src)
                                        
                                        if self.cap.isOpened():
                                            ret, test_frame = self.cap.read()
                                            if ret and test_frame is not None:
                                                cap_opened = True
                                                break
                                            else:
                                                self.cap.release()
                                                self.cap = None
                                        else:
                                            if self.cap is not None:
                                                self.cap.release()
                                            self.cap = None
                                    except:
                                        if self.cap is not None:
                                            try:
                                                self.cap.release()
                                            except:
                                                pass
                                            self.cap = None
                                
                                # Set properties with error handling
                                if self.cap is not None:
                                    try:
                                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                                    except:
                                        pass
                                
                                # Check if camera opened successfully
                                if not cap_opened or self.cap is None:
                                    raise RuntimeError(f"Failed to open camera {self.src} with any backend")
                                
                                print(f"Successfully reinitialized camera {self.src}")
                                break
                                
                            except Exception as e:
                                if attempt == max_attempts - 1:
                                    print(f"Failed to reinitialize camera after {max_attempts} attempts: {e}")
                                    time.sleep(1)
                                    continue
                                time.sleep(0.5)
                        
                        if self.cap is None or not self.cap.isOpened():
                            print("Failed to reinitialize camera")
                            time.sleep(1)
                            continue
                    except Exception as e:
                        print(f"Error reinitializing camera: {e}")
                        time.sleep(1)
                        continue

                # Read frame with lock protection
                with self.lock:
                    ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors <= 5:  # Log first few errors
                        print(f"[_reader] Warning: Failed to read frame from camera (attempt {consecutive_errors}/{max_errors}), ret={ret}, frame is None: {frame is None}")
                    elif consecutive_errors == max_errors:
                        print(f"[_reader] Max errors reached, will attempt reinitialization")
                    
                    if consecutive_errors >= max_errors:
                        print("Max consecutive errors reached, attempting to reinitialize camera...")
                        try:
                            if self.cap is not None:
                                try:
                                    self.cap.release()
                                except:
                                    pass
                            
                            # Reinitialize without starting a new thread
                            max_attempts = 3
                            for attempt in range(max_attempts):
                                try:
                                    # Try different backends in order
                                    cap_opened = False
                                    backends_to_try = [
                                        (cv2.CAP_MSMF, "MSMF"),
                                        (None, "Default"),
                                        (cv2.CAP_DSHOW, "DSHOW")
                                    ]
                                    
                                    for backend, backend_name in backends_to_try:
                                        if cap_opened:
                                            break
                                        try:
                                            if backend is not None:
                                                self.cap = cv2.VideoCapture(self.src, backend)
                                            else:
                                                self.cap = cv2.VideoCapture(self.src)
                                            
                                            if self.cap.isOpened():
                                                ret, test_frame = self.cap.read()
                                                if ret and test_frame is not None:
                                                    cap_opened = True
                                                    # Set properties
                                                    try:
                                                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                                                    except:
                                                        pass
                                                    break
                                                else:
                                                    self.cap.release()
                                                    self.cap = None
                                            else:
                                                if self.cap is not None:
                                                    self.cap.release()
                                                self.cap = None
                                        except:
                                            if self.cap is not None:
                                                try:
                                                    self.cap.release()
                                                except:
                                                    pass
                                                self.cap = None
                                    
                                    if not cap_opened or self.cap is None:
                                        raise RuntimeError(f"Failed to open camera {self.src} with any backend")
                                    
                                    print(f"Successfully reinitialized camera {self.src}")
                                    consecutive_errors = 0
                                    break
                                except Exception as e:
                                    if attempt == max_attempts - 1:
                                        print(f"Failed to reinitialize camera: {e}")
                                    time.sleep(0.5)
                        except Exception as e:
                            print(f"Error during camera reinitialization: {e}")
                    time.sleep(0.1)
                    continue

                # Reset error counter on successful read
                consecutive_errors = 0
                frames_read += 1
                
                if frames_read == 1:
                    print(f"[_reader] Successfully read first frame from camera")
                elif frames_read % 30 == 0:
                    print(f"[_reader] Read {frames_read} frames so far")

                # Update FPS counter
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Update FPS every 30 frames
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed if elapsed > 0 else 0

                # Put frame in queue
                # Simple strategy: try to put, if full, remove one old frame and try again
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    # Queue is full, remove oldest frame and add new one
                    try:
                        self.q.get_nowait()  # Remove one old frame
                        self.q.put_nowait(frame)  # Add new frame
                    except:
                        pass  # Skip if there's an error

                # Small delay to prevent 100% CPU usage
                time.sleep(0.01)

            except Exception as e:
                import traceback
                print(f"[_reader] Error in camera reader thread: {e}")
                print(f"[_reader] Traceback: {traceback.format_exc()}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def read(self):
        """Get the most recent frame from the queue"""
        try:
            # Get the latest frame (non-blocking)
            # Read all available frames and keep only the most recent one
            frame = None
            frames_available = 0
            while not self.q.empty():
                try:
                    frame = self.q.get_nowait()
                    frames_available += 1
                except queue.Empty:
                    break
            
            if frame is None:
                # If no frame in queue, return a blank frame with message
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No frame available", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return False, frame
            
            # We got a frame - return it
            return True, frame
            
        except Exception as e:
            print(f"Error in read(): {e}")
            # Return a blank frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Error: {str(e)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False, frame
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def release(self):
        """Stop capture and release resources"""
        with self.lock:
            self.stopped = True
            
            # Clear the queue
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            
            # Wait for thread to finish
            if hasattr(self, 'thread') and self.thread.is_alive():
                self.thread.join(timeout=1.0)
            
            # Release camera
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    if self.cap.isOpened():
                        self.cap.release()
                    print("Camera released successfully")
                except Exception as e:
                    print(f"Error releasing camera: {e}")
    
    def isOpened(self):
        """Check if capture is open - same as main_preview.py"""
        if self.cap is None:
            return False
        return self.cap.isOpened() and not self.stopped
        
    def stop(self):
        """Stop the video capture"""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception as e:
                print(f"Error stopping camera: {e}")


class FrameProcessor:
    """
    Process frames asynchronously with better resource management
    """
    def __init__(self, captioner, classifier, alert_manager, process_every_n=3):
        """Initialize frame processor with required components"""
        self.captioner = captioner
        self.classifier = classifier
        self.alert_manager = alert_manager
        self.process_every_n = max(1, process_every_n)  # Ensure at least 1
        self.frame_count = 0
        self.last_result = None
        self.lock = threading.Lock()
        
        # Processing queue with small size to prevent memory bloat
        self.input_queue = queue.Queue(maxsize=1)  # Only keep most recent frame
        self.result_queue = queue.Queue(maxsize=1)  # Only keep latest result
        self.stopped = False
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        print(f"FrameProcessor started with process_every_n={self.process_every_n}")
    
    def _process_loop(self):
        """Process frames from the input queue with error handling"""
        while not self.stopped:
            try:
                # Clear input queue to only process the latest frame
                while not self.input_queue.empty():
                    try:
                        frame_data = self.input_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Get the most recent frame if available
                try:
                    frame_data = self.input_queue.get(timeout=0.5)
                except queue.Empty:
                    time.sleep(0.01)
                    continue
                
                frame = frame_data['frame']
                frame_num = frame_data['frame_num']
                
                try:
                    # Generate caption (this is the most expensive operation)
                    caption = self.captioner.generate_caption(frame)
                    
                    # Classify the caption
                    prediction = self.classifier.classify(caption)
                    
                    # Check for alerts
                    alert_triggered = False
                    if (self.alert_manager and 
                        hasattr(self.alert_manager, 'should_trigger_alert') and 
                        callable(self.alert_manager.should_trigger_alert)):
                        
                        try:
                            if self.alert_manager.should_trigger_alert(
                                prediction.get('label', 'NORMAL'), 
                                float(prediction.get('confidence', 0))
                            ):
                                self.alert_manager.log_alert(
                                    prediction.get('label', 'NORMAL'),
                                    float(prediction.get('confidence', 0)),
                                    caption,
                                    frame
                                )
                                alert_triggered = True
                        except Exception as e:
                            print(f"Error in alert processing: {e}")
                    
                    # Prepare result
                    result = {
                        'frame_num': frame_num,
                        'caption': caption,
                        'prediction': prediction.get('label', 'NORMAL'),
                        'confidence': float(prediction.get('confidence', 0)),
                        'alert_triggered': alert_triggered,
                        'timestamp': datetime.now().isoformat(),
                        'severity': self._get_severity(float(prediction.get('confidence', 0)))
                    }
                    
                    # Update last result (discard old one if queue is full)
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.result_queue.put(result)
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                
            except Exception as e:
                print(f"Unexpected error in process_loop: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def _get_severity(self, confidence):
        """Determine severity based on confidence"""
        if confidence > 0.7:
            return 'HIGH'
        elif confidence > 0.4:
            return 'MEDIUM'
        return 'LOW'
        
    def process_frame(self, frame):
        """Add a frame to the processing queue if not full"""
        if self.stopped or frame is None:
            return False
            
        self.frame_count += 1
        
        # Skip processing if we're not on the right frame interval
        if self.frame_count % self.process_every_n != 0:
            return False
        
        try:
            # Clear queue if it's full to only process the latest frame
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Add the new frame
            self.input_queue.put_nowait({
                'frame': frame,
                'frame_num': self.frame_count
            })
            return True
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return False
    
    def get_result(self):
        """Get the latest processing result if available"""
        try:
            # Get the most recent result
            result = None
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                except queue.Empty:
                    break
            
            if result is not None:
                self.last_result = result
                
            return self.last_result
            
        except Exception as e:
            print(f"Error getting result: {e}")
            return self.last_result
            
    def stop(self):
        """Stop the frame processor and clean up resources"""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        # Clear the queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                continue
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                continue
