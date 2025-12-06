"""
Optimized main surveillance script with multi-threading
"""
import cv2
import argparse
import time
from captioner import OptimizedCaptioner
from caption_classifier import OptimizedCaptionClassifier
from event_logic import AlertManager
from video_processor import ThreadedVideoCapture, FrameProcessor
import config

def draw_overlay(frame, caption, prediction, fps):
    """Draw information overlay on frame with severity-based coloring"""
    height, width = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # FPS
    if config.DISPLAY_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Caption
    if config.DISPLAY_CAPTION and caption:
        caption_text = caption[:70] + "..." if len(caption) > 70 else caption
        cv2.putText(frame, f"Caption: {caption_text}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Prediction with severity-based coloring
    if config.DISPLAY_PREDICTIONS and prediction:
        label = prediction['label']
        confidence = prediction['confidence']
        
        # Get severity and color
        severity = config.EVENT_SEVERITY.get(label, "NORMAL")
        color = config.SEVERITY_COLORS.get(severity, (255, 255, 255))
        
        # Main prediction
        text = f"Event: {label.replace('_', ' ')} ({confidence:.1%})"
        cv2.putText(frame, text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Severity badge
        severity_text = f"[{severity}]"
        cv2.putText(frame, severity_text, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Top 3 predictions (excluding Normal if not top)
        sorted_scores = sorted(
            prediction['all_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter out normal events from top predictions unless it's the top one
        top_predictions = []
        for event, score in sorted_scores:
            if event != "Normal_Videos_event" or len(top_predictions) == 0:
                top_predictions.append((event, score))
            if len(top_predictions) >= 3:
                break
        
        y_offset = 125
        for i, (event, score) in enumerate(top_predictions[:3]):
            event_severity = config.EVENT_SEVERITY.get(event, "NORMAL")
            event_color = config.SEVERITY_COLORS.get(event_severity, (200, 200, 200))
            text = f"  {i+1}. {event.replace('_', ' ')}: {score:.1%}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, event_color, 1)
            y_offset += 20
    
    return frame

def main():
    """Main surveillance loop"""
    parser = argparse.ArgumentParser(description="Campus Surveillance System - Crime Detection")
    parser.add_argument('--camera', default=config.DEFAULT_CAMERA,
                        help='Camera source (0 for webcam, RTSP URL for IP camera)')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without display (headless mode)')
    args = parser.parse_args()
    
    print("="*70)
    print("  CAMPUS SURVEILLANCE SYSTEM - CRIME DETECTION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Camera: {args.camera}")
    print(f"  GPU Enabled: {config.ENABLE_GPU}")
    print(f"  Process Every N Frames: {config.PROCESS_EVERY_N_FRAMES}")
    print(f"  Alert Thresholds:")
    print(f"    CRITICAL: {config.ALERT_THRESHOLDS['CRITICAL']:.0%}")
    print(f"    HIGH:     {config.ALERT_THRESHOLDS['HIGH']:.0%}")
    print(f"    MEDIUM:   {config.ALERT_THRESHOLDS['MEDIUM']:.0%}")
    print(f"    LOW:      {config.ALERT_THRESHOLDS['LOW']:.0%}")
    print(f"  Cache Enabled: {config.ENABLE_CAPTION_CACHE}")
    print()
    
    # Initialize components
    print("Initializing models...")
    captioner = OptimizedCaptioner()
    classifier = OptimizedCaptionClassifier()
    alert_manager = AlertManager()
    
    print("\nStarting video capture...")
    cap = ThreadedVideoCapture(args.camera)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    # Initialize frame processor
    processor = FrameProcessor(
        captioner,
        classifier,
        alert_manager,
        process_every_n=config.PROCESS_EVERY_N_FRAMES
    )
    
    print("\n" + "="*70)
    print("  SURVEILLANCE ACTIVE - Press 'Q' to quit, 'S' to save snapshot")
    print("="*70 + "\n")
    
    # Latest results for display
    latest_caption = ""
    latest_prediction = None
    
    try:
        while True:
            # Get frame
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame")
                break
            
            # Submit frame for processing
            processor.process_frame(frame)
            
            # Check for results
            result = processor.get_result()
            if result:
                latest_caption = result['caption']
                latest_prediction = result['prediction']
            
            # Draw overlay
            if not args.no_display:
                display_frame = draw_overlay(
                    frame.copy(),
                    latest_caption,
                    latest_prediction,
                    cap.get_fps()
                )
                
                # Show frame
                cv2.imshow('Campus Surveillance - Crime Detection', display_frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nShutting down...")
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save current frame
                    filename = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(config.ALERT_DIR / filename), frame)
                    print(f"Saved snapshot: {filename}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print detailed statistics
        print("\n" + "="*70)
        print("  SESSION STATISTICS - CRIME DETECTION")
        print("="*70)
        
        print(f"\nCaption Cache:")
        cache_stats = captioner.get_cache_stats()
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nAlert Summary:")
        alert_stats = alert_manager.get_statistics()
        print(f"  Total Alerts: {alert_stats['total_alerts']}")
        print(f"  Session Duration: {alert_stats['session_duration']}")
        
        print(f"\n  By Severity:")
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = alert_stats['alerts_by_severity'][severity]
            if count > 0:
                print(f"    {severity}: {count}")
        
        if alert_stats['alerts_by_event']:
            print(f"\n  By Event Type:")
            for event, count in sorted(alert_stats['alerts_by_event'].items(), 
                                       key=lambda x: x[1], reverse=True):
                severity = config.EVENT_SEVERITY.get(event, "NORMAL")
                print(f"    {event}: {count} [{severity}]")
        
        if alert_stats['recent_alerts']:
            print(f"\n  Recent Alerts:")
            for alert in alert_stats['recent_alerts'][-3:]:
                print(f"    - {alert['datetime']}: {alert['label']} "
                      f"({alert['confidence']:.1%})")
        
        print("\n" + "="*70)
        print("  Surveillance system stopped")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
