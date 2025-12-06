"""
Event detection logic with severity-based alert management
"""
import time
import cv2
import json
from datetime import datetime
from pathlib import Path
import config

# Optional: Sound alert
try:
    import winsound  # Windows
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

class AlertManager:
    def __init__(self):
        """Initialize alert manager with severity-based handling"""
        self.alerts_log = []
        self.last_alert_time = {}
        self.alert_count = {}
        self.session_start = time.time()
        
        # Statistics by severity
        self.severity_stats = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "NORMAL": 0
        }
        
        print("\n" + "="*60)
        print("Alert Thresholds Configured:")
        for severity, threshold in config.ALERT_THRESHOLDS.items():
            print(f"  {severity:10s}: {threshold:.0%} confidence")
        print("="*60 + "\n")
    
    def get_event_severity(self, label):
        """
        Get severity level for an event
        
        Args:
            label: event label
            
        Returns:
            str: severity level
        """
        return config.EVENT_SEVERITY.get(label, "NORMAL")
    
    def should_trigger_alert(self, label, confidence):
        """
        Determine if an alert should be triggered based on severity
        
        Args:
            label: predicted event label
            confidence: prediction confidence
            
        Returns:
            bool: whether to trigger alert
        """
        # Get severity
        severity = self.get_event_severity(label)
        
        # Never alert for normal events
        if severity == "NORMAL":
            return False
        
        # Check confidence threshold based on severity
        threshold = config.ALERT_THRESHOLDS.get(severity, 0.80)
        if confidence < threshold:
            return False
        
        # Check cooldown period based on severity
        current_time = time.time()
        cooldown = config.ALERT_COOLDOWN.get(severity, 5)
        
        if label in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[label]
            if time_since_last < cooldown:
                return False
        
        # Check overall rate limiting
        recent_alerts = [
            a for a in self.alerts_log 
            if current_time - a['timestamp'] < 60
        ]
        if len(recent_alerts) >= config.MAX_ALERTS_PER_MINUTE:
            return False
        
        return True
    
    def log_alert(self, label, confidence, caption, frame=None):
        """
        Log an alert with severity-based handling
        
        Args:
            label: event label
            confidence: prediction confidence
            caption: image caption
            frame: numpy array of the frame (optional)
        """
        timestamp = time.time()
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        severity = self.get_event_severity(label)
        
        alert_data = {
            "timestamp": timestamp,
            "datetime": datetime_str,
            "label": label,
            "severity": severity,
            "confidence": float(confidence),
            "caption": caption
        }
        
        # Save frame if enabled
        if config.SAVE_ALERT_IMAGES and frame is not None:
            image_filename = f"alert_{severity}_{label}_{datetime_str}.jpg"
            image_path = config.ALERT_DIR / image_filename
            cv2.imwrite(str(image_path), frame)
            alert_data["image_path"] = str(image_path)
        
        # Update tracking
        self.alerts_log.append(alert_data)
        self.last_alert_time[label] = timestamp
        self.alert_count[label] = self.alert_count.get(label, 0) + 1
        self.severity_stats[severity] += 1
        
        # Save to JSON
        self._save_alerts_log()
        
        # Print alert with severity-based formatting
        self._print_alert(alert_data)
        
        # Play sound for critical events
        if config.ENABLE_SOUND_ALERT and severity in ["CRITICAL", "HIGH"]:
            self._play_alert_sound(severity)
        
        return alert_data
    
    def _print_alert(self, alert_data):
        """Print formatted alert to console"""
        label = alert_data['label']
        severity = alert_data['severity']
        confidence = alert_data['confidence']
        caption = alert_data['caption']
        datetime_str = alert_data['datetime']
        
        # Severity symbols
        symbols = {
            "CRITICAL": "🚨",
            "HIGH": "⚠️",
            "MEDIUM": "⚡",
            "LOW": "ℹ️",
            "NORMAL": "✓"
        }
        symbol = symbols.get(severity, "•")
        
        print(f"\n{'='*70}")
        print(f"{symbol} [{severity}] {label.upper()} DETECTED {symbol}")
        print(f"{'='*70}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Caption: {caption}")
        print(f"Time: {datetime_str}")
        if 'image_path' in alert_data:
            print(f"Saved: {alert_data['image_path']}")
        print(f"{'='*70}\n")
    
    def _play_alert_sound(self, severity):
        """Play alert sound (Windows only)"""
        if not SOUND_AVAILABLE:
            return
        
        try:
            if severity == "CRITICAL":
                # High-pitched beep for critical
                winsound.Beep(2000, 500)
            elif severity == "HIGH":
                # Medium beep
                winsound.Beep(1500, 300)
        except:
            pass
    
    def _save_alerts_log(self):
        """Save alerts log to JSON file"""
        log_file = config.ALERT_DIR / "alerts_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.alerts_log, f, indent=2)
    
    def get_statistics(self):
        """Get comprehensive alert statistics"""
        session_duration = time.time() - self.session_start
        
        return {
            "total_alerts": len(self.alerts_log),
            "session_duration": f"{session_duration/60:.1f} minutes",
            "alerts_by_event": self.alert_count,
            "alerts_by_severity": self.severity_stats,
            "recent_alerts": self.alerts_log[-5:] if self.alerts_log else []
        }
    
    def get_severity_summary(self):
        """Get summary by severity level"""
        summary = []
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = self.severity_stats[severity]
            if count > 0:
                summary.append(f"{severity}: {count}")
        return ", ".join(summary) if summary else "No alerts"
