"""
Alert management for the surveillance system
"""
import os
import json
import cv2
from datetime import datetime

class AlertManager:
    def __init__(self, alerts_folder='alerts', min_confidence=0.4):
        """Initialize the alert manager"""
        self.alerts_folder = alerts_folder
        self.min_confidence = min_confidence
        
        # Create alerts directory if it doesn't exist
        os.makedirs(self.alerts_folder, exist_ok=True)
        
        # Initialize alerts log file
        self.alerts_file = os.path.join(self.alerts_folder, 'alerts_log.json')
        if not os.path.exists(self.alerts_file):
            with open(self.alerts_file, 'w') as f:
                json.dump([], f)
    
    def should_trigger_alert(self, label, confidence):
        """
        Determine if an alert should be triggered based on label and confidence
        
        Args:
            label (str): The predicted label
            confidence (float): The confidence score (0-1)
            
        Returns:
            bool: True if an alert should be triggered, False otherwise
        """
        # Only trigger alerts for non-NORMAL predictions with sufficient confidence
        return label != 'NORMAL' and confidence >= self.min_confidence
    
    def log_alert(self, label, confidence, caption, frame=None):
        """
        Log an alert to the alerts log
        
        Args:
            label (str): The predicted label
            confidence (float): The confidence score (0-1)
            caption (str): The generated caption
            frame: Optional frame data to save with the alert
        """
        # Create alert data
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'confidence': float(confidence),
            'caption': caption,
            'severity': self._get_severity(confidence)
        }
        
        # Save frame if provided
        if frame is not None:
            frame_filename = f"alert_{int(datetime.now().timestamp())}.jpg"
            frame_path = os.path.join(self.alerts_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            alert_data['frame_path'] = frame_filename
        
        # Load existing alerts
        try:
            with open(self.alerts_file, 'r') as f:
                alerts = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            alerts = []
        
        # Add new alert
        alerts.append(alert_data)
        
        # Save updated alerts
        with open(self.alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def _get_severity(self, confidence):
        """Determine severity based on confidence"""
        if confidence > 0.7:
            return 'HIGH'
        elif confidence > 0.4:
            return 'MEDIUM'
        return 'LOW'
    
    def get_recent_alerts(self, limit=10):
        """Get the most recent alerts"""
        try:
            with open(self.alerts_file, 'r') as f:
                alerts = json.load(f)
            return alerts[-limit:]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
