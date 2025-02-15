import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time
import os
from ultralytics import YOLO

@dataclass
class Zone:
    id: int
    x: int
    y: int
    width: int
    height: int
    timer: float = 0
    is_occupied: bool = False
    last_update: float = time.time()

class DetectionSystem:
    def __init__(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Error opening video file")
        
        # Initialize modes
        self.MODE_PEOPLE = 'people'
        self.MODE_HEATMAP = 'heatmap'
        self.current_mode = self.MODE_PEOPLE
        
        # Initialize zones
        self.zones = [
            Zone(1, 300, 200, 200, 200),    # zone_rect_right
            Zone(2, 1050, 200, 200, 200)    # zone_rect_comparison
        ]
        
        # Initialize heatmap
        self.heatmap = None
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model
        # Set confidence threshold
        self.conf_threshold = 0.3

    def update_zone_timers(self):
        """Update timers for occupied zones"""
        current_time = time.time()
        for zone in self.zones:
            if zone.is_occupied:
                time_diff = current_time - zone.last_update
                zone.timer += time_diff
            zone.last_update = current_time

    def is_point_in_zone(self, point: Tuple[int, int], zone: Zone) -> bool:
        """Check if a point is within a zone"""
        x, y = point
        return (zone.x <= x <= zone.x + zone.width and 
                zone.y <= y <= zone.y + zone.height)

    def detect_people(self, frame):
        """Detect people using YOLO and update zones"""
        # Initialize heatmap if not already done
        if self.heatmap is None:
            self.heatmap = np.zeros_like(frame, dtype=np.float32)
        
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold)
        boxes = []
        
        # Reset occupation status
        for zone in self.zones:
            zone.is_occupied = False
        
        # Process detections
        for result in results:
            for box in result.boxes:
                # Check if the detection is a person (class 0 in COCO dataset)
                if box.cls == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append((x1, y1, w, h))
                    
                    # Update heatmap

                    self.heatmap[y1:y2, x1:x2] += 0.5
                    
                    # Update zone occupation
                    center_point = (x1 + w//2, y1 + h//2)
                    for zone in self.zones:
                        if self.is_point_in_zone(center_point, zone):
                            zone.is_occupied = True
        
        # Clip heatmap values

            self.heatmap = np.clip(self.heatmap, 0, 255)
        
        return boxes

    def process_frame(self, frame):
        """Process a single frame based on current mode"""
        boxes = self.detect_people(frame)
        self.update_zone_timers()
        
        output_frame = frame.copy()
        
        if self.current_mode == self.MODE_PEOPLE:
            # Draw zones and timers
            for zone in self.zones:
                color = (0, 0, 255) if zone.is_occupied else (128, 128, 128)
                cv2.rectangle(output_frame, 
                            (zone.x, zone.y),
                            (zone.x + zone.width, zone.y + zone.height),
                            color, 2)
                
                label = "Right" if zone.id == 1 else "Comparison"
                timer_text = f"{label}: {zone.timer:.1f}s"
                cv2.putText(output_frame, timer_text,
                          (zone.x + 5, zone.y + 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                          (255, 255, 255), 2)
            
            # Draw detected people
            for (x, y, w, h) in boxes:
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
        elif self.current_mode == self.MODE_HEATMAP:
            # Apply colormap to heatmap
            heatmap_image = cv2.applyColorMap(np.uint8(self.heatmap), cv2.COLORMAP_JET)
            # Blend with original frame
            output_frame = cv2.addWeighted(frame, 0.7, heatmap_image, 0.3, 0)
        
        return output_frame

    def toggle_mode(self, key):
        """Toggle between modes based on key press"""
        if key == ord('u'):
            self.current_mode = self.MODE_HEATMAP
        elif key == ord('p'):
            self.current_mode = self.MODE_PEOPLE

    def release(self):
        """Release video capture resources"""
        self.cap.release()

def main():
    video_path = "video/mall.mp4"
    
    try:
        detection_system = DetectionSystem(video_path)
        
        while True:
            ret, frame = detection_system.cap.read()
            if not ret:
                print("Video processing completed")
                break

            output_frame = detection_system.process_frame(frame)

            mode_text = f"Mode: {detection_system.current_mode}"
            cv2.putText(output_frame, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Detection System', output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            detection_system.toggle_mode(key)

        # Show final stats
        print("\nFinal Results:")
        for zone in detection_system.zones:
            label = "Right" if zone.id == 1 else "Comparison"
            print(f"{label} Zone: {zone.timer:.1f} seconds")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the video file exists in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'detection_system' in locals():
            detection_system.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()