"""
Python application for detecting a laser pointer through a webcam feed.
Author: Wang Yiran
Description: This script detects a red laser pointer using OpenCV by identifying its color and extracting contours.
"""

import cv2
import numpy as np

def detect_laser():
    # Initialize webcam feed
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to HSV (Hue, Saturation, Value) color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the HSV range for detecting the red color of the laser pointer
        lower_red = np.array([0, 100, 100])  # Lower boundary for red
        upper_red = np.array([10, 255, 255])  # Upper boundary for red
        
        # Create a mask to isolate the red regions in the frame
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Perform morphological operations to clean the mask (reduce noise)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours in the mask (detect red regions)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found
        if contours:
            # Select the largest contour to represent the laser dot
            c = max(contours, key=cv2.contourArea)
            
            # Get the minimum enclosing circle around the largest contour
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Only process if the detected radius is significant (avoid noise)
            if radius > 5:
                # Draw the circle around the detected laser dot
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                # Display the coordinates of the laser dot
                cv2.putText(frame, f"Laser Detected at ({int(x)}, {int(y)})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display the resulting frame with the laser pointer detected
        cv2.imshow('Laser Detection', frame)
        
        # Press 'q' to exit the detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_laser()
