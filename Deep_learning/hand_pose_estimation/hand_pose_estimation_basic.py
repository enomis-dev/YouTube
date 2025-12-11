import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create a blank canvas for virtual hand
canvas_width, canvas_height = 800, 600
virtual_hand_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

def get_finger_state(landmarks):
    """Calculates which fingers are raised and returns the state for gesture detection."""
    lm = landmarks.landmark
    
    # Get key points
    thumb_tip = lm[4]
    thumb_pip = lm[3]
    index_tip = lm[8]
    index_pip = lm[6]
    middle_tip = lm[12]
    middle_pip = lm[10]
    ring_tip = lm[16]
    ring_pip = lm[14]
    pinky_tip = lm[20]
    pinky_pip = lm[18]
    wrist = lm[0]
    
    # Check if finger is raised (tip above pip) - index 0=index, 1=middle, 2=ring, 3=pinky
    # Note: Thumb logic is more complex, so we handle it separately.
    fingers_raised = []
    fingers_raised.append(index_tip.y < index_pip.y)   # Index
    fingers_raised.append(middle_tip.y < middle_pip.y) # Middle
    fingers_raised.append(ring_tip.y < ring_pip.y)     # Ring
    fingers_raised.append(pinky_tip.y < pinky_pip.y)   # Pinky
    
    # Thumb logic: check if the thumb tip is further from the wrist than the thumb pip on the x-axis (for right hand/flipped view)
    # This is a simplified check that works better than a simple y-comparison for the thumb.
    thumb_raised = distance(thumb_tip, wrist) > distance(thumb_pip, wrist)
    
    return fingers_raised, thumb_raised

def distance(p1, p2):
    """Calculates Euclidean distance between two landmark points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def draw_virtual_hand(canvas, landmarks, handedness, fingers_raised, thumb_raised):
    """Draw virtual hand on canvas with skeleton and colored joints based on state."""
    if landmarks is None:
        return canvas
    
    h, w = canvas.shape[:2]
    points = []
    
    # Convert normalized coordinates to canvas coordinates and draw joints
    for i, lm in enumerate(landmarks.landmark):
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        
        # Color joints based on finger state (New Feature 2)
        color = (0, 0, 255) # Default Red
        
        # Landamark indices: Index=5-8, Middle=9-12, Ring=13-16, Pinky=17-20
        if i in [5, 6, 7, 8] and fingers_raised[0]:
            color = (0, 255, 0) # Green for raised Index
        elif i in [9, 10, 11, 12] and fingers_raised[1]:
            color = (0, 255, 0) # Green for raised Middle
        elif i in [13, 14, 15, 16] and fingers_raised[2]:
            color = (0, 255, 0) # Green for raised Ring
        elif i in [17, 18, 19, 20] and fingers_raised[3]:
            color = (0, 255, 0) # Green for raised Pinky
        elif i in [1, 2, 3, 4] and thumb_raised:
            color = (0, 255, 0) # Green for raised Thumb
            
        cv2.circle(canvas, (x, y), 5, color, -1) # Slightly larger circle
    
    # Draw connections
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        start_pos = points[start_idx]
        end_pos = points[end_idx]
        cv2.line(canvas, start_pos, end_pos, (0, 150, 0), 2) # Slightly darker green lines
    
    # Add label
    label = "Right Hand" if handedness == "Right" else "Left Hand"
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return canvas

def get_hand_gesture(landmarks):
    """Detect simple hand gestures (Updated with Thumb and OK/Thumbs Up)."""
    if landmarks is None:
        return "Unknown"
    
    lm = landmarks.landmark
    fingers_raised, thumb_raised = get_finger_state(landmarks) # Get finger states
    raised_count = sum(fingers_raised)
    
    # Get key points for specific gesture checks
    thumb_tip = lm[4]
    index_tip = lm[8]
    
    # NEW GESTURES (New Feature 1)
    # Check for OK Sign: Index tip and Thumb tip are close, and other three fingers are raised.
    # We use a threshold distance based on normalized coordinates.
    if distance(thumb_tip, index_tip) < 0.05 and raised_count == 2 and fingers_raised[1] and fingers_raised[2]:
        # Middle, Ring, Pinky should technically be down, but if we assume they are down:
        return "OK Sign"
        
    # Thumbs Up: Only thumb is raised and other fingers are closed/bent (raised_count == 0)
    elif thumb_raised and raised_count == 0:
        # Check if index finger is significantly lower than thumb tip
        if lm[8].y > lm[4].y:
            return "Thumbs Up!"
    
    # ORIGINAL GESTURES
    elif raised_count == 4 and thumb_raised:
        return "Open Hand"
    elif raised_count == 0 and not thumb_raised:
        return "Closed Fist"
    elif raised_count == 1 and fingers_raised[0]:
        return "Pointing"
    elif raised_count == 2 and fingers_raised[0] and fingers_raised[1]:
        return "Peace Sign"
    elif raised_count == 3:
        return "Three Fingers"
    else:
        # Use a more descriptive fallback
        return f"Custom ({raised_count} up)"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame")
        break
    
    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands
    hands_results = hands.process(rgb_frame)
    
    # Clear canvas each frame
    virtual_hand_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Draw webcam hand landmarks
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for hand_landmarks, handedness in zip(
            hands_results.multi_hand_landmarks,
            hands_results.multi_handedness
        ):
            # Draw on webcam feed (Original)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
            
            # Get finger states for visualization and detection
            fingers_raised, thumb_raised = get_finger_state(hand_landmarks)
            
            # Draw on virtual canvas (Uses new coloring logic)
            hand_label = handedness.classification[0].label
            virtual_hand_canvas = draw_virtual_hand(
                virtual_hand_canvas,
                hand_landmarks,
                hand_label,
                fingers_raised,
                thumb_raised
            )
            
            # Detect and display gesture (Uses new gesture logic)
            gesture = get_hand_gesture(hand_landmarks)
            
            # Display gesture on virtual canvas
            cv2.putText(
                virtual_hand_canvas,
                f"Gesture: {gesture}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )
            
            # Also show on webcam
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
    
    # Resize virtual hand canvas to fit as subwindow
    virtual_hand_resized = cv2.resize(virtual_hand_canvas, (400, 300))
    
    # Place virtual hand in bottom-right corner of webcam frame
    x_offset = frame.shape[1] - 400 - 10
    y_offset = frame.shape[0] - 300 - 10
    
    # Create ROI and place the virtual hand
    frame[y_offset:y_offset+300, x_offset:x_offset+400] = virtual_hand_resized
    
    # Add border around subwindow
    cv2.rectangle(frame, (x_offset-2, y_offset-2), (x_offset+402, y_offset+302), (0, 255, 0), 2)
    
    # Display combined feed
    cv2.imshow("Hand Control", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
