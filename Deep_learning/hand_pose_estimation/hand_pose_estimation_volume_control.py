import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess
import platform
import os

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

# Volume control variables
current_volume = 50
volume_changed_time = 0
last_volume = 50

try:
    from pynput.keyboard import Controller, Key
    keyboard = Controller()
    pynput_available = True
except ImportError:
    print("pynput not found. Install with: pip install pynput")
    pynput_available = False

def set_volume(volume):
    """Set system volume using keyboard volume keys"""
    global last_volume
    
    if not pynput_available:
        return
    
    volume = max(0, min(100, volume))
    
    if platform.system() == "Windows":
        try:
            # Calculate difference and press keys accordingly
            diff = volume - last_volume
            
            if diff > 0:
                # Increase volume
                for _ in range(min(int(abs(diff) / 2), 5)):  # Limit rapid presses
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
            elif diff < 0:
                # Decrease volume
                for _ in range(min(int(abs(diff) / 2), 5)):
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
            
            last_volume = volume
        except Exception as e:
            print(f"Volume control error: {e}")
    
    elif platform.system() == "Darwin":
        os.system(f'osascript -e "set volume output volume {int(volume)}"')
    
    elif platform.system() == "Linux":
        os.system(f'amixer set Master {int(volume)}%')

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
    
    fingers_raised = []
    fingers_raised.append(index_tip.y < index_pip.y)
    fingers_raised.append(middle_tip.y < middle_pip.y)
    fingers_raised.append(ring_tip.y < ring_pip.y)
    fingers_raised.append(pinky_tip.y < pinky_pip.y)
    
    thumb_raised = distance(thumb_tip, wrist) > distance(thumb_pip, wrist)
    
    return fingers_raised, thumb_raised

def distance(p1, p2):
    """Calculates Euclidean distance between two landmark points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def get_hand_position(landmarks):
    """Get the position of the hand (wrist coordinates)"""
    wrist = landmarks.landmark[0]
    return (wrist.x, wrist.y)

def get_hand_gesture(landmarks):
    """Detect simple hand gestures"""
    if landmarks is None:
        return "Unknown"
    
    lm = landmarks.landmark
    fingers_raised, thumb_raised = get_finger_state(landmarks)
    raised_count = sum(fingers_raised)
    
    thumb_tip = lm[4]
    index_tip = lm[8]
    
    if distance(thumb_tip, index_tip) < 0.05 and raised_count == 2 and fingers_raised[1] and fingers_raised[2]:
        return "OK Sign"
    elif thumb_raised and raised_count == 0:
        if lm[8].y > lm[4].y:
            return "Thumbs Up!"
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
        return f"Custom ({raised_count} up)"

def draw_virtual_hand(canvas, landmarks, handedness, fingers_raised, thumb_raised):
    """Draw virtual hand on canvas with skeleton and colored joints based on state."""
    if landmarks is None:
        return canvas
    
    h, w = canvas.shape[:2]
    points = []
    
    for i, lm in enumerate(landmarks.landmark):
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        
        color = (0, 0, 255)
        
        if i in [5, 6, 7, 8] and fingers_raised[0]:
            color = (0, 255, 0)
        elif i in [9, 10, 11, 12] and fingers_raised[1]:
            color = (0, 255, 0)
        elif i in [13, 14, 15, 16] and fingers_raised[2]:
            color = (0, 255, 0)
        elif i in [17, 18, 19, 20] and fingers_raised[3]:
            color = (0, 255, 0)
        elif i in [1, 2, 3, 4] and thumb_raised:
            color = (0, 255, 0)
            
        cv2.circle(canvas, (x, y), 5, color, -1)
    
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        start_pos = points[start_idx]
        end_pos = points[end_idx]
        cv2.line(canvas, start_pos, end_pos, (0, 150, 0), 2)
    
    label = "Right Hand" if handedness == "Right" else "Left Hand"
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return canvas

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hands_results = hands.process(rgb_frame)
    virtual_hand_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for hand_landmarks, handedness in zip(
            hands_results.multi_hand_landmarks,
            hands_results.multi_handedness
        ):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
            
            fingers_raised, thumb_raised = get_finger_state(hand_landmarks)
            
            hand_label = handedness.classification[0].label
            virtual_hand_canvas = draw_virtual_hand(
                virtual_hand_canvas,
                hand_landmarks,
                hand_label,
                fingers_raised,
                thumb_raised
            )
            
            gesture = get_hand_gesture(hand_landmarks)
            
            # VOLUME CONTROL: Use vertical hand position for volume
            hand_x, hand_y = get_hand_position(hand_landmarks)
            # Map hand height (0-1) to volume (0-100)
            new_volume = int((1 - hand_y) * 100)
            current_volume = new_volume
            set_volume(current_volume)
            
            cv2.putText(
                virtual_hand_canvas,
                f"Gesture: {gesture}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )
            
            cv2.putText(
                virtual_hand_canvas,
                f"Volume: {current_volume}%",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                f"Volume: {current_volume}%",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
    
    virtual_hand_resized = cv2.resize(virtual_hand_canvas, (400, 300))
    x_offset = frame.shape[1] - 400 - 10
    y_offset = frame.shape[0] - 300 - 10
    
    frame[y_offset:y_offset+300, x_offset:x_offset+400] = virtual_hand_resized
    cv2.rectangle(frame, (x_offset-2, y_offset-2), (x_offset+402, y_offset+302), (0, 255, 0), 2)
    
    cv2.imshow("Hand Control - Volume", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
