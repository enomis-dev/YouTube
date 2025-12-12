# Requirements: mediapipe, opencv-python, numpy
# pip install mediapipe opencv-python numpy

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque, Counter

# -------------------------
# Settings
# -------------------------
MAX_HISTORY = 6  # number of frames to smooth over (increase for more stability)
OK_DIST_THRESHOLD = 0.05  # normalized distance threshold for OK sign
THUMBS_UP_Y_DIFF = 0.05   # how much thumb tip should be above thumb mcp (normalized) for thumbs-up
# -------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Webcam init
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Virtual canvas
canvas_width, canvas_height = 800, 600
virtual_hand_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# History buffers for smoothing
fingers_history = deque(maxlen=MAX_HISTORY)  # each element: [index,middle,ring,pinky] booleans
thumb_history = deque(maxlen=MAX_HISTORY)    # each element: bool
gesture_history = deque(maxlen=MAX_HISTORY)  # recent detected gestures (strings)

def euclidean_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def smooth_bool_list(history_deque):
    """Return boolean list or value averaged across history (majority voting)."""
    if len(history_deque) == 0:
        return None
    # If elements are lists (finger booleans), compute per-index majority
    if isinstance(history_deque[0], (list, tuple)):
        L = len(history_deque[0])
        result = []
        for i in range(L):
            votes = sum(1 for item in history_deque if item[i])
            result.append(votes >= (len(history_deque) / 2.0))
        return result
    else:
        # single boolean values
        votes = sum(1 for v in history_deque if v)
        return votes >= (len(history_deque) / 2.0)

def get_finger_state(landmarks, handedness_label="Right"):
    """
    Return (fingers_raised_list, thumb_raised_bool)
    fingers_raised_list = [index, middle, ring, pinky] as booleans
    Thumb uses x-axis ordering and handedness.
    """
    lm = landmarks.landmark

    # Fingers (compare tip vs pip on Y axis; y smaller = higher on image)
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y

    fingers_raised = [index_up, middle_up, ring_up, pinky_up]

    # Thumb: use x ordering of tip (4), ip (3) and mcp (2)
    thumb_tip = lm[4]
    thumb_ip  = lm[3]
    thumb_mcp = lm[2]

    # handedness_label typically "Left" or "Right" (as provided by MediaPipe)
    if handedness_label.lower().startswith("r"):
        # For right hand (mirrored selfie view), thumb extends to the LEFT (smaller x)
        thumb_raised = (thumb_tip.x < thumb_ip.x) and (thumb_ip.x < thumb_mcp.x)
    else:
        # For left hand, thumb extends to the RIGHT (larger x)
        thumb_raised = (thumb_tip.x > thumb_ip.x) and (thumb_ip.x > thumb_mcp.x)

    return fingers_raised, thumb_raised

def draw_virtual_hand(canvas, landmarks, handedness, fingers_raised, thumb_raised):
    """Draw a simplified virtual hand skeleton and color joints based on raised state."""
    if landmarks is None:
        return canvas

    h, w = canvas.shape[:2]
    points = []

    # Pre-calc smoothed booleans
    # color green for raised, red for down
    for i, lm in enumerate(landmarks.landmark):
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

        color = (0, 0, 255)  # red default (BGR)

        # Index joints 5-8, Middle 9-12, Ring 13-16, Pinky 17-20, Thumb 1-4
        if i in [5,6,7,8] and fingers_raised[0]:
            color = (0, 255, 0)
        elif i in [9,10,11,12] and fingers_raised[1]:
            color = (0, 255, 0)
        elif i in [13,14,15,16] and fingers_raised[2]:
            color = (0, 255, 0)
        elif i in [17,18,19,20] and fingers_raised[3]:
            color = (0, 255, 0)
        elif i in [1,2,3,4] and thumb_raised:
            color = (0, 255, 0)

        cv2.circle(canvas, (x, y), 6, color, -1)

    # draw bone connections (MediaPipe connections)
    for conn in mp_hands.HAND_CONNECTIONS:
        s = conn[0]
        e = conn[1]
        start = points[s]
        end = points[e]
        cv2.line(canvas, start, end, (0, 150, 0), 2)

    label = "Right Hand" if handedness.lower().startswith("r") else "Left Hand"
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    return canvas

def detect_gesture_from_states(landmarks, fingers_raised, thumb_raised):
    """
    A more robust gesture decision using smoothed finger states and landmark geometry:
    - OK sign (thumb+index touching and middle/ring/pinky down OR any combination)
    - Thumbs Up: thumb extended upwards (thumb_tip.y < thumb_mcp.y by threshold) and other fingers down
    - Open Hand: 4 fingers up + thumb up
    - Closed Fist: no fingers, no thumb
    - Pointing, Peace, Three fingers as before
    """
    if landmarks is None:
        return "Unknown"

    lm = landmarks.landmark

    raised_count = sum(fingers_raised)
    # thumb & index tip positions for OK
    thumb_tip = lm[4]
    index_tip = lm[8]
    thumb_mcp = lm[2]
    wrist = lm[0]

    # distance between thumb tip and index tip (normalized)
    tip_dist = euclidean_distance(thumb_tip, index_tip)

    # OK sign: thumb & index close AND other fingers down (or at least not all raised)
    # allow some tolerance: if tip distance small and (middle, ring, pinky down)
    if tip_dist < OK_DIST_THRESHOLD and not fingers_raised[1] and not fingers_raised[2] and not fingers_raised[3]:
        return "OK Sign"

    # Thumbs Up:
    # - thumb is geometrically above its mcp by some normalized amount (y smaller on image)
    # - other fingers are down (raised_count == 0)
    # - also ensure thumb is extended (thumb_raised True)
    if thumb_raised and raised_count == 0 and (thumb_tip.y + THUMBS_UP_Y_DIFF) < thumb_mcp.y:
        return "Thumbs Up!"

    # Original/other gestures
    if raised_count == 4 and thumb_raised:
        return "Open Hand"
    if raised_count == 0 and not thumb_raised:
        return "Closed Fist"
    if raised_count == 1 and fingers_raised[0]:
        return "Pointing"
    if raised_count == 2 and fingers_raised[0] and fingers_raised[1]:
        return "Peace Sign"
    if raised_count == 3:
        return "Three Fingers"

    # fallback
    return f"Custom ({raised_count} up)"

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # selfie view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # clear virtual canvas each time
    virtual_hand_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    current_fingers = None
    current_thumb = None
    current_gesture = "No Hand"

    if results.multi_hand_landmarks and results.multi_handedness:
        # we handle only first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label  # "Left" or "Right"

        # compute raw states for this frame
        raw_fingers, raw_thumb = get_finger_state(hand_landmarks, handedness)
        current_fingers = raw_fingers
        current_thumb = raw_thumb

        # push into history deques
        fingers_history.append(raw_fingers)
        thumb_history.append(raw_thumb)

        # smoothed states (majority vote across history)
        smoothed_fingers = smooth_bool_list(fingers_history)
        smoothed_thumb = smooth_bool_list(thumb_history)

        # detect gesture from smoothed states and landmarks
        gesture_candidate = detect_gesture_from_states(hand_landmarks, smoothed_fingers, smoothed_thumb)

        gesture_history.append(gesture_candidate)
        # stable gesture = mode of recent gestures (or candidate if short history)
        if len(gesture_history) > 0:
            most_common = Counter(gesture_history).most_common(1)[0][0]
            stable_gesture = most_common
        else:
            stable_gesture = gesture_candidate

        # Draw landmarks on camera preview (original view)
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

        # Draw on virtual canvas using smoothed states
        virtual_hand_canvas = draw_virtual_hand(
            virtual_hand_canvas,
            hand_landmarks,
            handedness,
            smoothed_fingers,
            smoothed_thumb
        )

        current_gesture = stable_gesture

        # Show gesture on virtual canvas and webcam
        cv2.putText(virtual_hand_canvas, f"Gesture: {current_gesture}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # No hand detected => reset nothing, keep history for a short time
        cv2.putText(frame, "Gesture: No Hand", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(virtual_hand_canvas, "No Hand", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    # Resize and overlay virtual canvas into the webcam frame
    virtual_hand_resized = cv2.resize(virtual_hand_canvas, (400, 300))
    x_offset = frame.shape[1] - 400 - 10
    y_offset = frame.shape[0] - 300 - 10
    frame[y_offset:y_offset+300, x_offset:x_offset+400] = virtual_hand_resized
    cv2.rectangle(frame, (x_offset-2, y_offset-2), (x_offset+402, y_offset+302), (0,255,0), 2)

    cv2.imshow("Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
