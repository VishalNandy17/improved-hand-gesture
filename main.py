import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import time
from datetime import datetime
import os

class HandGestureCalculator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.expression = []
        self.gesture_history = deque(maxlen=10)
        self.last_gesture = None
        self.current_number = ""
        self.last_number_time = 0
        self.last_number_gesture = None
        self.window_name = 'Hand Gesture Calculator'
        self.theme = {
            'background': (40, 44, 52),
            'text': (255, 255, 255),
            'accent': (97, 175, 239),
            'success': (152, 195, 121),
            'error': (224, 108, 117),
            'highlight': (229, 192, 123)
        }
        self.setup_logging()
        self.tutorial_mode = True
        self.tutorial_step = 0
        self.tutorial_steps = [
            "Welcome! Let's learn how to use the calculator.",
            "Show numbers 0-9 using finger combinations.",
            "Use thumb only for addition (+)",
            "Use pinky only for subtraction (-)",
            "Show thumb + index + pinky for multiplication (*)",
            "Show thumb + pinky close together for division (/)",
            "Use index + ring fingers for equals (=)",
            "Hold any number for 5 seconds to repeat it",
            "Tutorial complete! Press 'T' to toggle tutorial mode"
        ]
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.gesture_accuracy = {}
        self.feedback_messages = deque(maxlen=3)
        self.message_timers = deque(maxlen=3)
        self.MESSAGE_DURATION = 2.0

    def setup_logging(self):
        self.log_file = f"calculator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(self.log_file, 'w') as f:
            f.write("Hand Gesture Calculator Session Log\n")
            f.write(f"Session started at: {datetime.now()}\n\n")

    def add_feedback_message(self, message, color=None):
        if color is None:
            color = self.theme['text']
        self.feedback_messages.append((message, color))
        self.message_timers.append(time.time())

    def update_feedback_messages(self):
        current_time = time.time()
        while self.message_timers and (current_time - self.message_timers[0]) > self.MESSAGE_DURATION:
            self.message_timers.popleft()
            self.feedback_messages.popleft()

    def log_action(self, action, details):
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {action} - {details}\n")
        self.add_feedback_message(f"{action}: {details}")

    def get_finger_state(self, hand_landmarks):
        finger_tips = [4, 8, 12, 16, 20]
        finger_state = [0, 0, 0, 0, 0]
        for i, tip in enumerate(finger_tips):
            if i == 0:
                if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x:
                    finger_state[i] = 1
            else:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    finger_state[i] = 1
        return finger_state

    def get_finger_angles(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        finger_bases = [1, 5, 9, 13, 17]
        finger_tips = [4, 8, 12, 16, 20]
        angles = []
        for base, tip in zip(finger_bases, finger_tips):
            base_point = hand_landmarks.landmark[base]
            tip_point = hand_landmarks.landmark[tip]
            angle = math.degrees(math.atan2(tip_point.y - wrist.y, tip_point.x - wrist.x) - 
                               math.atan2(base_point.y - wrist.y, base_point.x - wrist.x))
            angle = (angle + 360) % 360
            angles.append(angle)
        return angles

    def interpret_gesture(self, finger_state, finger_angles):
        gestures = {
            tuple([0, 0, 0, 0, 0]): (0, 0.9),
            tuple([0, 1, 0, 0, 0]): (1, 0.9),
            tuple([0, 1, 1, 0, 0]): (2, 0.85),
            tuple([0, 1, 1, 1, 0]): (3, 0.8),
            tuple([0, 1, 1, 1, 1]): (4, 0.8),
            tuple([1, 1, 1, 1, 1]): (5, 0.9),
            tuple([1, 0, 0, 0, 0]): ('+', 0.85),
            tuple([0, 0, 0, 0, 1]): ('-', 0.85),
            tuple([1, 1, 0, 0, 1]): ('*', 0.8),
            tuple([1, 0, 0, 0, 1]): ('/', 0.8),
            tuple([0, 1, 0, 1, 0]): ('=', 0.85),
            tuple([0, 1, 0, 1, 1]): ('C', 0.85),
            tuple([1, 0, 1, 0, 0]): (6, 0.8),
            tuple([1, 0, 1, 1, 0]): (7, 0.8),
            tuple([1, 0, 1, 1, 1]): (8, 0.8),
            tuple([1, 1, 0, 1, 0]): (9, 0.8),
        }
        finger_tuple = tuple(finger_state)
        if finger_tuple in gestures:
            gesture, confidence = gestures[finger_tuple]
            if gesture == '/' and finger_angles[0] >= 90:
                return None
            if gesture not in self.gesture_accuracy:
                self.gesture_accuracy[gesture] = deque(maxlen=100)
            self.gesture_accuracy[gesture].append(confidence)
            return gesture
        return None

    def get_stable_gesture(self, gesture):
        self.gesture_history.append(gesture)
        if len(self.gesture_history) == self.gesture_history.maxlen:
            if all(g == self.gesture_history[0] for g in self.gesture_history):
                return self.gesture_history[0]
        return None

    def draw_gesture(self, image, gesture):
        h, w, _ = image.shape
        gesture_area = np.zeros((200, 200, 3), dtype=np.uint8)
        gesture_area[:] = self.theme['background']
        if isinstance(gesture, int):
            cv2.putText(gesture_area, str(gesture), (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 5, self.theme['accent'], 5)
        elif gesture in ['+', '-', '*', '/', '^', '%', '.', '(', ')', '√']:
            cv2.putText(gesture_area, gesture, (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 5, self.theme['accent'], 5)
        elif gesture == 'C':
            cv2.putText(gesture_area, "CLR", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, self.theme['error'], 5)
        elif gesture == '=':
            cv2.putText(gesture_area, "=", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 5, self.theme['success'], 5)
        alpha = 0.9
        roi = image[h-220:h-20, w-220:w-20]
        image[h-220:h-20, w-220:w-20] = cv2.addWeighted(roi, 1-alpha, gesture_area, alpha, 0)
        return image

    def evaluate_expression(self, expr):
        try:
            expr = expr.replace('√', 'math.sqrt')
            result = eval(expr, {"__builtins__": None}, {"math": math})
            self.log_action("Calculation", f"Expression: {expr}, Result: {result}")
            return result
        except Exception as e:
            self.log_action("Error", f"Failed to evaluate: {expr}. Error: {str(e)}")
            return "Error"

    def draw_result(self, image, text, position, font_scale=1, thickness=2, 
                   text_color=None, bg_color=None):
        if text_color is None:
            text_color = self.theme['text']
        if bg_color is None:
            bg_color = self.theme['background']
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_offset_x, text_offset_y = position
        padding = 10
        box_coords = (
            (text_offset_x - padding, text_offset_y + padding),
            (text_offset_x + text_width + padding, text_offset_y - text_height - padding)
        )
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, text_color, thickness)
        return image

    def draw_interface(self, image):
        current_time = time.time()
        fps = 1 / (current_time - self.last_frame_time)
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        self.last_frame_time = current_time
        self.update_feedback_messages()
        image = self.draw_result(
            image, 
            f"Hand Gesture Calculator (FPS: {avg_fps:.1f})", 
            (10, 30), 
            0.7, 
            2, 
            self.theme['accent']
        )
        y_offset = 160
        for msg, color in self.feedback_messages:
            image = self.draw_result(
                image,
                msg,
                (10, y_offset),
                0.6,
                1,
                color
            )
            y_offset += 30
        if self.tutorial_mode:
            image = self.draw_result(
                image,
                f"Tutorial ({self.tutorial_step + 1}/{len(self.tutorial_steps)}): {self.tutorial_steps[self.tutorial_step]}",
                (10, image.shape[0] - 20),
                0.6,
                2,
                self.theme['highlight']
            )
        if not self.tutorial_mode:
            instructions = [
                "0-9: Show fingers",
                "+: Thumb, -: Pinky, *: Thumb+Index+Pinky",
                "/: Thumb+Pinky close, =: Index+Ring",
                "C: Index+Ring+Pinky (Clear)",
                "Hold number gesture for 5s to repeat",
                "Press 'T' for tutorial mode"
            ]
            for i, instruction in enumerate(instructions):
                image = self.draw_result(
                    image,
                    instruction,
                    (10, image.shape[0] - 20 - 25*i),
                    0.5,
                    1,
                    self.theme['text']
                )
        return image

    def handle_keyboard_input(self, key):
        if key == ord('t'):
            self.tutorial_mode = not self.tutorial_mode
            self.log_action("Tutorial mode", f"{'enabled' if self.tutorial_mode else 'disabled'}")
        elif key == ord('n') and self.tutorial_mode:
            self.tutorial_step = (self.tutorial_step + 1) % len(self.tutorial_steps)
        elif key == ord('p') and self.tutorial_mode:
            self.tutorial_step = (self.tutorial_step - 1) % len(self.tutorial_steps)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            image = self.draw_interface(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=self.theme['accent'], thickness=2),
                        self.mp_drawing.DrawingSpec(color=self.theme['text'], thickness=1)
                    )
                    finger_state = self.get_finger_state(hand_landmarks)
                    finger_angles = self.get_finger_angles(hand_landmarks)
                    gesture = self.interpret_gesture(finger_state, finger_angles)
                    stable_gesture = self.get_stable_gesture(gesture)
                    current_time = time.time()
                    if stable_gesture is not None:
                        if isinstance(stable_gesture, int):
                            if stable_gesture == self.last_number_gesture:
                                if current_time - self.last_number_time >= 5:
                                    self.current_number += str(stable_gesture)
                                    self.last_number_time = current_time
                                    self.add_feedback_message(
                                        f"Repeated number: {stable_gesture}",
                                        self.theme['accent']
                                    )
                            else:
                                self.last_number_gesture = stable_gesture
                                self.last_number_time = current_time
                                if stable_gesture != self.last_gesture:
                                    self.current_number += str(stable_gesture)
                                    self.add_feedback_message(
                                        f"Number input: {stable_gesture}",
                                        self.theme['accent']
                                    )
                        else:
                            self.last_number_gesture = None
                            if stable_gesture != self.last_gesture:
                                self.last_gesture = stable_gesture
                                if stable_gesture == 'C':
                                    self.expression = []
                                    self.current_number = ""
                                    self.add_feedback_message(
                                        "Expression cleared",
                                        self.theme['error']
                                    )
                                elif stable_gesture == '=':
                                    if self.current_number:
                                        self.expression.append(self.current_number)
                                        self.current_number = ""
                                    result = self.evaluate_expression(''.join(map(str, self.expression)))
                                    if result != "Error":
                                        self.add_feedback_message(
                                            f"Result: {result}",
                                            self.theme['success']
                                        )
                                    else:
                                        self.add_feedback_message(
                                            "Calculation error",
                                            self.theme['error']
                                        )
                                    self.expression = [str(result)]
                                else:
                                    if self.current_number:
                                        self.expression.append(self.current_number)
                                        self.current_number = ""
                                    self.expression.append(stable_gesture)
                                    self.add_feedback_message(
                                        f"Operator: {stable_gesture}",
                                        self.theme['highlight']
                                    )
                        if gesture is not None:
                            image = self.draw_gesture(image, gesture)
            display_text = ''.join(map(str, self.expression)) + self.current_number
            image = self.draw_result(
                image, 
                f"Expression: {display_text}", 
                (10, 70), 
                1, 
                2, 
                self.theme['text']
            )
            if len(self.expression) > 0 and isinstance(self.expression[-1], str) and self.expression[-1] == '=':
                result = self.evaluate_expression(''.join(map(str, self.expression[:-1])))
                result_text = f"Result: {result}"
                image = self.draw_result(
                    image, 
                    result_text, 
                    (10, 120), 
                    1.5, 
                    3, 
                    self.theme['success'] if result != "Error" else self.theme['error'],
                    (0, 0, 128)
                )
            if self.gesture_accuracy:
                accuracy_text = "Gesture Recognition Accuracy:"
                image = self.draw_result(
                    image,
                    accuracy_text,
                    (image.shape[1] - 300, 30),
                    0.6,
                    2,
                    self.theme['accent']
                )
                y_offset = 60
                for gesture, accuracies in self.gesture_accuracy.items():
                    if accuracies:
                        avg_accuracy = sum(accuracies) / len(accuracies) * 100
                        accuracy_text = f"{gesture}: {avg_accuracy:.1f}%"
                        image = self.draw_result(
                            image,
                            accuracy_text,
                            (image.shape[1] - 300, y_offset),
                            0.5,
                            1,
                            self.theme['text']
                        )
                        y_offset += 20
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            self.handle_keyboard_input(key)
            cv2.imshow(self.window_name, image)
        self.log_action("Session ended", "Application closed")
        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        calculator = HandGestureCalculator()
        print("Starting Hand Gesture Calculator...")
        print("Controls:")
        print("- Press 'T' to toggle tutorial mode")
        print("- Press 'N' to go to next tutorial step")
        print("- Press 'P' to go to previous tutorial step")
        print("- Press 'ESC' to exit")
        calculator.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        with open("error_log.txt", "a") as f:
            f.write(f"\n{datetime.now()}: {str(e)}")

if __name__ == "__main__":
    main()
