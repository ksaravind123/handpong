"""
hand_pong.py
Pong-like game controlled by webcam hand tracking (MediaPipe + OpenCV + Pygame).

Controls:
 - Move your index fingertip left/right to move the paddle.
 - Pinch (thumb tip near index tip) to serve / restart the ball.
"""

import sys
import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame

# --------------------------
# Configuration
# --------------------------
CAMERA_INDEX = 0
SCREEN_W, SCREEN_H = 960, 640  # game window size
PADDLE_W, PADDLE_H = 160, 16
BALL_R = 10
FPS = 60

# Hand smoothing: keep last N x-values and average
SMOOTHING_WINDOW = 5

# Pinch threshold (normalized distance between thumb tip and index tip)
PINCH_DIST_THRESHOLD = 0.05

# --------------------------
# MediaPipe hand helper
# --------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandTracker:
    def __init__(self, max_num_hands=1, detection_conf=0.6, tracking_conf=0.6):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=max_num_hands,
                                    min_detection_confidence=detection_conf,
                                    min_tracking_confidence=tracking_conf)
    def find_hands(self, frame):
        # frame in BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

# --------------------------
# Game objects
# --------------------------
class Paddle:
    def __init__(self, width, height, screen_w, screen_h, y_offset=40):
        self.width = width
        self.height = height
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x = (screen_w - width) // 2
        self.y = screen_h - y_offset
        self.color = (30, 200, 120)

    def set_center_x(self, cx):
        # cx is center x in pixels
        self.x = int(cx - self.width / 2)
        # clamp
        self.x = max(0, min(self.x, self.screen_w - self.width))

    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, surf):
        pygame.draw.rect(surf, self.color, self.rect(), border_radius=8)

class Ball:
    def __init__(self, x, y, r=10):
        self.x = x
        self.y = y
        self.r = r
        self.vx = 0
        self.vy = 0
        self.color = (240, 80, 50)
        self.stuck = True  # stuck to paddle initially

    def update(self, dt):
        if not self.stuck:
            self.x += self.vx * dt
            self.y += self.vy * dt

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), self.r)

# --------------------------
# Utilities
# --------------------------
def normalized_to_pixel(norm_x, norm_y, w, h):
    """Normalized (0..1) to pixel coords."""
    return int(norm_x * w), int(norm_y * h)

def landmark_pos(landmark, image_w, image_h):
    return normalized_to_pixel(landmark.x, landmark.y, image_w, image_h)

# --------------------------
# Main game
# --------------------------
def main():
    # Init OpenCV camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Init Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Hand-Pong (camera control)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    # Tracker
    tracker = HandTracker(max_num_hands=1)
    smooth_buffer = deque(maxlen=SMOOTHING_WINDOW)

    # Game objects
    paddle = Paddle(PADDLE_W, PADDLE_H, SCREEN_W, SCREEN_H, y_offset=40)
    ball = Ball(SCREEN_W//2, paddle.y - BALL_R - 4, r=BALL_R)
    lives = 3
    score = 0

    last_time = time.time()
    running = True

    # Helper: serve ball with initial velocity
    def serve_ball():
        ball.stuck = False
        # base speed -> adjust
        ball.vx = 300 * (1 if np.random.rand() > 0.5 else -1)
        ball.vy = -380

    # Main loop
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds per frame

        # --- handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        frame_h, frame_w = frame.shape[:2]

        # Flip horizontally so it's mirror-like
        frame = cv2.flip(frame, 1)

        results = tracker.find_hands(frame)

        # default: keep paddle where it is
        hand_x_pixel = None
        pinch = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            # index fingertip is landmark 8; thumb tip is 4
            idx = hand.landmark[8]  # index tip normalized
            thumb = hand.landmark[4]
            ix_pix, iy_pix = landmark_pos(idx, frame_w, frame_h)
            tx_pix, ty_pix = landmark_pos(thumb, frame_w, frame_h)

            # compute normalized x (0..1) using camera frame width and map to game screen
            norm_x = idx.x  # (0..1)
            # convert to game pixel x
            hand_x_pixel = norm_x * SCREEN_W

            # smoothing
            smooth_buffer.append(hand_x_pixel)
            smooth_x = sum(smooth_buffer) / len(smooth_buffer)

            # pinch detection: normalized euclidean distance between thumb tip and index tip
            dx = idx.x - thumb.x
            dy = idx.y - thumb.y
            d = math.hypot(dx, dy)
            pinch = d < PINCH_DIST_THRESHOLD

            # Draw landmarks on the OpenCV preview window (small)
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Optionally show debug circles
            cv2.circle(frame, (ix_pix, iy_pix), 6, (0,255,0), -1)
            cv2.circle(frame, (tx_pix, ty_pix), 6, (0,0,255), -1)
            # cv2.putText(frame, f"pinch={pinch}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        else:
            smooth_buffer.clear()

        # Set paddle from smoothed x if available
        if hand_x_pixel is not None and len(smooth_buffer) > 0:
            paddle.set_center_x(smooth_x)

        # If ball is stuck to paddle, update its position
        if ball.stuck:
            ball.x = paddle.x + paddle.width/2
            ball.y = paddle.y - ball.r - 2
            # If user pinches, serve
            if pinch:
                serve_ball()
                # brief delay to avoid immediate re-trigger
                time.sleep(0.15)
        else:
            # Update ball
            ball.update(dt)

            # Collisions with walls
            if ball.x - ball.r <= 0:
                ball.x = ball.r
                ball.vx *= -1
            if ball.x + ball.r >= SCREEN_W:
                ball.x = SCREEN_W - ball.r
                ball.vx *= -1
            if ball.y - ball.r <= 0:
                ball.y = ball.r
                ball.vy *= -1

            # Collision with paddle
            if ball.y + ball.r >= paddle.y:
                if paddle.x <= ball.x <= paddle.x + paddle.width:
                    # reflect based on where it hits the paddle
                    rel = (ball.x - (paddle.x + paddle.width/2)) / (paddle.width/2)
                    bounce_angle = rel * (math.pi/3)  # -60deg..60deg
                    speed = math.hypot(ball.vx, ball.vy) * 1.02  # slight speed-up
                    ball.vx = speed * math.sin(bounce_angle)
                    ball.vy = -abs(speed * math.cos(bounce_angle))
                    # make sure ball is above paddle
                    ball.y = paddle.y - ball.r - 1
                    score += 1

            # bottom miss -> lose life
            if ball.y - ball.r > SCREEN_H:
                lives -= 1
                ball.stuck = True
                ball.vx = ball.vy = 0
                if lives <= 0:
                    # reset game
                    lives = 3
                    score = 0
                    # keep ball stuck

        # --- draw everything in pygame
        screen.fill((18, 24, 44))  # dark bg

        # draw game
        paddle.draw(screen)
        ball.draw(screen)

        # HUD
        hud = font.render(f"Score: {score}    Lives: {lives}", True, (220,220,220))
        screen.blit(hud, (10, 10))
        hint = font.render("Move your index finger left/right to move paddle. Pinch to serve.", True, (160,160,160))
        screen.blit(hint, (10, 40))

        # Draw debug small webcam preview in top-right
        # convert OpenCV BGR to RGB then to surface
        small = cv2.resize(frame, (240, 180))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        small_surf = pygame.surfarray.make_surface(np.rot90(small_rgb))
        sw = small_surf.get_width()
        sh = small_surf.get_height()
        screen.blit(small_surf, (SCREEN_W - sw - 12, 12))

        pygame.display.flip()

    # cleanup
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
