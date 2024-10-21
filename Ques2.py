import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Game settings
width, height = 1280, 640
player_pos = [320, 440]
enemy_speed = 5
enemy_size = 50
enemy_list = []
score = 0

# Create random enemy
def create_enemy():
    x_pos = random.randint(0, width - enemy_size)
    return [x_pos, 0]

# Move enemies down 
def move_enemies(enemy_list):
    global score  # Use the global score variable
    new_enemy_list = []
    for enemy in enemy_list:
        enemy[1] += enemy_speed
        if enemy[1] < height:
            new_enemy_list.append(enemy)
        else:
            score += 1  # Increment score for dodged enemies
    return new_enemy_list

# Check for collisions
def check_collision(player_pos, enemy_list):
    player_rect = (player_pos[0], player_pos[1], enemy_size, enemy_size)
    for enemy in enemy_list:
        enemy_rect = (enemy[0], enemy[1], enemy_size, enemy_size)
        if (player_rect[0] < enemy_rect[0] + enemy_rect[2] and
            player_rect[0] + player_rect[2] > enemy_rect[0] and
            player_rect[1] < enemy_rect[1] + enemy_rect[3] and
            player_rect[1] + player_rect[3] > enemy_rect[1]):
            return True
    return False

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    # Get coordinates of the index finger tip (landmark 8)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
        finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

        # Move player based on hand movement
        player_pos[0] = finger_tip_x - enemy_size // 2
        player_pos[0] = max(0, min(player_pos[0], width - enemy_size))  # Keep player within bounds

    # Add new enemies
    if random.randint(1, 20) == 1:  # Adjust frequency of enemy creation
        enemy_list.append(create_enemy())

    # Move enemies and update score
    enemy_list = move_enemies(enemy_list)

    # Check for collision
    if check_collision(player_pos, enemy_list):
        print("Collision! Game Over!")
        break

    # Draw game elements
    frame = cv2.rectangle(frame, (player_pos[0], player_pos[1]), (player_pos[0] + enemy_size, player_pos[1] + enemy_size), (0, 255, 0), -1)  # Player
    for enemy in enemy_list:
        frame = cv2.rectangle(frame, (enemy[0], enemy[1]), (enemy[0] + enemy_size, enemy[1] + enemy_size), (255, 0, 0), -1)  # Enemy

    # Display score on the frame
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Dodging Game using hands", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
