#!/usr/bin/env python3
from controller import Supervisor, Keyboard
import math

# ===================== CONFIGURATION =====================
# Base directions at yaw = 0 (world coordinates)
FORWARD0 = [1, 0]    # at yaw=0, forward = +X
RIGHT0   = [0, -1]   # at yaw=0, right   = –Y (since left is +Y)

# Key codes
FORWARD_KEY  = ord('W')
BACKWARD_KEY = ord('S')
LEFT_KEY     = ord('A')
RIGHT_KEY    = ord('D')
TURN_LEFT_KEY  = ord('Q')
TURN_RIGHT_KEY = ord('E')
PITCH_UP_KEY   = ord('R')
PITCH_DOWN_KEY = ord('F')
UP_KEY        = ord(' ')    # Space
DOWN_KEY      = ord('C')

# Direction signs (1 or -1) – adjust if a key is reversed
FORWARD_SIGN =  1
RIGHT_SIGN   =  1
TURN_SIGN    =  1
PITCH_SIGN   =  1   # set to -1 if R pitches down and F up
VERTICAL_SIGN=  1
# ==========================================================

supervisor = Supervisor()
time_step = int(supervisor.getBasicTimeStep())

# --- Camera device (optional, for image) ---
camera = supervisor.getDevice("CAMERA")
if camera is None:
    print("ERROR: Camera device not found!")
    exit(1)
camera.enable(time_step)

# --- Camera pitch transform node ---
pitch_node = supervisor.getFromDef("CAMERA_PITCH")
if pitch_node is None:
    print("ERROR: CAMERA_PITCH node not found! Check your world file.")
    exit(1)
pitch_rot_field = pitch_node.getField("rotation")

# --- Keyboard ---
keyboard = supervisor.getKeyboard()
keyboard.enable(time_step)

# --- Robot node ---
robot_node = supervisor.getFromDef("MainCameraController")
if robot_node is None:
    print("ERROR: MainCameraController node not found!")
    exit(1)

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# --- Movement parameters ---
move_speed = 0.1
rot_speed = 0.05
pitch = 0.0

while supervisor.step(time_step) != -1:
    pos = translation_field.getSFVec3f()
    rot = rotation_field.getSFRotation()
    yaw = rot[3]  # rotation angle around Z

    forward_input = 0
    right_input   = 0
    pitch_input   = 0

    key = keyboard.getKey()
    while key != -1:
        old_f, old_r, old_p = forward_input, right_input, pitch_input

        if key == FORWARD_KEY:
            forward_input += FORWARD_SIGN
        elif key == BACKWARD_KEY:
            forward_input -= FORWARD_SIGN
        elif key == LEFT_KEY:
            right_input -= RIGHT_SIGN
        elif key == RIGHT_KEY:
            right_input += RIGHT_SIGN
        elif key == TURN_LEFT_KEY:
            yaw += rot_speed * TURN_SIGN
        elif key == TURN_RIGHT_KEY:
            yaw -= rot_speed * TURN_SIGN
        elif key == PITCH_UP_KEY:
            pitch_input -= PITCH_SIGN
        elif key == PITCH_DOWN_KEY:
            pitch_input += PITCH_SIGN
        elif key == UP_KEY:
            pos[2] += move_speed * VERTICAL_SIGN
        elif key == DOWN_KEY or key == ord('c') or key == ord('C'):
            pos[2] -= move_speed * VERTICAL_SIGN

        # Debug: show when inputs change
        # if (forward_input != old_f or right_input != old_r or pitch_input != old_p):
        #     key_name = chr(key) if 32 <= key < 127 else f"0x{key:02X}"
        #     print(f"Key {key_name}: fwd={forward_input}, right={right_input}, pitch={pitch_input}")

        key = keyboard.getKey()

    # Update pitch
    if pitch_input != 0:
        pitch += pitch_input * rot_speed
        pitch_rot_field.setSFRotation([0, 1, 0, pitch])

    # --- Robot movement (local X/Y) ---
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward_vec = [
        FORWARD0[0] * cos_yaw - FORWARD0[1] * sin_yaw,
        FORWARD0[0] * sin_yaw + FORWARD0[1] * cos_yaw, 0]
    right_vec = [
        RIGHT0[0] * cos_yaw - RIGHT0[1] * sin_yaw,
        RIGHT0[0] * sin_yaw + RIGHT0[1] * cos_yaw, 0]

    pos[0] += (forward_input * forward_vec[0] + right_input * right_vec[0]) * move_speed
    pos[1] += (forward_input * forward_vec[1] + right_input * right_vec[1]) * move_speed

    # Apply robot yaw
    rotation_field.setSFRotation([0, 0, 1, yaw])
    translation_field.setSFVec3f(pos)