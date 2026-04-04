import pygame
import requests
import threading
import time

PI_URL = "http://192.168.2.2:8084"

# Channel mapping
MOTOR_1    = 0
SOLENOID_1 = 1
MOTOR_2    = 2
SOLENOID_2 = 3

def set_speed(channel, percent):
    try:
        requests.post(f"{PI_URL}/motor/{channel}/speed",
                      json={"percent": percent}, timeout=2)
    except Exception as e:
        print(f"Error: {e}")

def stop_all():
    for ch in [MOTOR_1, SOLENOID_1, MOTOR_2, SOLENOID_2]:
        set_speed(ch, 0)
    print("Emergency stop — all channels off, valves released")

def pulse_pair(motor_ch, solenoid_ch, duration=1.0):
    """Motor + solenoid ON together, both OFF after duration."""
    set_speed(motor_ch, 100)
    set_speed(solenoid_ch, 100)
    time.sleep(duration)
    set_speed(motor_ch, 0)
    set_speed(solenoid_ch, 0)

def pulse_both(duration=1.0):
    """Both pairs ON together, all OFF after duration."""
    for ch in [MOTOR_1, SOLENOID_1, MOTOR_2, SOLENOID_2]:
        set_speed(ch, 100)
    time.sleep(duration)
    for ch in [MOTOR_1, SOLENOID_1, MOTOR_2, SOLENOID_2]:
        set_speed(ch, 0)
    print("Both pairs done")

def trap_air(motor_ch, solenoid_ch, duration=1.0):
    """Motor ON for duration then OFF, solenoid stays ON to trap air."""
    set_speed(motor_ch, 100)
    set_speed(solenoid_ch, 100)
    time.sleep(duration)
    set_speed(motor_ch, 0)
    print(f"Motor ch{motor_ch} OFF — air trapped, solenoid ch{solenoid_ch} holding")

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller found. Connect PS4 controller and try again.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Connected: {joystick.get_name()}")
print("L1 → Pair 1 pulse 0.7s    | R1 → Pair 2 pulse 0.7s")
print("X  → Both pairs 1s        | O  → Emergency stop + reset lock")
print("Square → Pair 1 pulse 1s  | Triangle → Pair 2 pulse 1s")
print("L2 → Trap air pair 1 1s   | R2 → Trap air pair 2 1s (one use until O)")

X        = 0
O        = 1
SQUARE   = 2
TRIANGLE = 3
L1       = 9
R1       = 10
L2_AXIS  = 4
R2_AXIS  = 5
AXIS_THRESHOLD = 0.5

active = {
    "l1": False,
    "r1": False,
    "x": False,
    "sq": False,
    "tri": False,
    "l2": False,
    "r2": False,
}

l2_locked = False
r2_locked = False
l2_prev = False
r2_prev = False

while True:
    pygame.event.pump()

    l2_pressed = joystick.get_axis(L2_AXIS) > AXIS_THRESHOLD
    r2_pressed = joystick.get_axis(R2_AXIS) > AXIS_THRESHOLD

    # L1 -> Pair 1 pulse 0.7s
    if joystick.get_button(L1) and not active["l1"]:
        active["l1"] = True
        def run_l1():
            pulse_pair(MOTOR_1, SOLENOID_1, 0.7)
            active["l1"] = False
        threading.Thread(target=run_l1, daemon=True).start()

    # R1 -> Pair 2 pulse 0.7s
    if joystick.get_button(R1) and not active["r1"]:
        active["r1"] = True
        def run_r1():
            pulse_pair(MOTOR_2, SOLENOID_2, 0.7)
            active["r1"] = False
        threading.Thread(target=run_r1, daemon=True).start()

    # X -> Both pairs ON for 1s, all off
    if joystick.get_button(X) and not active["x"]:
        active["x"] = True
        def run_x():
            pulse_both(1.0)
            active["x"] = False
        threading.Thread(target=run_x, daemon=True).start()

    # Square -> Pair 1 pulse 1s
    if joystick.get_button(SQUARE) and not active["sq"]:
        active["sq"] = True
        def run_sq():
            pulse_pair(MOTOR_1, SOLENOID_1, 1.0)
            active["sq"] = False
        threading.Thread(target=run_sq, daemon=True).start()

    # Triangle -> Pair 2 pulse 1s
    if joystick.get_button(TRIANGLE) and not active["tri"]:
        active["tri"] = True
        def run_tri():
            pulse_pair(MOTOR_2, SOLENOID_2, 1.0)
            active["tri"] = False
        threading.Thread(target=run_tri, daemon=True).start()

    # L2 -> Trap air pair 1 (one use only until O pressed)
    if l2_pressed and not l2_prev and not active["l2"] and not l2_locked:
        l2_locked = True
        active["l2"] = True
        def run_l2():
            trap_air(MOTOR_1, SOLENOID_1, 1.0)
            active["l2"] = False
            print("Pair 1 air trapped — press O to release")
        threading.Thread(target=run_l2, daemon=True).start()
    elif l2_pressed and l2_locked:
        print("Pair 1 locked — press O to release valve first")

    # R2 -> Trap air pair 2 (one use only until O pressed)
    if r2_pressed and not r2_prev and not active["r2"] and not r2_locked:
        r2_locked = True
        active["r2"] = True
        def run_r2():
            trap_air(MOTOR_2, SOLENOID_2, 1.0)
            active["r2"] = False
            print("Pair 2 air trapped — press O to release")
        threading.Thread(target=run_r2, daemon=True).start()
    elif r2_pressed and r2_locked:
        print("Pair 2 locked — press O to release valve first")

    # O -> Emergency stop, release all valves, reset locks
    if joystick.get_button(O):
        stop_all()
        for key in active:
            active[key] = False
        l2_locked = False
        r2_locked = False
        print("Locks reset — L2 and R2 available again")

    l2_prev = l2_pressed
    r2_prev = r2_pressed

    time.sleep(0.05)