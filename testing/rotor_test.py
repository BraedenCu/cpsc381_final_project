#!/usr/bin/env python3

import time
from pymavlink import mavutil

# ---- Configuration ----
PIXHAWK_PORT = '/dev/cu.usbmodem01'  # adjust as needed
PIXHAWK_BAUD = 57600

# 1. Establish connection
master = mavutil.mavlink_connection(PIXHAWK_PORT, baud=PIXHAWK_BAUD)  # import and connect to Pixhawk :contentReference[oaicite:5]{index=5}  
master.wait_heartbeat()                                               # wait for heartbeat (≥1 Hz) :contentReference[oaicite:6]{index=6}  
print("Heartbeat received from system (system %u component %u)" %
      (master.target_system, master.target_component))

# 2. Arm the vehicle
# Option A: convenience helper (ArduPilot-specific)
master.arducopter_arm()                                                # arm via helper :contentReference[oaicite:7]{index=7}  
master.motors_armed_wait()                                             # wait until armed :contentReference[oaicite:8]{index=8}  
print("Armed!")

# 3. Set GUIDED mode
mode = 'GUIDED'
mode_id = master.mode_mapping()[mode]                                  # lookup mode ID :contentReference[oaicite:9]{index=9}  
master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    mode_id
)                                                                      # send SET_MODE :contentReference[oaicite:10]{index=10}  
# wait for ACK 
while True:
    ack = master.recv_match(type='COMMAND_ACK', blocking=True).to_dict()
    if ack['command'] == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
        print("Mode change ACK:", ack['result'])
        break

# 4. RC override: set overall throttle (channel 3) to mid-stick (1500µs)
def set_throttle(pwm=1500):
    """Override all 8 RC channels; set channel 3 (throttle) to pwm."""
    rc_vals = [65535]*18                                             # default no-override for 18 channels :contentReference[oaicite:11]{index=11}  
    rc_vals[2] = pwm                                                 # channel index 2 => RC3 :contentReference[oaicite:12]{index=12}  
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        *rc_vals
    )                                                                 # send RC override :contentReference[oaicite:13]{index=13}  

print("Setting throttle to 1600µs")
set_throttle(1600)
time.sleep(5)

print("Returning throttle to neutral (1500µs)")
set_throttle(1500)
time.sleep(2)

# 5. Motor test: spin motor #1 at 50% throttle for 3 s
print("Running motor test on motor 1")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,  # motor test command :contentReference[oaicite:14]{index=14}  
    0,                                      # confirmation
    1,                                      # param1: motor instance #1 :contentReference[oaicite:15]{index=15}  
    0,                                      # param2: throttle type (0=percent) :contentReference[oaicite:16]{index=16}  
    50,                                     # param3: throttle % (0–100) :contentReference[oaicite:17]{index=17}  
    3,                                      # param4: timeout seconds :contentReference[oaicite:18]{index=18}  
    0,                                      # param5: motor count (0=1) :contentReference[oaicite:19]{index=19}  
    0, 0                                    # param6–7 empty :contentReference[oaicite:20]{index=20}  
)
time.sleep(4)

# 6. Disarm
print("Disarming")
master.arducopter_disarm()                                             # disarm helper :contentReference[oaicite:21]{index=21}  
master.motors_disarmed_wait()                                          # wait until disarmed :contentReference[oaicite:22]{index=22}  
print("Disarmed")
