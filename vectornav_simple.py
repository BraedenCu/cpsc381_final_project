#!/usr/bin/env python3
import time
from pymavlink import mavutil
import serial.tools.list_ports

def find_pixhawk_port():
    """
    Auto-detect a Pixhawk-style USB serial port on macOS/Linux.
    Falls back to a hardcoded path if none found.
    """
    for port in serial.tools.list_ports.comports():
        # match common identifiers; adjust as needed
        if 'PX4' in port.description or 'Pixhawk' in port.description or 'usbmodem' in port.device:
            return port.device
    # fallback
    return '/dev/cu.usbmodem01'

def main():
    port = find_pixhawk_port()
    baud = 57600  # or 115200 if you’ve reconfigured
    print(f"Connecting to Pixhawk on {port} @ {baud} baud...")
    master = mavutil.mavlink_connection(port, baud=baud)
    
    # wait for the first heartbeat 
    print("Waiting for heartbeat from system...")
    master.wait_heartbeat()
    print(f"Heartbeat received from system (sys {master.target_system}, comp {master.target_component})\n")
    
    try:
        while True:
            # blocking read; returns a Message or None
            msg = master.recv_match(blocking=True, timeout=5)
            if msg is None:
                print("No message received in 5 seconds, still waiting...")
                continue
            # print the raw MAVLink msg type and its fields
            print(f"[{msg.get_type()}] {msg.to_dict()}")
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
    finally:
        master.close()
import serial, serial.tools.list_ports, subprocess, time

def open_port(device, baud=57600, retries=3):
    for attempt in range(retries):
        try:
            return serial.Serial(device, baud, timeout=1)
        except serial.SerialException as e:
            print(f"Attempt {attempt+1}: {e}")
            # Find and kill processes using the port
            procs = subprocess.check_output(
                ["lsof", "-t", device], text=True).split()
            for pid in procs:
                subprocess.run(["kill", "-9", pid])
            time.sleep(1)
    raise RuntimeError(f"Failed to open {device} after {retries} retries")

# Usage
if __name__ == "__main__":
    port = "/dev/cu.usbmodem01"
    ser = open_port(port)
    print(f"Opened {port}")
    main()
    ser.close()