#!/usr/bin/env python3
import time
from pymavlink import mavutil
import serial.tools.list_ports

def find_pixhawk_port():
    # Auto-detect Pixhawk USB port
    for port in serial.tools.list_ports.comports():
        if 'PX4' in port.description or 'Pixhawk' in port.description or 'usbmodem' in port.device:
            return port.device
    return '/dev/cu.usbmodem01'

def main():
    port = find_pixhawk_port()
    baud = 57600
    print(f"Connecting to Pixhawk on {port}@{baud}...")
    master = mavutil.mavlink_connection(port, baud=baud)  # open link  [oai_citation:7‡MAVLink](https://mavlink.io/en/mavgen_python/?utm_source=chatgpt.com)

    print("Waiting for heartbeat...")
    master.wait_heartbeat()  # sets target_system & target_component  [oai_citation:8‡MAVLink](https://mavlink.io/en/mavgen_python/?utm_source=chatgpt.com)
    print(f"Heartbeat from sys {master.target_system}, comp {master.target_component}\n")

    # ▶️ Request common data streams
    streams = [
        ('ALL',          mavutil.mavlink.MAV_DATA_STREAM_ALL,             2),
        ('RAW_SENSORS',  mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,     5),
        ('EXT_STATUS',   mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 5),
        ('RC_CHANNELS',  mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,     10),
        ('POSITION',     mavutil.mavlink.MAV_DATA_STREAM_POSITION,        10),
        ('EXTRA1',       mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,          2),
        ('EXTRA2',       mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,          2),
        ('EXTRA3',       mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,          2),
    ]
    for name, stream_id, rate in streams:
        print(f"Requesting stream {name} @ {rate} Hz")
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            stream_id,
            rate,
            1  # start
        )  # group‐stream request  [oai_citation:9‡ENSTA Bretagne](https://www.ensta-bretagne.fr/lebars/Share/nogps_takeoff_land_pymavlink.py)

    # ▶️ (Optional) Fine-tune individual messages
    # master.mav.command_long_send(
    #    master.target_system, master.target_component,
    #    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    #    0, MAV_LINK_MSG_ID_GLOBAL_POSITION_INT, 100000, 0,0,0,0,0
    # )  # 10 Hz position  [oai_citation:10‡ArduPilot.org](https://ardupilot.org/dev/docs/mavlink-requesting-data.html)

    try:
        while True:
            msg = master.recv_match(blocking=True, timeout=5)
            if msg is None:
                print("⏳ No msg in 5 s, waiting...")
                continue
            print(f"[{msg.get_type()}] {msg.to_dict()}")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    finally:
        master.close()

if __name__ == "__main__":
    main()