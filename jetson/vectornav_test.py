import serial
from collections import deque
import time

class VectorNavCollector:
    def __init__(self, port='/dev/ttyTHS1', baud=115200, timeout=1):
        # On Jetson Nano, UART1 is on J41 pins 8 (TXD) & 10 (RXD), exposed as /dev/ttyTHS1  [oai_citation:10‡JetsonHacks](https://jetsonhacks.com/2019/10/10/jetson-nano-uart/?utm_source=chatgpt.com)
        self.ser = serial.Serial(port, baud, timeout=timeout)
        # Dispatch table: maps header → parse function
        self.parsers = {
            'VNYMR': self._parse_vnymr,
            'VNMAG': self._parse_vnmag,
            'VNACC': self._parse_vnacc,
            'VNGYR': self._parse_vngyr,
            # extend for VNQTN, VNLLA, VNVEL, etc.
        }
        # Store the latest values in descriptive variables
        self.latest = {}

    def _readline(self):
        # Read until '\n'; returns raw bytes
        return self.ser.readline()

    def _validate_checksum(self, line: bytes) -> bool:
        # Strip leading '$' and split at '*'
        try:
            body, chk = line[1:].split(b'*')
        except ValueError:
            return False
        calc = 0
        for b in body:
            calc ^= b
        # Compare to integer value of hex checksum
        return calc == int(chk[:2], 16)

    def _parse_generic(self, fields, names):
        # Convert each field to float and map to descriptive name
        return {name: float(val) for name, val in zip(names, fields)}

    def _parse_vnymr(self, fields):
        names = [
            'yaw_deg','pitch_deg','roll_deg',
            'mag_x_gauss','mag_y_gauss','mag_z_gauss',
            'accel_x_m_s2','accel_y_m_s2','accel_z_m_s2',
            'gyro_x_rad_s','gyro_y_rad_s','gyro_z_rad_s'
        ]
        return self._parse_generic(fields, names)

    def _parse_vnmag(self, fields):
        names = ['mag_x_gauss','mag_y_gauss','mag_z_gauss']
        return self._parse_generic(fields, names)

    def _parse_vnacc(self, fields):
        names = ['accel_x_m_s2','accel_y_m_s2','accel_z_m_s2']
        return self._parse_generic(fields, names)

    def _parse_vngyr(self, fields):
        names = ['gyro_x_rad_s','gyro_y_rad_s','gyro_z_rad_s']
        return self._parse_generic(fields, names)

    def run(self):
        while True:
            raw = self._readline()
            if not raw or not raw.startswith(b'$'):
                continue
            if not self._validate_checksum(raw):
                continue  # drop bad packets  [oai_citation:11‡QSO](https://www.qso.com.ar/datasheets/Receptores%20GNSS-GPS/NMEA_Format_v0.1.pdf?utm_source=chatgpt.com)
            # Strip off '$' and checksum, then split on commas
            body = raw.strip()[1:raw.find(b'*')]
            parts = body.split(b',')
            header = parts[0].decode('ascii')
            vals = [p.decode('ascii') for p in parts[1:]]
            if header in self.parsers:
                parsed = self.parsers[header](vals)
                self.latest.update(parsed)
                # e.g. self.latest['yaw_deg'], self.latest['accel_z_m_s2'], etc.
                # These descriptive variables can now feed directly into your dashboard

#collector = VectorNavCollector()
#collector.run()

def test_vectornav_collection(duration_sec: float = 10.0, port: str = '/dev/ttyTHS1', baud: int = 115200):
    """
    Runs the VectorNavCollector for a fixed duration and prints every new data update.

    Args:
        duration_sec (float): how many seconds to run the test loop
        port (str): serial port device for VectorNav
        baud (int): baud rate for serial communications
    """
    collector = VectorNavCollector(port=port, baud=baud, timeout=1)
    start = time.time()
    
    print(f"Starting VectorNav test for {duration_sec} seconds on {port} @ {baud} baud...\n")
    try:
        # Keep looping until time’s up
        while time.time() - start < duration_sec:
            raw = collector._readline()
            if not raw or not raw.startswith(b'$'):
                continue
            if not collector._validate_checksum(raw):
                continue
            body = raw.strip()[1:raw.find(b'*')]
            parts = body.split(b',')
            header = parts[0].decode('ascii')
            vals = [p.decode('ascii') for p in parts[1:]]
            if header in collector.parsers:
                parsed = collector.parsers[header](vals)
                # Merge into latest
                collector.latest.update(parsed)
                # Print every field, sorted by name
                for name in sorted(parsed):
                    print(f"{name:20s}: {parsed[name]:>8.4f}")
                print('-' * 40)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        collector.ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    test_vectornav_collection(duration_sec=10.0, port='/dev/ttyTHS1', baud=115200)
    # test_vectornav_collection(duration_sec=10.0, port='/dev/ttyUSB0', baud=115200)