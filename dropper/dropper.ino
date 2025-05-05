#include <Servo.h>

Servo dropper;

// ——— CONFIG ———
const int SERVO_PIN    = 9;     // PWM pin
const int CLOSED_POS   =  90;   // [deg] home position
const int DROP_OFFSET  =  -55;  // [deg] how far to swing open
const unsigned long MOVE_DELAY = 300;  // [ms] pause at each end

void setup() {
  Serial.begin(115200);
  delay(100);              // allow Serial to initialize

  // initialize servo to closed position
  dropper.attach(SERVO_PIN);
  dropper.write(CLOSED_POS);
  delay(500);
  dropper.detach();
}

void loop() {
  // read full line if available
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();  // remove any \r or whitespace

    if (cmd.equalsIgnoreCase("DROP")) {
      doDrop();
      Serial.println("OK");  // acknowledge
    }
    // else ignore unknown commands
  }
}

// ——— DROP SEQUENCE ———
void doDrop() {
  int dropAngle = CLOSED_POS + DROP_OFFSET;

  dropper.attach(SERVO_PIN);

  // 1) swing open
  dropper.write(dropAngle);
  delay(MOVE_DELAY);

  // 2) return to home
  dropper.write(CLOSED_POS);
  delay(MOVE_DELAY);

  // 3) detach so it holds position
  dropper.detach();
}
