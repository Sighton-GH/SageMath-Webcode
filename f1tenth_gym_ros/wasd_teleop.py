# MIT License
#
# Copyright (c) 2026
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import select
import termios
import tty
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


@dataclass
class TeleopState:
    speed: float = 0.0
    steering: float = 0.0


def _get_key(timeout_sec: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if not ready:
        return None

    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch

    # Arrow keys: ESC [ A/B/C/D
    if select.select([sys.stdin], [], [], 0.0001)[0]:
        ch2 = sys.stdin.read(1)
        if ch2 == "[" and select.select([sys.stdin], [], [], 0.0001)[0]:
            ch3 = sys.stdin.read(1)
            return f"\x1b[{ch3}"
    return ch


class WasdTeleop(Node):
    def __init__(self) -> None:
        super().__init__("wasd_teleop")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("speed_step", 0.5)
        self.declare_parameter("steer_step", 0.1)
        self.declare_parameter("max_speed", 5.0)
        self.declare_parameter("max_steer", 0.4)
        self.declare_parameter("publish_rate_hz", 20.0)

        drive_topic = self.get_parameter("drive_topic").value
        self.speed_step = float(self.get_parameter("speed_step").value)
        self.steer_step = float(self.get_parameter("steer_step").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.max_steer = float(self.get_parameter("max_steer").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.state = TeleopState()

        self.get_logger().info("WASD/Arrow teleop ready.")
        self.get_logger().info("W/Up: forward, S/Down: reverse, A/Left: steer left, D/Right: steer right")
        self.get_logger().info("Space: stop, R: reset, Q/E: fine steer, Ctrl+C to quit")

    def apply_key(self, key: str) -> None:
        if key in ("w", "W", "\x1b[A"):
            self.state.speed += self.speed_step
        elif key in ("s", "S", "\x1b[B"):
            self.state.speed -= self.speed_step
        elif key in ("a", "A", "\x1b[D"):
            self.state.steering += self.steer_step
        elif key in ("d", "D", "\x1b[C"):
            self.state.steering -= self.steer_step
        elif key in ("q", "Q"):
            self.state.steering += self.steer_step * 0.5
        elif key in ("e", "E"):
            self.state.steering -= self.steer_step * 0.5
        elif key == " ":
            self.state.speed = 0.0
        elif key in ("r", "R"):
            self.state.speed = 0.0
            self.state.steering = 0.0

        self.state.speed = max(-self.max_speed, min(self.max_speed, self.state.speed))
        self.state.steering = max(-self.max_steer, min(self.max_steer, self.state.steering))

    def publish(self) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(self.state.speed)
        msg.drive.steering_angle = float(self.state.steering)
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = WasdTeleop()

    rate = 1.0 / node.publish_rate_hz if node.publish_rate_hz > 0 else 0.05
    old_attrs = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            key = _get_key(rate)
            if key:
                node.apply_key(key)
            node.publish()
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
