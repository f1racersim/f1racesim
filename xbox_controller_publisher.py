#!/usr/bin/env python3
import sys
import os
import struct

# Add the ros3 Python module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ros3', 'lang', 'python'))
import ros3 as rose

def main():
    # Initialize the ROS3 node
    node = rose.Node("xbox_controller_publisher")
    publisher = node.publisher("/xbox/controller", message_size=512, rate=60)

    device_path = "/dev/input/js0"

    if not os.path.exists(device_path):
        print(f"Error: {device_path} not found.")
        return

    # Pre-allocate flat lists with default values
    # axes: [LX, LY, LT, RX, RY, RT, DpadX, DpadY]
    # buttons: [A, B, X, Y, LB, RB, Back, Start, Guide, LS, RS]
    axes = [0.0] * 8
    buttons = [0] * 11

    print(f"Streaming flat lists from {device_path}...")

    try:
        with open(device_path, 'rb') as jsdev:
            while node.ok():
                # Read 8 bytes: [4:time, 2:value, 1:type, 1:number]
                evbuf = jsdev.read(8)
                if not evbuf:
                    break

                _, value, type_code, number = struct.unpack('IhBB', evbuf)

                # Strip the initial state bit (0x80)
                event_type = type_code & ~0x80 

                if event_type == 0x01: # Button
                    if number < len(buttons):
                        buttons[number] = value
                
                elif event_type == 0x02: # Axis
                    if number < len(axes):
                        # Normalize to -1.0 to 1.0 (32767 is max for __s16)
                        axes[number] = round(value / 32767.0, 4)

                # Publish as a combined list or a nested structure
                # Here we send a single list: [axes..., buttons...]
                payload = axes + buttons
                print(payload)
                publisher.publish(payload)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        node.shutdown()

if __name__ == "__main__":
    main()