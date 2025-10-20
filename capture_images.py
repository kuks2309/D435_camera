#!/usr/bin/env python3
"""
Intel RealSense D435 Camera Image Capture Script
This script captures RGB images from the D435 camera.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime


class D435Camera:
    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize RealSense D435 camera

        Args:
            width: Image width (default: 640)
            height: Image height (default: 480)
            fps: Frames per second (default: 30)
        """
        self.width = width
        self.height = height
        self.fps = fps

        # Create pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    def start(self):
        """Start the camera pipeline"""
        print("Starting RealSense D435 camera...")
        profile = self.pipeline.start(self.config)

        # Get device information
        device = profile.get_device()
        self.serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"Camera Serial Number: {self.serial_number}")

        # Wait for auto-exposure to stabilize
        print("Waiting for camera to stabilize...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("Camera ready!")

    def stop(self):
        """Stop the camera pipeline"""
        self.pipeline.stop()
        print("Camera stopped.")

    def get_frames(self):
        """
        Capture and return RGB frame

        Returns:
            color_image: numpy array of RGB image
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        # Get color frame
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        return color_image

    def get_intrinsics(self):
        """Get camera intrinsics"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        return {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy,
            'model': intrinsics.model,
            'coeffs': intrinsics.coeffs
        }

    def get_device_info(self):
        """Get camera device information"""
        return {
            'serial_number': self.serial_number
        }


def save_images(color_image, base_dir='captured_images'):
    """
    Save captured images to disk

    Args:
        color_image: RGB image
        base_dir: Base directory path
    """
    # Get current date and time
    now = datetime.now()
    date_folder = now.strftime("%Y%m%d")
    time_filename = now.strftime("%H%M%S_%f")[:-3]  # milliseconds precision

    # Create date-based folder
    output_dir = os.path.join(base_dir, date_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Save color image with time-based filename
    color_path = os.path.join(output_dir, f'color_{time_filename}.png')
    cv2.imwrite(color_path, color_image)

    print(f"Saved: {color_path}")

    return color_path


def main():
    """Main function for interactive image capture"""
    # Create camera instance
    camera = D435Camera(width=640, height=480, fps=30)

    try:
        # Start camera
        camera.start()

        # Print camera intrinsics
        intrinsics = camera.get_intrinsics()
        print("\nCamera Intrinsics:")
        print(f"  Resolution: {intrinsics['width']}x{intrinsics['height']}")
        print(f"  Focal Length: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
        print(f"  Principal Point: cx={intrinsics['ppx']:.2f}, cy={intrinsics['ppy']:.2f}")

        print("\nControls:")
        print("  Press 'c' or SPACE to capture image")
        print("  Press 'q' or ESC to quit")

        capture_count = 0

        while True:
            # Get frames
            color_image = camera.get_frames()

            if color_image is None:
                continue

            # Add text overlay
            display = color_image.copy()
            cv2.putText(display, f"Captured: {capture_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press 'c' to capture, 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Display
            cv2.imshow('RealSense D435 - Color', display)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF

            # Capture image
            if key == ord('c') or key == ord(' '):
                save_images(color_image)
                capture_count += 1

            # Quit
            elif key == ord('q') or key == 27:  # 27 is ESC
                break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print(f"\nTotal images captured: {capture_count}")


if __name__ == "__main__":
    main()