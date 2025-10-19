#!/usr/bin/env python3
"""
Intel RealSense D435 Camera Image Capture Script
This script captures RGB and Depth images from the D435 camera.
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
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

    def start(self):
        """Start the camera pipeline"""
        print("Starting RealSense D435 camera...")
        self.pipeline.start(self.config)

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
        Capture and return aligned RGB and Depth frames

        Returns:
            color_image: numpy array of RGB image
            depth_image: numpy array of depth image
            depth_colormap: numpy array of colorized depth image for visualization
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Create colorized depth image for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        return color_image, depth_image, depth_colormap

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


def save_images(color_image, depth_image, base_dir='captured_images'):
    """
    Save captured images to disk

    Args:
        color_image: RGB image
        depth_image: Depth image
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

    # Save depth image (as 16-bit PNG) with time-based filename
    depth_path = os.path.join(output_dir, f'depth_{time_filename}.png')
    cv2.imwrite(depth_path, depth_image)

    print(f"Saved: {color_path} and {depth_path}")

    return color_path, depth_path


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
            color_image, depth_image, depth_colormap = camera.get_frames()

            if color_image is None:
                continue

            # Create side-by-side display
            display = np.hstack((color_image, depth_colormap))

            # Add text overlay
            cv2.putText(display, f"Captured: {capture_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press 'c' to capture, 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Display
            cv2.imshow('RealSense D435 - Color | Depth', display)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF

            # Capture image
            if key == ord('c') or key == ord(' '):
                save_images(color_image, depth_image)
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