#!/usr/bin/env python3
"""
Intel RealSense D435 Calibrated Image Display Script
This script displays original and undistorted images side-by-side.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import sys


class D435CalibratedViewer:
    def __init__(self, calibration_file='ds435_calibration_data.json'):
        """
        Initialize D435 Calibrated Viewer

        Args:
            calibration_file: Path to calibration JSON file
        """
        # Load calibration data
        self.load_calibration(calibration_file)

        # Create pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure stream with calibration resolution
        self.config.enable_stream(
            rs.stream.color,
            self.calib_width,
            self.calib_height,
            rs.format.bgr8,
            30
        )

    def load_calibration(self, calibration_file):
        """Load calibration data from JSON file"""
        try:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)

            self.calib_serial = calib_data['seril_number']
            self.calib_width = calib_data['width']
            self.calib_height = calib_data['height']

            # Camera matrix
            self.camera_matrix = np.array([
                [calib_data['fx'], 0, calib_data['cx']],
                [0, calib_data['fy'], calib_data['cy']],
                [0, 0, 1]
            ], dtype=np.float32)

            # Distortion coefficients
            self.dist_coeffs = np.array([
                calib_data['k1'],
                calib_data['k2'],
                calib_data['p1'],
                calib_data['p2'],
                calib_data['k3']
            ], dtype=np.float32)

            print(f"Loaded calibration data:")
            print(f"  Serial Number: {self.calib_serial}")
            print(f"  Resolution: {self.calib_width}x{self.calib_height}")
            print(f"  Reprojection Error: {calib_data['reprojection_error']}")

        except FileNotFoundError:
            print(f"Error: Calibration file '{calibration_file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            sys.exit(1)

    def start(self):
        """Start the camera pipeline"""
        print("\nStarting RealSense D435 camera...")
        profile = self.pipeline.start(self.config)

        # Get device information
        device = profile.get_device()
        device_serial = device.get_info(rs.camera_info.serial_number)
        print(f"Camera Serial Number: {device_serial}")

        # Check if serial number matches
        if device_serial != self.calib_serial:
            print(f"\nWARNING: Serial number mismatch!")
            print(f"  Calibration file: {self.calib_serial}")
            print(f"  Connected camera: {device_serial}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                self.pipeline.stop()
                sys.exit(0)
        else:
            print("Serial number matched! Using calibration data.")

        # Wait for auto-exposure to stabilize
        print("Waiting for camera to stabilize...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("Camera ready!")

        # Compute optimal new camera matrix
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (self.calib_width, self.calib_height),
            1,  # alpha=1 keeps all pixels
            (self.calib_width, self.calib_height)
        )

        # Compute undistortion maps
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
            (self.calib_width, self.calib_height),
            cv2.CV_32FC1
        )

    def stop(self):
        """Stop the camera pipeline"""
        self.pipeline.stop()
        print("Camera stopped.")

    def get_frame(self):
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

    def undistort_image(self, image):
        """
        Apply calibration to undistort image

        Args:
            image: Input distorted image

        Returns:
            undistorted_image: Undistorted image
        """
        # Apply undistortion using pre-computed maps
        undistorted = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

        return undistorted


def main():
    """Main function for displaying original and calibrated images"""
    # Create viewer instance
    viewer = D435CalibratedViewer(calibration_file='ds435_calibration_data.json')

    try:
        # Start camera
        viewer.start()

        print("\nControls:")
        print("  Press 'q' or ESC to quit")

        while True:
            # Get frame
            original_image = viewer.get_frame()

            if original_image is None:
                continue

            # Apply calibration
            calibrated_image = viewer.undistort_image(original_image)

            # Create side-by-side display
            display = np.hstack((original_image, calibrated_image))

            # Add labels
            cv2.putText(display, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display, "Calibrated (Undistorted)", (original_image.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display
            cv2.imshow('D435 - Original (Left) | Calibrated (Right)', display)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord('q') or key == 27:  # 27 is ESC
                break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        viewer.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
