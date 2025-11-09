#!/usr/bin/env python3
"""
Intel RealSense D435 ArUco Marker Detection Script
This script detects ArUco markers using calibrated camera parameters.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import json
import sys


class D435ArUcoDetector:
    def __init__(self, calibration_file='ds435_calibration_data.json',
                 aruco_dict_type=aruco.DICT_APRILTAG_36h11, marker_size=0.02):
        """
        Initialize D435 ArUco/AprilTag Detector

        Args:
            calibration_file: Path to calibration JSON file
            aruco_dict_type: ArUco/AprilTag dictionary type (default: DICT_APRILTAG_36h11)
            marker_size: Physical size of marker in meters (default: 0.02m = 20mm)
        """
        # Load calibration data
        self.load_calibration(calibration_file)

        # Check if using AprilTag
        self.is_apriltag = 'APRILTAG' in str(aruco_dict_type)

        # ArUco detector parameters (using modern OpenCV API)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

        # Try using new API (OpenCV 4.7+), fallback to legacy API
        try:
            # Modern API
            self.detector = aruco.ArucoDetector(self.aruco_dict)

            # Optimize detection parameters (more permissive for small markers)
            params = aruco.DetectorParameters()
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 10
            params.adaptiveThreshConstant = 7
            params.minMarkerPerimeterRate = 0.01  # Lower = detect smaller markers
            params.maxMarkerPerimeterRate = 4.0
            params.polygonalApproxAccuracyRate = 0.05  # Higher = more permissive
            params.minCornerDistanceRate = 0.01  # Lower = more permissive
            params.minDistanceToBorder = 1  # Lower = detect markers near edges
            params.minMarkerDistanceRate = 0.01  # Allow close markers
            # Use AprilTag corner refinement if using AprilTag dictionary
            if self.is_apriltag:
                params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
            else:
                params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 30
            params.cornerRefinementMinAccuracy = 0.1
            params.markerBorderBits = 1  # For 6x6 markers
            params.minOtsuStdDev = 5.0  # Lower = more permissive thresholding
            params.perspectiveRemovePixelPerCell = 4
            params.perspectiveRemoveIgnoredMarginPerCell = 0.13
            self.detector.setDetectorParameters(params)
            self.use_new_api = True
            print("Using new OpenCV ArUco API (4.7+)")
        except AttributeError:
            # Legacy API for older OpenCV versions (more permissive settings)
            self.aruco_params = aruco.DetectorParameters_create()
            self.aruco_params.adaptiveThreshWinSizeMin = 3
            self.aruco_params.adaptiveThreshWinSizeMax = 23
            self.aruco_params.adaptiveThreshWinSizeStep = 10
            self.aruco_params.adaptiveThreshConstant = 7
            self.aruco_params.minMarkerPerimeterRate = 0.01  # Lower = more permissive
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            self.aruco_params.polygonalApproxAccuracyRate = 0.05  # Higher = more permissive
            self.aruco_params.minCornerDistanceRate = 0.01  # Lower = more permissive
            self.aruco_params.minDistanceToBorder = 1  # Lower = detect near edges
            self.aruco_params.minMarkerDistanceRate = 0.01
            # Use AprilTag corner refinement if using AprilTag dictionary
            if self.is_apriltag:
                self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
            else:
                self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            self.aruco_params.cornerRefinementWinSize = 5
            self.aruco_params.cornerRefinementMaxIterations = 30
            self.aruco_params.cornerRefinementMinAccuracy = 0.1
            self.aruco_params.markerBorderBits = 1  # For 6x6 markers
            self.aruco_params.minOtsuStdDev = 5.0
            self.aruco_params.perspectiveRemovePixelPerCell = 4
            self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
            self.use_new_api = False
            print("Using legacy OpenCV ArUco API")

        self.marker_size = marker_size

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

    def detect_markers(self, image, debug=False):
        """
        Detect ArUco markers in the image

        Args:
            image: Input image
            debug: If True, show debug information

        Returns:
            corners: Detected marker corners
            ids: Detected marker IDs
            rejected: Rejected candidates
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)

        # Optional: Apply bilateral filter to reduce noise while preserving edges
        # gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Detect markers using appropriate API
        if self.use_new_api:
            # Modern API (OpenCV 4.7+)
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            # Legacy API
            corners, ids, rejected = aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

        # Debug information
        if debug:
            print(f"\n[DEBUG] Detection results:")
            print(f"  Detected markers: {len(ids) if ids is not None else 0}")
            print(f"  Rejected candidates: {len(rejected)}")
            if ids is not None:
                print(f"  Marker IDs: {ids.flatten()}")

        return corners, ids, rejected

    def estimate_pose(self, corners):
        """
        Estimate pose of detected markers

        Args:
            corners: Marker corners

        Returns:
            rvecs: Rotation vectors
            tvecs: Translation vectors
        """
        # Estimate pose
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_size,
            self.camera_matrix,
            self.dist_coeffs
        )

        return rvecs, tvecs

    def draw_detections(self, image, corners, ids, rvecs=None, tvecs=None):
        """
        Draw detected markers and pose on image

        Args:
            image: Input image
            corners: Marker corners
            ids: Marker IDs
            rvecs: Rotation vectors (optional)
            tvecs: Translation vectors (optional)

        Returns:
            output_image: Image with drawn markers
        """
        output_image = image.copy()

        if ids is not None and len(ids) > 0:
            # Draw detected markers
            aruco.drawDetectedMarkers(output_image, corners, ids)

            # Draw axis for each marker if pose is estimated
            if rvecs is not None and tvecs is not None:
                for i in range(len(ids)):
                    # Draw 3D axis
                    cv2.drawFrameAxes(
                        output_image,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvecs[i],
                        tvecs[i],
                        self.marker_size * 0.5
                    )

                    # Calculate distance
                    tvec = tvecs[i][0]
                    distance = np.linalg.norm(tvec)

                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvecs[i])

                    # Calculate Euler angles (roll, pitch, yaw) in degrees
                    # Using the rotation matrix to get angles
                    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)

                    singular = sy < 1e-6

                    if not singular:
                        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    else:
                        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                        yaw = 0

                    # Convert to degrees
                    roll_deg = np.degrees(roll)
                    pitch_deg = np.degrees(pitch)
                    yaw_deg = np.degrees(yaw)

                    # Get corner position for text
                    corner = corners[i][0][0]  # Top-left corner
                    x, y = int(corner[0]), int(corner[1])

                    # Display marker ID and distance
                    text1 = f"ID:{ids[i][0]} D:{distance*100:.1f}cm"
                    cv2.putText(
                        output_image,
                        text1,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    # Display position (X, Y, Z)
                    text2 = f"Pos: X:{tvec[0]*100:.1f} Y:{tvec[1]*100:.1f} Z:{tvec[2]*100:.1f}cm"
                    cv2.putText(
                        output_image,
                        text2,
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1
                    )

                    # Display orientation (Roll, Pitch, Yaw)
                    text3 = f"Rot: R:{roll_deg:.1f} P:{pitch_deg:.1f} Y:{yaw_deg:.1f}deg"
                    cv2.putText(
                        output_image,
                        text3,
                        (x, y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1
                    )

        return output_image


def main():
    """Main function for ArUco marker detection"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='D435 ArUco/AprilTag Marker Detection')
    parser.add_argument('--dict', type=str, default=None,
                       choices=['DICT_4X4_50', 'DICT_5X5_100', 'DICT_6X6_250',
                               'DICT_7X7_50', 'DICT_ARUCO_ORIGINAL',
                               'DICT_APRILTAG_16h5', 'DICT_APRILTAG_25h9',
                               'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11'],
                       help='ArUco/AprilTag dictionary type (if not specified, will ask interactively)')
    parser.add_argument('--size', type=float, default=None,
                       help='Marker size in meters (if not specified, will ask interactively)')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive selection (use defaults)')

    args = parser.parse_args()

    # Get dictionary type
    dict_mapping = {
        'DICT_4X4_50': aruco.DICT_4X4_50,
        'DICT_5X5_100': aruco.DICT_5X5_100,
        'DICT_6X6_250': aruco.DICT_6X6_250,
        'DICT_7X7_50': aruco.DICT_7X7_50,
        'DICT_ARUCO_ORIGINAL': aruco.DICT_ARUCO_ORIGINAL,
        'DICT_APRILTAG_16h5': aruco.DICT_APRILTAG_16h5,
        'DICT_APRILTAG_25h9': aruco.DICT_APRILTAG_25h9,
        'DICT_APRILTAG_36h10': aruco.DICT_APRILTAG_36h10,
        'DICT_APRILTAG_36h11': aruco.DICT_APRILTAG_36h11,
    }

    # Interactive selection if not specified via command line
    if args.dict is None and not args.no_interactive:
        print("="*60)
        print("D435 ArUco/AprilTag Marker Detection")
        print("="*60)
        print("\nSelect marker type:")
        print("\n[AprilTag]")
        print("  1. DICT_APRILTAG_36h11  (recommended for AprilTag)")
        print("  2. DICT_APRILTAG_16h5")
        print("  3. DICT_APRILTAG_25h9")
        print("  4. DICT_APRILTAG_36h10")
        print("\n[ArUco]")
        print("  5. DICT_6X6_250  (recommended for ArUco)")
        print("  6. DICT_4X4_50")
        print("  7. DICT_5X5_100")
        print("  8. DICT_7X7_50")
        print("  9. DICT_ARUCO_ORIGINAL")

        choice = input("\nEnter choice (1-9) [default: 1]: ").strip()

        choice_map = {
            '1': 'DICT_APRILTAG_36h11',
            '2': 'DICT_APRILTAG_16h5',
            '3': 'DICT_APRILTAG_25h9',
            '4': 'DICT_APRILTAG_36h10',
            '5': 'DICT_6X6_250',
            '6': 'DICT_4X4_50',
            '7': 'DICT_5X5_100',
            '8': 'DICT_7X7_50',
            '9': 'DICT_ARUCO_ORIGINAL',
        }

        if choice == '':
            choice = '1'

        if choice in choice_map:
            args.dict = choice_map[choice]
        else:
            print(f"Invalid choice. Using default: DICT_APRILTAG_36h11")
            args.dict = 'DICT_APRILTAG_36h11'
    elif args.dict is None:
        # Use default if --no-interactive is specified
        args.dict = 'DICT_APRILTAG_36h11'

    # Interactive marker size selection
    if args.size is None and not args.no_interactive:
        size_input = input("\nEnter marker size in mm [default: 20]: ").strip()
        if size_input == '':
            args.size = 0.02  # 20mm
        else:
            try:
                args.size = float(size_input) / 1000.0  # Convert mm to meters
            except ValueError:
                print("Invalid size. Using default: 20mm")
                args.size = 0.02
    elif args.size is None:
        args.size = 0.02  # 20mm default

    print(f"\nSelected configuration:")
    print(f"  Dictionary: {args.dict}")
    print(f"  Marker size: {args.size * 1000:.1f}mm")
    print(f"  Debug mode: {args.debug}")
    print()

    # Create detector instance
    detector = D435ArUcoDetector(
        calibration_file='ds435_calibration_data.json',
        aruco_dict_type=dict_mapping[args.dict],
        marker_size=args.size
    )

    try:
        # Start camera
        detector.start()

        marker_type = "AprilTag" if detector.is_apriltag else "ArUco"
        print(f"\n{marker_type} Marker Detection:")
        print(f"  Dictionary: {args.dict}")
        print(f"  Marker Size: {detector.marker_size * 1000}mm")
        print(f"  Debug Mode: {args.debug}")
        print("\nControls:")
        print("  Press 'q' or ESC to quit")
        print("  Press 'd' to toggle debug mode")
        print("\nTip: Use --dict to change marker type")
        print("  AprilTag: DICT_APRILTAG_36h11, DICT_APRILTAG_16h5, etc.")
        print("  ArUco: DICT_6X6_250, DICT_4X4_50, etc.")

        marker_count = 0
        debug_mode = args.debug

        while True:
            # Get frame
            image = detector.get_frame()

            if image is None:
                continue

            # Detect markers
            corners, ids, rejected = detector.detect_markers(image, debug=debug_mode)

            # Estimate pose if markers detected
            rvecs, tvecs = None, None
            if ids is not None and len(ids) > 0:
                rvecs, tvecs = detector.estimate_pose(corners)
                marker_count = len(ids)
            else:
                marker_count = 0

            # Draw detections
            output_image = detector.draw_detections(image, corners, ids, rvecs, tvecs)

            # Add status text with marker type
            marker_type = "AprilTag" if detector.is_apriltag else "ArUco"
            status_text = f"{marker_type} Detected: {marker_count} | {args.dict}"
            cv2.putText(
                output_image,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if marker_count > 0 else (0, 0, 255),
                2
            )

            # Add rejected candidates count
            if rejected is not None:
                rejected_text = f"Rejected: {len(rejected)}"
                cv2.putText(
                    output_image,
                    rejected_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1
                )

            # Display
            cv2.imshow('D435 ArUco Marker Detection', output_image)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF

            # Toggle debug mode
            if key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

            # Quit
            if key == ord('q') or key == 27:  # 27 is ESC
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            detector.stop()
        except:
            pass
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
