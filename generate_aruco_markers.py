#!/usr/bin/env python3
"""
Generate ArUco markers for testing
"""

import cv2
import numpy as np
from cv2 import aruco

def generate_single_marker(marker_id, dict_type, marker_size_px=200, save_path=None):
    """
    Generate a single ArUco marker

    Args:
        marker_id: ID of the marker to generate (0-249 for DICT_6X6_250)
        dict_type: ArUco dictionary type
        marker_size_px: Size of the marker in pixels
        save_path: Path to save the marker image
    """
    # Get the dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

    # Generate the marker
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

    # Add white border for better detection
    border_size = marker_size_px // 10
    marker_with_border = cv2.copyMakeBorder(
        marker_image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=255
    )

    if save_path:
        cv2.imwrite(save_path, marker_with_border)
        print(f"Saved marker {marker_id} to {save_path}")

    return marker_with_border

def generate_marker_sheet(dict_type, num_markers=12, marker_size_px=150,
                         cols=4, dict_name="DICT_6X6_250"):
    """
    Generate a sheet with multiple ArUco markers

    Args:
        dict_type: ArUco dictionary type
        num_markers: Number of markers to generate
        marker_size_px: Size of each marker in pixels
        cols: Number of columns
        dict_name: Name of the dictionary for filename
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

    rows = (num_markers + cols - 1) // cols
    spacing = marker_size_px // 5

    # Calculate sheet size
    sheet_width = cols * marker_size_px + (cols + 1) * spacing
    sheet_height = rows * marker_size_px + (rows + 1) * spacing

    # Create white sheet
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255

    # Generate and place markers
    for i in range(num_markers):
        marker_id = i
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

        row = i // cols
        col = i % cols

        y_start = spacing + row * (marker_size_px + spacing)
        x_start = spacing + col * (marker_size_px + spacing)

        sheet[y_start:y_start+marker_size_px,
              x_start:x_start+marker_size_px] = marker_img

        # Add ID text below marker
        text = f"ID: {marker_id}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x_start + (marker_size_px - text_size[0]) // 2
        text_y = y_start + marker_size_px + spacing // 2
        cv2.putText(sheet, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    save_path = f"aruco_sheet_{dict_name}_{num_markers}markers.png"
    cv2.imwrite(save_path, sheet)
    print(f"\nGenerated sheet with {num_markers} markers")
    print(f"Saved to: {save_path}")
    print(f"Image size: {sheet_width}x{sheet_height} pixels")
    print(f"For 20mm markers, print at approximately 250-300 DPI")

    return sheet

if __name__ == "__main__":
    print("="*60)
    print("ArUco Marker Generator")
    print("="*60)

    # Dictionary options
    dict_options = {
        '1': ('DICT_4X4_50', aruco.DICT_4X4_50),
        '2': ('DICT_5X5_100', aruco.DICT_5X5_100),
        '3': ('DICT_6X6_250', aruco.DICT_6X6_250),
        '4': ('DICT_7X7_50', aruco.DICT_7X7_50),
        '5': ('DICT_ARUCO_ORIGINAL', aruco.DICT_ARUCO_ORIGINAL),
    }

    print("\nAvailable dictionaries:")
    for key, (name, _) in dict_options.items():
        print(f"  {key}. {name}")

    choice = input("\nSelect dictionary (default=3 for DICT_6X6_250): ").strip()
    if not choice:
        choice = '3'

    if choice not in dict_options:
        print("Invalid choice, using DICT_6X6_250")
        choice = '3'

    dict_name, dict_type = dict_options[choice]
    print(f"\nUsing dictionary: {dict_name}")

    # Generate options
    print("\nGeneration options:")
    print("  1. Single marker")
    print("  2. Sheet with multiple markers (recommended)")

    gen_choice = input("\nSelect option (default=2): ").strip()
    if not gen_choice:
        gen_choice = '2'

    if gen_choice == '1':
        marker_id = int(input("Enter marker ID (0-249): "))
        marker_size = int(input("Enter marker size in pixels (default=400): ") or "400")
        generate_single_marker(marker_id, dict_type, marker_size,
                              f"aruco_{dict_name}_id{marker_id}.png")
    else:
        num_markers = int(input("Number of markers to generate (default=12): ") or "12")
        generate_marker_sheet(dict_type, num_markers, dict_name=dict_name)

    print("\n" + "="*60)
    print("PRINTING TIPS:")
    print("  1. Print on matte paper (avoid glossy)")
    print("  2. Use highest quality settings")
    print("  3. Ensure 100% scale (no 'fit to page')")
    print("  4. For 20mm physical markers, print at 250-300 DPI")
    print("  5. Mount on flat surface for best detection")
    print("="*60)
