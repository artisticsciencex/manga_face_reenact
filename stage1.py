

import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import os
import json

# --- Helper functions (to_square_bbox, get_padded_landmark_bbox) remain the same ---
def to_square_bbox(bbox: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2
    side_length = max(width, height)
    new_x1 = int(center_x - side_length / 2)
    new_y1 = int(center_y - side_length / 2)
    new_x2 = int(center_x + side_length / 2)
    new_y2 = int(center_y + side_length / 2)
    return (new_x1, new_y1, new_x2, new_y2)

def get_padded_landmark_bbox(landmarks: np.ndarray, padding_factor: float = 1.0) -> tuple:
    min_x = int(np.min(landmarks[:, 0]))
    min_y = int(np.min(landmarks[:, 1]))
    max_x = int(np.max(landmarks[:, 0]))
    max_y = int(np.max(landmarks[:, 1]))
    width = max_x - min_x
    height = max_y - min_y
    pad_x = int(width * padding_factor)
    pad_y = int(height * padding_factor)
    return (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)

# --- Manual Cropping Logic (manual_crop_callback) remains the same ---
ref_points = []
cropping = False
image_for_drawing = None

def manual_crop_callback(event, x, y, flags, param):
    global ref_points, cropping, image_for_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        ref_points.append((x, y))
        cropping = False
        start_point = ref_points[0]
        end_point = ref_points[1]
        dx = abs(end_point[0] - start_point[0])
        dy = abs(end_point[1] - start_point[1])
        side_length = max(dx, dy)
        square_end_point = (start_point[0] + side_length, start_point[1] + side_length)
        cv2.rectangle(image_for_drawing, start_point, square_end_point, (0, 255, 0), 2)
        param.append((start_point[0], start_point[1], square_end_point[0], square_end_point[1]))
        cv2.imshow("image", image_for_drawing)
        print(f"Manual box added: {param[-1]}")

# --- Main Preparation Function (with updated final loop) ---

def prepare_faces(image_path: str, 
                  mode: str = 'auto',
                  output_dir: str = "prepared_faces_final"):
    """
    Prepares faces and saves them into a dedicated sub-folder named after the source image.
    Each sub-folder will contain the cropped faces and its own metadata.json file.
    """
    try:
        source_img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Source image not found at {image_path}")
        return []

    # --- NEW: Create a dedicated sub-directory for this source image ---
    base_filename = os.path.basename(image_path)
    source_name, _ = os.path.splitext(base_filename)
    specific_output_dir = os.path.join(output_dir, source_name)
    os.makedirs(specific_output_dir, exist_ok=True)
    print(f"--- Processing source image: {base_filename} ---")
    print(f"Saving results to: {specific_output_dir}")

    square_bounding_boxes = []

    # --- Face detection part remains the same ---
    if mode == 'auto':
        print("Running in AUTO (Landmark-Based) mode...")
        app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        source_img_cv = cv2.imread(image_path)
        faces = app.get(source_img_cv)
        
        if not faces:
            print("Auto mode did not detect any faces.")
        else:
            print(f"Auto mode found {len(faces)} faces.")
            for face in faces:
                landmark_bbox = get_padded_landmark_bbox(face.kps, padding_factor=1.0)
                square_bbox = to_square_bbox(landmark_bbox)
                square_bounding_boxes.append(square_bbox)

    elif mode == 'manual':
        print("Running in MANUAL mode...")
        print("Draw boxes around faces. Press 'r' to reset. Press 'q' to quit and process.")
        global image_for_drawing
        source_img_cv = cv2.imread(image_path)
        clone = source_img_cv.copy()
        image_for_drawing = source_img_cv.copy()
        
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", manual_crop_callback, square_bounding_boxes)
        
        while True:
            cv2.imshow("image", image_for_drawing)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                image_for_drawing = clone.copy()
                square_bounding_boxes.clear()
                print("Boxes have been reset.")
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        print(f"Manual mode finished with {len(square_bounding_boxes)} selected faces.")

    else:
        raise ValueError("Mode must be either 'auto' or 'manual'")

    # --- REVISED Shared Final Processing Step ---
    all_faces_metadata = []
    if len(square_bounding_boxes) > 0:
        print(f"Cropping, resizing, and saving faces...")

        # Face IDs will now start from 0 for each source image.
        for i, coords in enumerate(square_bounding_boxes):
            face_id = i
            coords_tuple = tuple(map(int, coords))
            
            cropped_face = source_img_pil.crop(coords_tuple)
            
            original_size = cropped_face.size
            paste_coordinates = (coords_tuple[0], coords_tuple[1])
            
            resized_face = cropped_face.resize((512, 512), Image.Resampling.LANCZOS)
            
            face_filename = f"face_{face_id}.png"
            # Save the face image in the new specific directory
            face_saved_path = os.path.join(specific_output_dir, face_filename)
            resized_face.save(face_saved_path)
            
            face_metadata = {
                'face_id': face_id,
                'face_image_512px_path': face_saved_path,
                'original_size': original_size,
                'paste_coordinates': paste_coordinates,
                'original_bbox_square': coords_tuple,
                'source_image': image_path
            }
            all_faces_metadata.append(face_metadata)
            print(f"  - Processed and saved Face {face_id} to {face_saved_path}")

        # --- SIMPLIFIED Metadata Saving ---
        # Save the metadata.json inside the specific output directory.
        metadata_path = os.path.join(specific_output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(all_faces_metadata, f, indent=4)
            
        print(f"Successfully saved metadata to {metadata_path}")
        return all_faces_metadata
    else:
        print("No new faces were processed for this image.")
        return []

# --- Example Usage ---
if __name__ == '__main__':
    # Define two different source images to demonstrate the new structure
    source_image_1 = "e6bf7ae0-4923-46c6-9bc9-c3142652118b.png"
    # source_image_2 = "efc23902-07b7-46ff-ad9a-74b8ab4c6c52.png" # A different image

    # Process the first image
    if not os.path.exists(source_image_1):
        print(f"!!! ERROR: Source image '{source_image_1}' not found.")
    else:
        prepare_faces(image_path=source_image_1, mode='manual') # auto or manual
    
    print("\n" + "="*50 + "\n")

    # Process the second image
    # if not os.path.exists(source_image_2):
    #     print(f"!!! ERROR: Source image '{source_image_2}' not found.")
    # else:
    #     prepare_faces(image_path=source_image_2, mode='manual') # auto or manual
