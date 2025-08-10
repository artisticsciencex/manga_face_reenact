import json
from PIL import Image
import os

def recombine_panel(original_panel_path: str, 
                      metadata_path: str, 
                      reenacted_faces_dir: str, 
                      output_path: str):
    """
    Recombines reenacted faces onto the original manga panel using metadata.

    Args:
        original_panel_path (str): Path to the original, full manga panel image.
        metadata_path (str): Path to the metadata.json file from Stage 1.
        reenacted_faces_dir (str): Path to the folder containing the new, expressive faces.
        output_path (str): Path to save the final, recombined image.
    """
    print("--- Starting Stage 3: Recombination ---")
    
    # --- 1. Load all necessary files ---
    try:
        original_manga_panel = Image.open(original_panel_path).convert("RGBA")
        print(f"Loaded original panel: {original_panel_path}")
    except FileNotFoundError:
        print(f"!!! ERROR: Original panel not found at '{original_panel_path}'")
        return

    try:
        with open(metadata_path, 'r') as f:
            all_faces_metadata = json.load(f)
        print(f"Loaded metadata for {len(all_faces_metadata)} faces from '{metadata_path}'")
    except FileNotFoundError:
        print(f"!!! ERROR: Metadata file not found at '{metadata_path}'")
        return

    # --- 2. Process each face using the metadata ---
    recombined_count = 0
    for face_data in all_faces_metadata:
        face_id = face_data['face_id']
        
        # Construct the expected path for the reenacted face
        # Assumes the filename (e.g., 'face_0.png') is the same as in the metadata
        original_filename = os.path.basename(face_data['face_image_512px_path'])
        reenacted_face_path = os.path.join(reenacted_faces_dir, original_filename)

        try:
            reenacted_face_512px = Image.open(reenacted_face_path).convert("RGBA")
        except FileNotFoundError:
            print(f"--- WARNING: Reenacted face for ID {face_id} not found at '{reenacted_face_path}'. Skipping.")
            continue

        # --- 3. Get the "clues" from the metadata ---
        orig_size = tuple(face_data['original_size'])
        paste_coords = tuple(face_data['paste_coordinates'])

        # --- 4. Resize the reenacted face BACK to its original size ---
        print(f"  - Processing face {face_id}: Resizing from 512x512 to {orig_size}...")
        face_to_paste = reenacted_face_512px.resize(orig_size, Image.Resampling.LANCZOS)

        # --- 5. Paste it back onto the original manga panel ---
        print(f"  - Pasting face {face_id} at coordinates {paste_coords}...")
        # Using the image itself as the mask ensures transparency is handled correctly
        original_manga_panel.paste(face_to_paste, paste_coords, face_to_paste)
        recombined_count += 1

    # --- 6. Save the final result ---
    if recombined_count > 0:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        original_manga_panel.save(output_path)
        print(f"\n--- Recombination Complete ---")
        print(f"Successfully recombined {recombined_count} faces.")
        print(f"Final panel saved to: {output_path}")
        print("This is the 'before-artifact-removal' version, ready for the final polishing step.")
    else:
        print("\n--- Recombination Finished ---")
        print("No reenacted faces were found to process.")


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Define the path to your original full-scene image
    #    (The one you fed into the Stage 1 script)
    original_panel = "ChatGPT Image Jun 24, 2025, 06_24_00 PM.png"

    # 2. Define the path to the metadata file created by Stage 1
    #    (This should be correct if you use the default folder names)
    metadata_file = "prepared_faces_final/metadata.json"

    # 3. Define the path to the FOLDER where you saved your NEW expressive faces
    #    (The output from LivePortrait)
    reenacted_folder = "reenacted_faces/"

    # 4. Define where you want to save the final, combined image
    final_output_image = "output/recombined_panel_01.png"

    # Run the recombination tool
    recombine_panel(
        original_panel_path=original_panel,
        metadata_path=metadata_file,
        reenacted_faces_dir=reenacted_folder,
        output_path=final_output_image
    )
