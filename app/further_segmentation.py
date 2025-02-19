import os
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def apply_background_filter(image, mask_binary):
    """
    Applies a darkened background filter while keeping the mask area highlighted.

    :param image: Original image.
    :param mask_binary: Binary mask for segmentation.
    :return: Image with background filter applied.
    """
    # Ensure the mask size matches the image
    mask_binary_resized = cv2.resize(mask_binary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert grayscale mask to 3-channel format
    mask_binary_3ch = cv2.cvtColor(mask_binary_resized, cv2.COLOR_GRAY2BGR)

    # Create a darkened version of the image
    darkened_background = cv2.addWeighted(image, 0.3, np.zeros_like(image), 0.7, 0)

    # Combine darkened background with original image inside the mask
    filtered_image = np.where(mask_binary_3ch > 0, image, darkened_background)

    return filtered_image

def segment_selected_region(selected_mask_path, sam_checkpoint, model_type="vit_b", output_dir="polyps_masks"):
    """
    Further segments only the selected region.

    :param selected_mask_path: Path to the cropped selected mask.
    :param sam_checkpoint: Path to SAM model weights.
    :param model_type: "vit_b", "vit_l", or "vit_h"
    :param output_dir: Where to save refined segmentation masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load selected mask region
    image = cv2.imread(selected_mask_path)
    if image is None:
        raise FileNotFoundError(f"Could not load selected mask: {selected_mask_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate refined masks
    masks = mask_generator.generate(image_rgb)

    for idx, mask in enumerate(masks):
        mask_overlay = np.zeros_like(image)
        mask_resized = mask['segmentation'].astype(np.uint8) * 255

        # Apply thresholding to remove weak masks
        _, mask_binary = cv2.threshold(mask_resized, 50, 255, cv2.THRESH_BINARY)

        # Create mask overlay
        overlay = np.zeros_like(image)
        overlay[mask_binary > 0] = (0, 255, 0)  # Green

        # Blend with original image
        blended_mask = cv2.addWeighted(image, 0.7, overlay, 0.5, 0)

        # Save mask without background filter
        mask_path = os.path.join(output_dir, f"polyps_mask_{idx + 1}.png")
        cv2.imwrite(mask_path, blended_mask)
        print(f"Saved: {mask_path}")

        # Save mask with background filter
        filtered_mask = apply_background_filter(image, mask_binary)
        blended_mask_filtered = cv2.addWeighted(filtered_mask, 0.7, overlay, 0.5, 0)

        mask_filtered_path = os.path.join(output_dir, f"polyps_mask_filtered_{idx + 1}.png")
        cv2.imwrite(mask_filtered_path, blended_mask_filtered)
        print(f"Saved filtered mask: {mask_filtered_path}")

    cv2.destroyAllWindows()

def create_polygon_and_filter_masks(original_image_path, sam_checkpoint, model_type="vit_h", output_dir="fine_tune_masks"):
    """
    Step 3: Selects a polygon on the original image, applies SAM segmentation inside the polygon.

    :param original_image_path: Path to the original image.
    :param sam_checkpoint: Path to SAM model weights.
    :param model_type: Model type for SAM ("vit_b", "vit_l", "vit_h").
    :param output_dir: Directory to save fine-tuned segmentation masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Ensure model_type is valid
    if model_type not in sam_model_registry:
        raise ValueError(f"Invalid model type '{model_type}'. Choose from 'vit_b', 'vit_l', or 'vit_h'.")

    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load original image
    image = cv2.imread(original_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load original image: {original_image_path}")

    # Define parameters for the polygon tool
    points = []
    max_points = 8  # Maximum polygon points

    def select_polygon_points(event, x, y, flags, param):
        """ Capture polygon points from user clicks """
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < max_points:
                points.append((x, y))
                print(f"Point {len(points)} added at position ({x}, {y})")
            else:
                print("Maximum points reached (8). Press Enter to finalize.")

    cv2.namedWindow("Polygon Selection", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Polygon Selection", select_polygon_points)

    # Loop to display image and capture points
    while True:
        display_image = image.copy()
        if len(points) > 1:
            cv2.polylines(display_image, [np.array(points, np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)
        for point in points:
            cv2.circle(display_image, point, 5, (0, 0, 255), -1)

        cv2.imshow("Polygon Selection", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc to exit
            points.clear()
            print("Exited without saving the mask.")
            cv2.destroyAllWindows()
            return
        elif key == 13:  # Enter to finalize and proceed
            print("Polygon selection finalized.")
            break

    cv2.destroyAllWindows()

    if len(points) >= 3:
        # Create a polygon mask
        polygon_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        points_array = np.array(points, np.int32)
        cv2.fillPoly(polygon_mask, [points_array], 255)
        print("Polygon mask created from points.")

        # Apply background filter to polygon region
        image_filtered = apply_background_filter(image, polygon_mask)

        # Determine bounding box around the polygon
        x_min, y_min = np.min(points_array[:, 0]), np.min(points_array[:, 1])
        x_max, y_max = np.max(points_array[:, 0]), np.max(points_array[:, 1])
        cropped_image = image_filtered[y_min:y_max, x_min:x_max]

        # Generate SAM masks inside the polygon
        masks = mask_generator.generate(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        if masks:
            for i, mask in enumerate(masks):
                mask_overlay = np.zeros_like(image)
                mask_resized = cv2.resize(mask['segmentation'].astype(np.uint8), 
                                          (x_max - x_min, y_max - y_min), 
                                          interpolation=cv2.INTER_NEAREST)

                mask_overlay[y_min:y_max, x_min:x_max][mask_resized > 0] = (0, 255, 0)  # Green

                blended_mask = cv2.addWeighted(image_filtered, 0.7, mask_overlay, 0.3, 0)

                final_mask_path = os.path.join(output_dir, f"fine_tune_mask_{i + 1}.png")
                cv2.imwrite(final_mask_path, blended_mask)
                print(f"Saved fine-tuned mask: {final_mask_path}")

            cv2.destroyAllWindows()
        else:
            print("No segmentation masks generated.")
    else:
        print("Not enough points to create a polygon.")