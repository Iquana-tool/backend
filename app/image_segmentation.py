import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def select_mask(
    image_path,
    sam_checkpoint,
    model_type="vit_b",  # Choose the fastest model variant
    output_dir="output",
    selected_mask_dir="selected_masks"
):
    """
    Performs interactive mask selection using SAM.
    The user can hover to highlight a region and click to select and save a mask.

    :param image_path: Path to input image.
    :param sam_checkpoint: Path to SAM model weights.
    :param model_type: SAM model variant ("vit_b", "vit_l", "vit_h").
    :param output_dir: Directory to store processed masks.
    :param selected_mask_dir: Directory to store selected masks.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(selected_mask_dir, exist_ok=True)

    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

    # Enable FP16 precision on GPU for speed improvement
    if device == "cuda":
        sam.half()

    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image = np.ascontiguousarray(image, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image for faster processing
    image_small = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)

    # Display image immediately while processing masks
    cv2.imshow("Hover and Click on Masks", image)
    cv2.waitKey(1)

    # Generate segmentation masks
    with torch.no_grad():
        masks = mask_generator.generate(image_small)

    # Resize masks and filter small ones
    refined_masks = []
    min_mask_area = 5000  

    for m in masks:
        segmentation = cv2.resize(
            m["segmentation"].astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        if np.count_nonzero(segmentation) > min_mask_area:
            refined_masks.append(segmentation)

    # Apply morphological operations to remove small artifacts
    kernel = np.ones((5, 5), np.uint8)
    refined_masks = [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) for mask in refined_masks]

    # Convert masks to a NumPy array for faster interaction
    refined_masks = np.array(refined_masks, dtype=np.uint8)

    # Enable real-time hover and click interaction
    highlighted_mask_idx = None

    def on_mouse(event, x, y, flags, param):
        """
        Handles real-time hover and click interaction for mask selection.
        - Hover: Highlights the mask under the cursor.
        - Click: Selects and saves the mask for further processing.
        """
        nonlocal highlighted_mask_idx
        display_img = image.copy()

        # Identify the mask under the cursor
        mask_found = False
        for idx, mask in enumerate(refined_masks):
            if mask[y, x] > 0:
                highlighted_mask_idx = idx
                mask_found = True
                break

        if not mask_found:
            highlighted_mask_idx = None

        # Apply hover effect (highlight in orange)
        if event == cv2.EVENT_MOUSEMOVE and highlighted_mask_idx is not None:
            hover_mask = np.zeros_like(display_img)
            hover_mask[refined_masks[highlighted_mask_idx] > 0] = (255, 180, 100)  
            display_img = cv2.addWeighted(display_img, 1.0, hover_mask, 0.3, 0)

        # Click event to select and save mask
        if event == cv2.EVENT_LBUTTONDOWN and highlighted_mask_idx is not None:
            selected_mask = refined_masks[highlighted_mask_idx]
            print(f"Selected Mask {highlighted_mask_idx + 1}")

            # Extract mask region
            x, y, w, h = cv2.boundingRect(selected_mask.astype(np.uint8))
            cropped_region = image[y:y+h, x:x+w]

            # Save the selected mask
            selected_mask_path = os.path.join(selected_mask_dir, "selected_mask.png")
            cv2.imwrite(selected_mask_path, cropped_region)

            print(f"[Selection] Saved selected mask: {selected_mask_path}")

            # Show the selected mask
            cv2.imshow("Selected Mask", cropped_region)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imshow("Hover and Click on Masks", display_img)

    # Attach mouse event callback to enable hover and click
    cv2.setMouseCallback("Hover and Click on Masks", on_mouse)
    print("Hover over a mask to highlight; left-click to select.")

    # Ensure image pops up instantly and interaction starts immediately
    cv2.waitKey(0)
    cv2.destroyAllWindows()
