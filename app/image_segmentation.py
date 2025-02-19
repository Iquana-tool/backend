import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def select_mask(
    image_path,
    sam_checkpoint,
    model_type="vit_b",
    output_dir="output",
    selected_mask_dir="selected_masks"
):
    """
    Refines segmentation masks before further processing.
    Filters out small/noisy masks, applies morphological operations.
    
    :param image_path: Path to input image.
    :param sam_checkpoint: Path to SAM model weights.
    :param model_type: "vit_b", "vit_l", or "vit_h"
    :param output_dir: Where to save processed masks.
    :param selected_mask_dir: Directory for further segmentation.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(selected_mask_dir, exist_ok=True)

    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_small = cv2.resize(image_rgb, (512, 512))

    # Generate masks
    masks = mask_generator.generate(image_small)

    # Resize masks to original image size and filter out small/noisy masks
    valid_masks = []
    min_mask_area = 5000  # Adjust this value based on dataset quality

    for idx, m in enumerate(masks):
        segmentation = cv2.resize(
            m["segmentation"].astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Calculate mask area and remove small/noisy masks
        mask_area = np.count_nonzero(segmentation)
        if mask_area > min_mask_area:
            valid_masks.append(segmentation)

    # Apply morphological filtering to remove small artifacts
    kernel = np.ones((5, 5), np.uint8)
    refined_masks = [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) for mask in valid_masks]

    # Allow user to select a refined mask
    highlighted_mask_idx = None

    def on_mouse(event, x, y, flags, param):
        nonlocal highlighted_mask_idx

        mask_found = False
        for idx, mask in enumerate(refined_masks):
            if mask[y, x] > 0:
                highlighted_mask_idx = idx
                mask_found = True
                break

        if not mask_found:
            highlighted_mask_idx = None

        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            display_img = image.copy()

            # Highlight hovered mask
            if highlighted_mask_idx is not None:
                hover_mask = np.zeros_like(display_img)
                hover_mask[refined_masks[highlighted_mask_idx] > 0] = (255, 180, 100)
                display_img = cv2.addWeighted(display_img, 1.0, hover_mask, 0.3, 0)

            if event == cv2.EVENT_LBUTTONDOWN and highlighted_mask_idx is not None:
                selected_mask = refined_masks[highlighted_mask_idx]

                print(f"Selected Mask {highlighted_mask_idx + 1}")

                # Extract mask region
                x, y, w, h = cv2.boundingRect(selected_mask.astype(np.uint8))
                cropped_region = image[y:y+h, x:x+w]

                # Save selected mask for further segmentation
                selected_mask_path = os.path.join(selected_mask_dir, "selected_mask.png")
                cv2.imwrite(selected_mask_path, cropped_region)

                print(f"[Selection] Saved selected mask: {selected_mask_path}")

                # Display selected region
                cv2.imshow("Selected Mask", cropped_region)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            cv2.imshow("Hover and Click on Masks", display_img)

    cv2.imshow("Hover and Click on Masks", image)
    cv2.setMouseCallback("Hover and Click on Masks", on_mouse)
    print("Hover over a mask to highlight; left-click to select.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()