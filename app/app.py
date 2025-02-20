from image_segmentation import select_mask
from further_segmentation import segment_selected_region, create_polygon_and_filter_masks

def main():
    # Step 1: Select a mask region
    original_image_path = "/Users/mikan/Downloads/C5.7/APAL23_AP021_2024_07_18 (17).JPG"
    
    select_mask(
        image_path=original_image_path,
        sam_checkpoint="/Users/mikan/Downloads/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        output_dir="output_masks",
        selected_mask_dir="selected_masks"
    )

    # Step 2: Further segment and refine (only for selected region)
    segment_selected_region(
        selected_mask_path="selected_masks/selected_mask.png",
        sam_checkpoint="/Users/mikan/Downloads/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        output_dir="polyps_masks"
    )

    create_polygon_and_filter_masks(
        original_image_path=original_image_path,
        sam_checkpoint="/Users/mikan/Downloads/sam_vit_b_01ec64.pth",
        model_type="vit_b",  # Ensure this is valid: "vit_b", "vit_l", or "vit_h"
        output_dir="fine_tune_masks"
    )
if __name__ == "__main__":
    main()