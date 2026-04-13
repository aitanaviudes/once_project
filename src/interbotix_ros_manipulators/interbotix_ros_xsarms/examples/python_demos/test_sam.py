import os
import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor

def mobile_sam_segmap_function(image_rgb, point_x=300, point_y=200):
    # --- 1. Load the Model ---
    model_type = "vit_t"
    # Updated path since you put the weights in a subfolder
    sam_checkpoint = "./mobile_sam_weights/mobile_sam.pt" 
    device = "cpu"

    print("Loading MobileSAM...")
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)

    # --- 2. Prepare the Image ---
    # (cv2.imread removed since image_rgb is passed directly)
    predictor.set_image(image_rgb)

    # --- 3. Provide Your Point ---
    input_point = np.array([[point_x, point_y]])
    input_label = np.array([1])

    # --- 4. Generate the Mask ---
    print("Generating mask...")
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False, # Set to False to get the single best mask
    )

    # Extract the best mask
    mask = masks[0]

    # --- 5. Visualize and Save the Result ---
    # Ensure the debug folder exists
    debug_dir = "debug_segmap_images"
    os.makedirs(debug_dir, exist_ok=True)

    # Create a transparent blue overlay for the mask
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Plot the original image, the mask, and a red star
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.imshow(mask_image)
    plt.plot(point_x, point_y, marker='*', color='red', markersize=15)
    plt.axis('off')

    # Save the final image with a timestamp so it doesn't overwrite
    timestamp = int(time.time())
    output_name = f"{debug_dir}/segmented_output_{timestamp}.png"
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
    plt.close() # CRITICAL: Frees up memory after saving!
    
    print(f"Success! Check '{output_name}'")

    # --- 6. Return the boolean mask ---
    return mask.astype(bool)