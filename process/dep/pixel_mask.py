import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_path = os.path.dirname(__file__)

# Load the image
image_path = f'{base_path}/train/positive/lantana camara/lantana camara_17.jpg'  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found or unable to load.")

# Convert to HSV color space (better for color-based segmentation)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for leaves and petals in HSV
# Healthy green leaves
lower_green = np.array([35, 40, 40])  # Hue: 35-85
upper_green = np.array([85, 255, 255])
# Unhealthy yellow leaves
lower_yellow_leaf = np.array([20, 40, 40])  # Hue: 20-35
upper_yellow_leaf = np.array([35, 255, 255])
# Unhealthy brown leaves
lower_brown = np.array([10, 40, 40])  # Hue: 10-20
upper_brown = np.array([20, 255, 255])
# Petals (e.g., red/pink petals, adjust based on your flower)
lower_petal = np.array([0, 40, 40])  # Hue: 0-10 or 160-180 for red/pink
upper_petal = np.array([10, 255, 255])
lower_petal2 = np.array([160, 40, 40])  # Second range for red/pink
upper_petal2 = np.array([180, 255, 255])

# Create masks for each component
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_yellow_leaf = cv2.inRange(hsv_image, lower_yellow_leaf, upper_yellow_leaf)
mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
mask_petal1 = cv2.inRange(hsv_image, lower_petal, upper_petal)
mask_petal2 = cv2.inRange(hsv_image, lower_petal2, upper_petal2)

# Combine petal masks (for colors spanning hue 0-10 and 160-180)
mask_petal = cv2.bitwise_or(mask_petal1, mask_petal2)

# Combine all masks (leaves + petals)
combined_mask = cv2.bitwise_or(mask_green, mask_yellow_leaf)
combined_mask = cv2.bitwise_or(combined_mask, mask_brown)
combined_mask = cv2.bitwise_or(combined_mask, mask_petal)

# Apply morphological operations to remove noise and fill gaps
kernel = np.ones((5, 5), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove small noise
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill small gaps

# Find contours in the combined mask
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image (copy to avoid modifying the original)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)  # Red contours, thickness 2

# Apply the combined mask to extract leaves and petals
plant_extracted = cv2.bitwise_and(image, image, mask=combined_mask)

# Convert images for display (OpenCV uses BGR, Matplotlib uses RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_with_contours_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
plant_extracted_rgb = cv2.cvtColor(plant_extracted, cv2.COLOR_BGR2RGB)

# Display the results
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Combined Mask")
plt.imshow(combined_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Plant with Contours")
plt.imshow(image_with_contours_rgb)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Extracted Leaves and Petals")
plt.imshow(plant_extracted_rgb)
plt.axis('off')

plt.show()

# Optional: Save the results
# cv2.imwrite('plant_with_contours.jpg', image_with_contours)
# cv2.imwrite('extracted_plant.jpg', plant_extracted)