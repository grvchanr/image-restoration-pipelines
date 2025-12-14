"""
Simple sketch generation using edge detection.
"""

import cv2
import matplotlib.pyplot as plt
import os

# Read input image
img = cv2.imread("input.png")
if img is None:
    raise FileNotFoundError("input.png not found")

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Edge detection
edges = cv2.Canny(blurred, 50, 150)

# Invert edges to create sketch effect
sketch = cv2.bitwise_not(edges)

# Show results
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sketch, cmap="gray")
plt.title("Sketch")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save output
output_dir = os.path.join(os.path.dirname(__file__), "/Users/grvchanr/Code/image_restoration/results")
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(os.path.join(output_dir, "sketch.png"), sketch)
