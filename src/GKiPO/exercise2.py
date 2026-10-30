#!/usr/bin/env python3
import io
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import requests

# 1. Load an image from the internet
#image_url = "https://picsum.photos/400/300" # gets random image every time
image_url = "https://photographylife.com/wp-content/uploads/2013/02/Underexposure.jpg"
#image_url = "https://photographylife.com/wp-content/uploads/2013/02/Overexposure.jpg"
response = requests.get(image_url)
image = Image.open(io.BytesIO(response.content)).convert("RGB")
img_np = np.array(image)

# 2. Set up the plotting layout using GridSpec
# We create a figure with 3 rows:
# Row 0: Original Image (spanning across all columns)
# Rows 1 & 2: The 4 histogram subplots (2x2 grid)
fig = plt.figure(figsize=(12.0, 10.0))
gs = fig.add_gridspec(3, 2)

# --- Plot Original Image ---
ax_img = fig.add_subplot(gs[0, :])
ax_img.imshow(image)
ax_img.set_title("Original Image", fontsize=14, fontweight="bold")
ax_img.axis("off")  # Hide axis ticks for the image

# --- Helper function for histograms ---
gray_image = image.convert("L")
gray_np = np.array(gray_image)

# --- Plot Whole Image (Grayscale / Luminance) ---
ax_gray = fig.add_subplot(gs[1, 0])
ax_gray.hist(gray_np.ravel(), bins=256, color="black", alpha=0.7)
ax_gray.set_title("Whole Image (Luminance)")
ax_gray.set_xlim((0, 256))

# --- Plot Red Channel ---
ax_r = fig.add_subplot(gs[1, 1])
ax_r.hist(img_np[:, :, 0].ravel(), bins=256, color="red", alpha=0.7)
ax_r.set_title("Red Channel")
ax_r.set_xlim((0, 256))

# --- Plot Green Channel ---
ax_g = fig.add_subplot(gs[2, 0])
ax_g.hist(img_np[:, :, 1].ravel(), bins=256, color="green", alpha=0.7)
ax_g.set_title("Green Channel")
ax_g.set_xlim((0, 256))

# --- Plot Blue Channel ---
ax_b = fig.add_subplot(gs[2, 1])
ax_b.hist(img_np[:, :, 2].ravel(), bins=256, color="blue", alpha=0.7)
ax_b.set_title("Blue Channel")
ax_b.set_xlim((0, 256))

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# 3. Evaluate image quality
def evaluate_image_quality(image_np):
  """Estimates image quality based on histogram distribution metrics.

  Parameters:
      image_np (numpy.ndarray): 2D (grayscale) or 3D (RGB) image array.

  Returns:
      dict: Quality score and diagnostic metrics.
  """
  # Convert RGB to grayscale luminance if it's a 3-channel image
  if len(image_np.shape) == 3:
    gray = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140])
  else:
    gray = image_np

  total_pixels = gray.size

  # 1. Calculate Overexposure (Highlight Clipping near 255)
  overexposed_pixels = np.sum(gray > 250)
  overexposure_ratio = overexposed_pixels / total_pixels

  # 2. Calculate Underexposure (Shadow Clipping near 0)
  underexposed_pixels = np.sum(gray < 5)
  underexposure_ratio = underexposed_pixels / total_pixels

  # 3. Calculate Dynamic Range / Contrast using percentiles
  p5, p95 = np.percentile(gray, (5, 95))
  dynamic_range = p95 - p5

  # --- Heuristic Scoring (0 to 100) ---
  # Ideal dynamic range is close to 200+ out of 255
  contrast_score = min(100, (dynamic_range / 200.0) * 100)

  # Penalize heavy clipping (over/underexposure)
  total_clipping = overexposure_ratio + underexposure_ratio
  clipping_penalty = total_clipping * 100

  # Combine into a final score
  quality_score = max(0, min(100, contrast_score - clipping_penalty))

  return {
      "quality_score": round(quality_score, 2),
      "overexposure_pct": round(overexposure_ratio * 100, 2),
      "underexposure_pct": round(underexposure_ratio * 100, 2),
      "dynamic_range": round(dynamic_range, 2),
  }

# Run evaluation
results = evaluate_image_quality(img_np)
print("Image Quality Assessment:")
for key, value in results.items():
  print(f" - {key.replace('_', ' ').title()}: {value}")

# 4. Improve image quality
def enhance_image_quality(img):
  """Improves image quality using contrast stretching, sharpness,

  and color adjustments.
  """
  # 1. Automatic Contrast Stretching
  # Stretches the pixel intensity distribution to use the full dynamic range
  enhanced = ImageOps.autocontrast(img, cutoff=2)

  # 2. Edge Sharpening
  # Enhances clarity by defining edges more clearly (factor > 1.0)
  sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
  sharpened = sharpness_enhancer.enhance(1.5)

  # 3. Color Saturation Adjustment
  # Boosts color vibrancy slightly for a more vivid look (factor > 1.0)
  color_enhancer = ImageEnhance.Color(sharpened)
  final_image = color_enhancer.enhance(1.2)

  return final_image

# Apply the enhancement algorithm
improved_image = enhance_image_quality(image)

# 5. Display both images side-by-side in a matplotlib plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Original Image
axes[0].imshow(image)
axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
axes[0].axis("off")

# Plot Enhanced Image
axes[1].imshow(improved_image)
axes[1].set_title("Enhanced Image", fontsize=14, fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.show()