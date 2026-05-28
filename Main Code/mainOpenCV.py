import cv2
import numpy as np

# =========================
# LOAD IMAGE
# =========================

img = cv2.imread("30_2.jpg")

# =========================
# CROP AREA
# =========================

x1 = 180
y1 = 50

x2 = 1740
y2 = 1030

crop = img[y1:y2, x1:x2]

# =========================
# BLUR IMAGE
# =========================

blur = cv2.GaussianBlur(
    crop,
    (7,7),
    0
)

# =========================
# CONVERT TO HSV
# =========================

hsv = cv2.cvtColor(
    blur,
    cv2.COLOR_BGR2HSV
)

# =========================
# KERNEL
# =========================

kernel = np.ones((5,5), np.uint8)

# =========================
# DETECT KULIT
# =========================
# kulit putih terang
lower_skin = np.array([0, 0, 200])
upper_skin = np.array([80, 58, 255])

mask_skin = cv2.inRange(
    hsv,
    lower_skin,
    upper_skin
)

# Morphology kulit
mask_skin = cv2.morphologyEx(
    mask_skin,
    cv2.MORPH_OPEN,
    kernel
)

mask_skin = cv2.morphologyEx(
    mask_skin,
    cv2.MORPH_CLOSE,
    kernel
)

# =========================
# DETECT OBJECT
# =========================
# semua biji + kulit

gray = cv2.cvtColor(
    blur,
    cv2.COLOR_BGR2GRAY
)

_, mask_object = cv2.threshold(
    gray,
    155,
    255,
    cv2.THRESH_BINARY
)

# Morphology object
mask_object = cv2.morphologyEx(
    mask_object,
    cv2.MORPH_OPEN,
    kernel
)

mask_object = cv2.morphologyEx(
    mask_object,
    cv2.MORPH_CLOSE,
    kernel
)

# =========================
# DETECT BIJI
# =========================
# biji = object - kulit

mask_bean = cv2.subtract(
    mask_object,
    mask_skin
)

# =========================
# DETECT BACKGROUND
# =========================

mask_background = cv2.bitwise_not(
    mask_object
)

# =========================
# OVERLAY COLOR
# =========================

overlay = crop.copy()

# Background = black
overlay[mask_background > 0] = (0,0,0)

# Bean = green
overlay[mask_bean > 0] = (0,255,0)

# Skin = red
overlay[mask_skin > 0] = (0,0,255)

# =========================
# PIXEL COUNT
# =========================

skin_pixels = cv2.countNonZero(
    mask_skin
)

bean_pixels = cv2.countNonZero(
    mask_bean
)

object_pixels = cv2.countNonZero(
    mask_object
)

background_pixels = cv2.countNonZero(
    mask_background
)

# =========================
# PERCENTAGE
# =========================
skin_percentage = (
    0.0008 * skin_pixels
) - 2.6368

skin_percentage = max(
    0,
    min(100, skin_percentage)
)

bean_percentage = 100 - skin_percentage


# =========================
# PRINT RESULT
# =========================

print("=================================")
print("RESULT")
print("=================================")

print(f"Skin Pixels       : {skin_pixels}")
print(f"Bean Pixels       : {bean_pixels}")
print(f"Object Pixels     : {object_pixels}")
print(f"Background Pixels : {background_pixels}")

print("---------------------------------")

print(f"Skin Percentage   : {skin_percentage:.2f}%")
print(f"Bean Percentage   : {bean_percentage:.2f}%")

print("=================================")

# =========================
# DISPLAY SCALE
# =========================

scale = 0.6

# =========================
# RESIZE DISPLAY
# =========================

crop_preview = cv2.resize(
    crop,
    None,
    fx=scale,
    fy=scale
)

mask_skin_preview = cv2.resize(
    mask_skin,
    None,
    fx=scale,
    fy=scale
)

mask_object_preview = cv2.resize(
    mask_object,
    None,
    fx=scale,
    fy=scale
)

mask_bean_preview = cv2.resize(
    mask_bean,
    None,
    fx=scale,
    fy=scale
)

mask_background_preview = cv2.resize(
    mask_background,
    None,
    fx=scale,
    fy=scale
)

overlay_preview = cv2.resize(
    overlay,
    None,
    fx=scale,
    fy=scale
)

# =========================
# SHOW WINDOWS
# =========================

cv2.imshow(
    "Original Crop",
    crop_preview
)

cv2.imshow(
    "Mask Skin",
    mask_skin_preview
)

cv2.imshow(
    "Mask Object",
    mask_object_preview
)

cv2.imshow(
    "Mask Bean",
    mask_bean_preview
)

cv2.imshow(
    "Mask Background",
    mask_background_preview
)

cv2.imshow(
    "Final Overlay",
    overlay_preview
)

cv2.waitKey(0)
cv2.destroyAllWindows()