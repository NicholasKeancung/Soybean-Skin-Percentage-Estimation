import cv2
import numpy as np
from ultralytics import YOLO

# =========================================
# LOAD MODEL
# =========================================

model = YOLO(
    r"D:\RisetMesinKedelai\Computer Vision\OpenCv\best.pt"
)

# =========================================
# CLASS CONFIDENCE
# =========================================

CONF_BEAN = 0.03
CONF_SKIN = 0.70

# =========================================
# MASK THRESHOLD
# =========================================

MASK_THRESHOLD = 0.75

# =========================================
# LOAD IMAGE
# =========================================

img = cv2.imread("30_2.jpg")

# =========================================
# CHECK IMAGE
# =========================================

if img is None:

    print("Image not found")
    exit()

# =========================================
# CROP AREA
# =========================================

x1 = 180
y1 = 50

x2 = 1740
y2 = 1030

crop = img[y1:y2, x1:x2]

# =========================================
# YOLO INFERENCE
# =========================================

results = model.predict(
    source=crop,
    conf=0.05,
    imgsz=1280,
    rect=True,
    retina_masks=True,
    half=True,
    device=0,
    verbose=False
)

# =========================================
# GET RESULT
# =========================================

result = results[0]

# =========================================
# EMPTY MASK
# =========================================

mask_skin = np.zeros(
    crop.shape[:2],
    dtype=np.uint8
)

mask_bean = np.zeros(
    crop.shape[:2],
    dtype=np.uint8
)

mask_object = np.zeros(
    crop.shape[:2],
    dtype=np.uint8
)

# =========================================
# CLASS NAMES
# =========================================

names = model.names

# =========================================
# PROCESS SEGMENTATION
# =========================================

if result.masks is not None:

    masks = result.masks.data.cpu().numpy()

    classes = result.boxes.cls.cpu().numpy()

    confidences = result.boxes.conf.cpu().numpy()

    for mask, cls_id, conf in zip(
        masks,
        classes,
        confidences
    ):

        cls_id = int(cls_id)

        class_name = names[cls_id]

        # =================================
        # FILTER CONFIDENCE PER CLASS
        # =================================

        if class_name == "bean":

            if conf < CONF_BEAN:
                continue

        elif class_name == "skin":

            if conf < CONF_SKIN:
                continue

        else:

            continue

        # =================================
        # RESIZE MASK
        # =================================

        mask = cv2.resize(
            mask,
            (
                crop.shape[1],
                crop.shape[0]
            )
        )

        # =================================
        # BINARY MASK
        # =================================

        binary_mask = (
            mask > MASK_THRESHOLD
        ).astype(np.uint8) * 255

        # =================================
        # OBJECT MASK
        # =================================

        mask_object = cv2.bitwise_or(
            mask_object,
            binary_mask
        )

        # =================================
        # SKIN
        # =================================

        if class_name == "skin":

            mask_skin = cv2.bitwise_or(
                mask_skin,
                binary_mask
            )

        # =================================
        # BEAN
        # =================================

        elif class_name == "bean":

            mask_bean = cv2.bitwise_or(
                mask_bean,
                binary_mask
            )

# =========================================
# MORPHOLOGY
# =========================================

kernel = np.ones((3,3), np.uint8)

# =========================================
# SKIN MORPHOLOGY
# =========================================

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

# =========================================
# SKIN EROSION
# =========================================

mask_skin = cv2.erode(
    mask_skin,
    kernel,
    iterations=1
)

# =========================================
# BEAN MORPHOLOGY
# =========================================

mask_bean = cv2.morphologyEx(
    mask_bean,
    cv2.MORPH_OPEN,
    kernel
)

mask_bean = cv2.morphologyEx(
    mask_bean,
    cv2.MORPH_CLOSE,
    kernel
)

# =========================================
# REMOVE OVERLAP
# =========================================
# bean diprioritaskan

mask_skin = cv2.subtract(
    mask_skin,
    mask_bean
)

# =========================================
# REMOVE SMALL SKIN NOISE
# =========================================

num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(
    mask_skin,
    8
)

clean_skin = np.zeros_like(
    mask_skin
)

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area > 100:

        clean_skin[
            labels_img == i
        ] = 255

mask_skin = clean_skin

# =========================================
# REBUILD OBJECT
# =========================================

mask_object = cv2.bitwise_or(
    mask_skin,
    mask_bean
)

# =========================================
# BACKGROUND
# =========================================

mask_background = cv2.bitwise_not(
    mask_object
)

# =========================================
# OVERLAY
# =========================================

overlay = crop.copy()

# background = black
overlay[mask_background > 0] = (0,0,0)

# bean = green
overlay[mask_bean > 0] = (0,255,0)

# skin = red
overlay[mask_skin > 0] = (0,0,255)

# =========================================
# PIXEL COUNT
# =========================================

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

# =========================================
# PERCENTAGE
# =========================================
skin_percentage = (
    0.0008 * skin_pixels
) - 2.6368

skin_percentage = max(
    0,
    min(100, skin_percentage)
)

bean_percentage = 100 - skin_percentage

# =========================================
# PRINT RESULT
# =========================================

print("=================================")
print("YOLO11m SEG RESULT")
print("=================================")

print(f"CONF BEAN         : {CONF_BEAN}")
print(f"CONF SKIN         : {CONF_SKIN}")
print(f"MASK THRESHOLD    : {MASK_THRESHOLD}")

print("---------------------------------")

print(f"Skin Pixels       : {skin_pixels}")
print(f"Bean Pixels       : {bean_pixels}")
print(f"Object Pixels     : {object_pixels}")
print(f"Background Pixels : {background_pixels}")

print("---------------------------------")

print(f"Skin Percentage   : {skin_percentage:.2f}%")
print(f"Bean Percentage   : {bean_percentage:.2f}%")

print("---------------------------------")

print(
    f"Total Percentage  : "
    f"{skin_percentage + bean_percentage:.2f}%"
)

print("=================================")

# =========================================
# DISPLAY SCALE
# =========================================

scale = 0.6

# =========================================
# RESIZE FUNCTION
# =========================================

def resize_img(image):

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale
    )

# =========================================
# SHOW WINDOWS
# =========================================

cv2.imshow(
    "Original Crop",
    resize_img(crop)
)

cv2.imshow(
    "Mask Skin",
    resize_img(mask_skin)
)

cv2.imshow(
    "Mask Bean",
    resize_img(mask_bean)
)

cv2.imshow(
    "Mask Object",
    resize_img(mask_object)
)

cv2.imshow(
    "Mask Background",
    resize_img(mask_background)
)

cv2.imshow(
    "Final Overlay",
    resize_img(overlay)
)

cv2.waitKey(0)
cv2.destroyAllWindows()