import cv2
import numpy as np

# =========================================
# LOAD IMAGE
# =========================================

img = cv2.imread("30_2.jpg")

# =========================================
# CROP AREA
# =========================================

x1 = 180
y1 = 50

x2 = 1740
y2 = 1030

crop = img[y1:y2, x1:x2]

# =========================================
# BLUR
# =========================================

blur = cv2.GaussianBlur(
    crop,
    (7,7),
    0
)

# =========================================
# HSV
# =========================================

hsv = cv2.cvtColor(
    blur,
    cv2.COLOR_BGR2HSV
)

# =========================================
# DETECT OBJECT
# =========================================
# object = bean + skin

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

# =========================================
# MORPHOLOGY OBJECT
# =========================================

kernel = np.ones((5,5), np.uint8)

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

# =========================================
# TAKE ONLY OBJECT PIXELS
# =========================================

object_pixels = blur[
    mask_object > 0
]

# =========================================
# FLOAT32
# =========================================

pixel_values = np.float32(
    object_pixels
)

# =========================================
# KMEANS
# =========================================

K = 3

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

_, labels, centers = cv2.kmeans(
    pixel_values,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

# =========================================
# CONVERT CENTER
# =========================================

centers = np.uint8(
    centers
)

# =========================================
# CENTER HSV
# =========================================

centers_bgr = np.uint8([centers])

centers_hsv = cv2.cvtColor(
    centers_bgr,
    cv2.COLOR_BGR2HSV
)[0]

# =========================================
# DETERMINE SKIN CLUSTER
# =========================================
# skin:
# low saturation
# high brightness

scores = []

for hsv_center in centers_hsv:

    h = int(hsv_center[0])
    s = int(hsv_center[1])
    v = int(hsv_center[2])

    score = (255 - s) + v

    scores.append(score)

scores = np.array(scores)

skin_cluster = np.argmax(
    scores
)

# =========================================
# EMPTY MASK
# =========================================

mask_skin = np.zeros(
    mask_object.shape,
    dtype=np.uint8
)

# =========================================
# OBJECT COORDINATES
# =========================================

coords = np.where(
    mask_object > 0
)

# =========================================
# INITIAL CLASSIFICATION
# =========================================

for i in range(len(labels)):

    y = coords[0][i]
    x = coords[1][i]

    cluster = labels[i][0]

    if cluster == skin_cluster:

        mask_skin[y, x] = 255

# =========================================
# HSV CORRECTION
# =========================================

for y in range(hsv.shape[0]):

    for x in range(hsv.shape[1]):

        # skip background
        if mask_object[y, x] == 0:
            continue

        h = int(hsv[y, x][0])
        s = int(hsv[y, x][1])
        v = int(hsv[y, x][2])

        # =================================
        # STRONG SKIN RULE
        # =================================

        if s < 35 and v > 215:

            mask_skin[y, x] = 255

        # =================================
        # STRONG BEAN RULE
        # =================================

        elif s > 60:

            mask_skin[y, x] = 0

# =========================================
# MORPHOLOGY SKIN
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
# REMOVE SMALL NOISE
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

    if area > 20:

        clean_skin[
            labels_img == i
        ] = 255

mask_skin = clean_skin

# =========================================
# REBUILD BEAN MASK
# =========================================
# bean = object - skin

mask_bean = cv2.subtract(
    mask_object,
    mask_skin
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
print("FINAL KMEANS RESULT")
print("=================================")

print("Centers BGR:")
print(centers)

print("---------------------------------")

print("Centers HSV:")
print(centers_hsv)

print("---------------------------------")

print(f"Skin Cluster      : {skin_cluster}")

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
    "Mask Object",
    resize_img(mask_object)
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
    "Mask Background",
    resize_img(mask_background)
)

cv2.imshow(
    "Final Overlay",
    resize_img(overlay)
)

cv2.waitKey(0)
cv2.destroyAllWindows()