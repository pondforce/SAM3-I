from pycocotools import mask as maskUtils
import numpy as np
import cv2
import json

# mask (grayscale)
mask = cv2.imread(r"C:\Users\MSI-TITAN\Documents\GitHub\SAM3-I\sam3\sam3\train\decoded_mask.png", 0)

# binary (0/1)
mask = (mask > 0).astype(np.uint8)

# many marks
# final_mask = mask1 + mask2 + mask3
# final_mask = (final_mask > 0).astype(np.uint8)

# Fortran order
mask_fortran = np.asfortranarray(mask)

# encode to RLE
rle = maskUtils.encode(mask_fortran)

# counts (bytes to string)
rle["counts"] = rle["counts"].decode("utf-8")


# [x_min, y_min, width, height]
# iscrowd : 0 เดี่ยว
# iscrowd : 1 group
# num_original_targets - SAM generate mask มีกี่ชิ้น
# area - พื้นที่ที่mark
area = mask.sum()
# output
annotation = {
    "image": "0001.jpg",
    "text": "segment the crack",
    "segmentation": {
        "counts": rle["counts"],
        "size": rle["size"],
        "area": area
    }
}

# print(json.dumps(annotation, indent=2))
print(annotation)