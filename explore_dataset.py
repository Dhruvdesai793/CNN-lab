from pathlib import Path
import cv2
import matplotlib.pyplot as plt

root = Path("data/CamVid")

image_path = sorted((root/ "train").glob("*"))[5]
mask_path = sorted((root / "train_labels").glob("*"))[5]

image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
mask = cv2.imread(str(mask_path))

print("image shape: ", image.shape)
print("mask shape: ", mask.shape)
print("mask dtype: ", mask.dtype)

plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("image")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.title("mask")

plt.show()

mask = cv2.imread(str(mask_path))
print(mask.shape)
print(mask[100, 100])