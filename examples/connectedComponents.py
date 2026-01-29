import cv2
import torch
import fastcv
import numpy as np

img = cv2.imread("../artifacts/binary.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = (np.random.rand(512, 512) > 0.8).astype(np.uint8) * 255
else:
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # 

n_cv2_impl, _ = cv2.connectedComponents(img, connectivity=4)
print(f"cv2 implementation found { n_cv2_impl } components")

img_tensor = torch.from_numpy(img).cuda()
labels_tensor = fastcv.connectedComponents(img_tensor)

labels_np = labels_tensor.cpu().numpy().astype(np.int32)
unique_labels = np.unique(labels_np)
num_labels = len(unique_labels)

max_label = labels_np.max()
if max_label > 0:
    vis_img = (labels_np.astype(float) / max_label * 255).astype(np.uint8)
    color_labels = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    
    color_labels[labels_np == 0] = [0, 0, 0]
else:
    color_labels = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

cv2.imwrite("output_components.jpg", color_labels)
print(f"Saved visualization, fastcv found {num_labels} components")

cv2.imshow("Original", img)
cv2.imshow("Connected Components", color_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()