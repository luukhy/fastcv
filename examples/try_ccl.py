import numpy as np
import torch
import fastcv

img = np.array([[0, 0, 0, 0],
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 1] ], dtype=np.uint8)

print(img)

img_tensor = torch.tensor(img.tolist(), dtype=torch.uint8, device='cuda') 
print(fastcv.connectedComponents(img_tensor))

