import torch
from einops import rearrange, einsum
import matplotlib.pyplot as plt


images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
## Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value
## Or in one go:
dimmed_images = einsum(
images, dim_by,
"batch height width channel, dim_value -> batch dim_value height width channel"
)

print("images:", images.shape)
print("dim_value:", dim_value.shape)
print("images_rearr:", images_rearr.shape)
print("dimmed_images:", dimmed_images.shape)

# Inspect the first image and its first dimmed version
print(dimmed_images[0, 0, :5, :5, 0])  # first 5x5 patch of channel 0
print(dimmed_images[0, 1, :5, :5, 0])  # scaled by next dim_value

# Select the first image and all 10 dimmed versions
plt.figure(figsize=(4, 4))

for img_idx in range(64):            # Loop through all images
    for b_idx in range(10):          # Loop through all 10 brightness levels
        plt.clf()                    # Clear previous frame
        plt.imshow(dimmed_images[img_idx, b_idx].numpy())
        plt.title(f"Image {img_idx}  |  Brightness {b_idx}")
        plt.axis("off")
        plt.pause(0.01)              # ~14 fps fast movie

plt.close()