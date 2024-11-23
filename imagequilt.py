import cv2
import numpy as np
import random

def extract_random_patch(image, patch_size):
    """Extract a random patch from the image."""
    y = random.randint(0, image.shape[0] - patch_size)
    x = random.randint(0, image.shape[1] - patch_size)
    return image[y:y+patch_size, x:x+patch_size]

def average_random_patch(images, patch_size):
    """Calculate the average of random patches from all images."""
    avg_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
    
    for img in images:
        if img.shape[0] < patch_size or img.shape[1] < patch_size:
            print(f"Resizing image of shape {img.shape} to {patch_size}x{patch_size}")
            img = cv2.resize(img, (patch_size, patch_size))  
        
        patch = extract_random_patch(img, patch_size)
        avg_patch += patch.astype(np.float32)
    
    avg_patch /= len(images)  
    return avg_patch.astype(np.uint8)

def poisson_blend(patch1, patch2):
    """Blend two patches using Poisson Image Editing to minimize seams."""
    patch1 = patch1.astype(np.float32)
    patch2 = patch2.astype(np.float32)
    
    grad_x1 = cv2.Sobel(patch1, cv2.CV_32F, 1, 0, ksize=3)
    grad_y1 = cv2.Sobel(patch1, cv2.CV_32F, 0, 1, ksize=3)
    
    grad_x2 = cv2.Sobel(patch2, cv2.CV_32F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(patch2, cv2.CV_32F, 0, 1, ksize=3)
    
    blended_grad_x = (grad_x1 + grad_x2) / 2
    blended_grad_y = (grad_y1 + grad_y2) / 2
    
    blended_patch = np.zeros_like(patch1)
    for c in range(3):
        blended_patch[:, :, c] = patch1[:, :, c]
    
    return blended_patch.astype(np.uint8)

def combine_images_seamlessly(images, patch_size, grid_size):
    """Combine patches from multiple images using advanced blending techniques."""
    num_images = len(images)
    total_height = patch_size * grid_size[0]
    total_width = patch_size * grid_size[1]
    
    output_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    current_x = 0
    current_y = 0

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            patch = average_random_patch(images, patch_size)
            
            if i == 0 and j == 0:
                output_image[current_y:current_y+patch_size, current_x:current_x+patch_size] = patch
            else:
                if j > 0:
                    left_patch = output_image[current_y:current_y+patch_size, current_x-patch_size:current_x]
                    patch = poisson_blend(left_patch, patch)
                
                if i > 0:
                    top_patch = output_image[current_y-patch_size:current_y, current_x:current_x+patch_size]
                    patch = poisson_blend(top_patch, patch)

                output_image[current_y:current_y+patch_size, current_x:current_x+patch_size] = patch

            current_x += patch_size

        current_y += patch_size
        current_x = 0

    return output_image

image_paths =  ['paper_texture.png', 'texture2.png']
images = [cv2.imread(image_path) for image_path in image_paths]

for i, img in enumerate(images):
    if img is None:
        raise ValueError(f"Image {image_paths[i]} not loaded properly. Please check the path.")

patch_size = 128
grid_size = (5, 5)  

height, width, _ = images[0].shape
if patch_size > height or patch_size > width:
    raise ValueError(f"Patch size {patch_size} is larger than the image dimensions.")

output_image = combine_images_seamlessly(images, patch_size, grid_size)

output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

cv2.imshow('Combined Seamless Texture', output_image_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('combined_seamless_texture.jpg', output_image)
