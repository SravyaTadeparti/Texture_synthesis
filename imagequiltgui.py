import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import random

# Create the main window
root = tk.Tk()
root.title("Creative Image Gallery")
root.geometry("1200x700")  # Larger window size for better viewing
root.config(bg='#D6EAF8')  # Light blue background

# Frame to hold images, with a colorful background and some padding
frame = tk.Frame(root, bg="#AED6F1", bd=5)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

# Right frame to show selected images and processed image
right_frame = tk.Frame(root, bg="#AED6F1", bd=5)
right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

# Add a title label at the top
title_label = tk.Label(root, text="Interactive Image Gallery", font=("Arial", 24), bg="#D6EAF8", fg="#1F618D")
title_label.pack(pady=10)

# Image names (adjust the file extension if needed)
image_names = [f"texture{i}.png" for i in range(1, 6)]  # List of image file names (im1.jpg, im2.jpg, ..., im10.jpg)
# Selected images list
selected_images = []

# Frame to contain selected images (3 images max)
selected_frame = tk.Frame(right_frame, bg="#AED6F1")
selected_frame.pack(pady=10)

# Label to show selected images (3 images max)
selected_labels = [tk.Label(selected_frame, bd=2, relief="solid", padx=10, pady=10, bg="white") for _ in range(3)]
for label in selected_labels:
    label.pack(side=tk.TOP, padx=10, pady=10)

# Label to show the processed image
processed_label = tk.Label(right_frame, bd=2, relief="solid", padx=10, pady=10, bg="white")
processed_label.pack(pady=20)

# Function to enlarge an image when clicked
def enlarge_image(image):
    enlarged_window = tk.Toplevel(root)
    enlarged_window.title("Enlarged Image")
    enlarged_window.geometry("500x500")
    
    img = Image.open(image)
    img = img.resize((500, 500), Image.ANTIALIAS)  # Resize to a larger image
    img_tk = ImageTk.PhotoImage(img)
    
    label = tk.Label(enlarged_window, image=img_tk)
    label.image = img_tk  # Keep a reference to the image
    label.pack()

# Function to handle image click for selection
def select_image(img_name, label):
    global selected_images

    # If already selected, deselect
    if img_name in selected_images:
        selected_images.remove(img_name)
        label.config(bg="white")  # Reset the label background to white
        update_selected_images()  # Update the selected images view
    else:
        if len(selected_images) < 3:  # Limit selection to 3 images
            selected_images.append(img_name)
            label.config(bg="lightblue")  # Change background to indicate selection
            update_selected_images()  # Update the selected images view
        else:
            messagebox.showwarning("Selection Limit", "You can only select up to 3 images.")

# Function to update the selected images in the right frame
def update_selected_images():
    for i in range(3):
        if i < len(selected_images):
            img = Image.open(selected_images[i])
            img.thumbnail((150, 150))  # Resize to fit the box
            img_tk = ImageTk.PhotoImage(img)
            selected_labels[i].config(image=img_tk)
            selected_labels[i].image = img_tk  # Keep a reference to the image
        else:
            selected_labels[i].config(image=None)  # Clear any remaining images in the labels

# Function to extract a random patch from an image
def extract_random_patch(image, patch_size):
    """Extract a random patch from the image."""
    y = random.randint(0, image.shape[0] - patch_size)
    x = random.randint(0, image.shape[1] - patch_size)
    return image[y:y+patch_size, x:x+patch_size]

# Function to calculate the average of random patches from all images with random selections
def average_random_patch(images, patch_size):
    avg_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
    for img in images:
        if img.shape[0] < patch_size or img.shape[1] < patch_size:
            img = cv2.resize(img, (patch_size, patch_size))  # Resize to match patch size
        patch = extract_random_patch(img, patch_size)
        avg_patch += patch.astype(np.float32)
    
    avg_patch /= len(images)
    return avg_patch.astype(np.uint8)

# Function to blend two patches using Poisson Image Editing (gradient domain blending)
def poisson_blend(patch1, patch2):
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

# Function to combine multiple image patches seamlessly
def combine_images_seamlessly(images, patch_size, grid_size):
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

# Function to process the selected images
def process_images():
    # if len(selected_images) != 3:
    #     messagebox.showerror("Error", "Please select exactly 3 images.")
    #     return

    # Load the selected images
    images = [cv2.imread(img_name) for img_name in selected_images]
    
    patch_size = 64  # Size of each patch
    grid_size = (5, 5)  # Size of the grid to combine patches into a larger image

    # Combine images seamlessly using advanced techniques with random patches
    output_image = combine_images_seamlessly(images, patch_size, grid_size)

    # Convert the output image to RGB for proper display
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Convert to ImageTk for display in Tkinter
    output_img_pil = Image.fromarray(output_image_rgb)
    output_img_tk = ImageTk.PhotoImage(output_img_pil)

    # Show the output image in the processed label
    processed_label.config(image=output_img_tk)
    processed_label.image = output_img_tk  # Keep a reference to the image

# Load and display images in the grid
# Load and display images in the grid
def display_images():
    for i, img_name in enumerate(image_names):
        if os.path.exists(img_name):
            img = Image.open(img_name)
            # Resize the image to fixed dimensions
            img = img.resize((150, 150), Image.Resampling.LANCZOS)  # Ensure all images are 150x150
            img_tk = ImageTk.PhotoImage(img)
            
            label = tk.Label(frame, image=img_tk, bd=2, relief="solid", padx=10, pady=10, bg="white")
            label.image = img_tk  # Keep a reference to the image
            label.grid(row=i // 5, column=i % 5, padx=10, pady=10)

            label.bind("<Button-1>", lambda event, img_name=img_name, label=label: select_image(img_name, label))

# Display images when the application starts
display_images()

# Button to process the selected images
process_button = tk.Button(right_frame, text="Process Images", command=process_images, font=("Arial", 14), bg="#1F618D", fg="white")
process_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()