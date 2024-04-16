import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the image using Matplotlib
img = "/content/drive/MyDrive/sample_data/Labels/aachen-mask1/aachen_000000_000019_gtFine_color.png"
img_data = mpimg.imread(img)
img_data.shape

img_dir = '/content/drive/MyDrive/sample_data/Images/aachen'
mask_dir = '/content/drive/MyDrive/sample_data/Labels/aachen-mask1'

img_files = os.listdir(img_dir)

# Display images along with their corresponding masks
for img_file in img_files:

    img_path = os.path.join(img_dir, img_file)
    img = mpimg.imread(img_path)

    # Construct the corresponding mask filename
    mask_file = img_file.replace('_leftImg8bit.png', '_gtFine_color.png')
    mask_path = os.path.join(mask_dir, mask_file)

    mask = mpimg.imread(mask_path)

    # Plot the image and mask side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title('Image')
    axes[0].axis('off')
    axes[1].imshow(mask)
    axes[1].set_title('Mask')
    axes[1].axis('off')
    plt.show()