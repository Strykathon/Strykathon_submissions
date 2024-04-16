import os

train_transforms = A.Compose([
                      A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, 
                                                    sat_shift_limit=0.2, 
                                                    val_shift_limit=0.2, 
                                                    p=0.2),      
                              A.RandomBrightnessContrast(brightness_limit=0.2, 
                                                         contrast_limit=0.2, 
                                                         p=0.9)],p=0.2),
                              A.ToGray(p=0.05),
                      A.OneOf(
                              [A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.5),
                               A.RandomRotate90(p=0.5),
                               A.Transpose(p=0.5),
                              ], p=0.5),
                      
                      #A.OneOf([
                      #         A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                      #         A.GridDistortion(p=0.5),
                      #         A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                      #      ], p=0.8),
                      #A.Resize(height=INPUT_SIZE, width=INPUT_SIZE, p=1),
                      #A.Cutout(num_holes=8, max_h_size=10, max_w_size=10, fill_value=0, p=0.1),
                      ], p=1.0)


def preprocess(path, training=False, labeling=False):
    img = Image.open(path)
    img1 = img.crop((0, 0, 256, 256)).resize((INPUT_SIZE, INPUT_SIZE))
    img2 = img.crop((256, 0, 512, 256)).resize((INPUT_SIZE, INPUT_SIZE))
    img1 = np.array(img1)
    img2 = np.array(img2) 
    if labeling:
        mask = np.zeros(shape=(img2.shape[0], img2.shape[1]), dtype = np.uint32)
        for row in range(img2.shape[0]):
            for col in range(img2.shape[1]):
                a = img2[row, col, :]
                final_key = final_d = None
                for key, value in id_map.items():
                    d = np.sum(np.sqrt(pow(a - value, 2)))
                    if final_key == None:
                        final_d, final_key = d, key
                    elif d < final_d:
                        final_d, final_key = d, key
                mask[row, col] = final_key
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1)).astype(np.uint8)
    else:
        mask = img2
    if training:
        sample = train_transforms(**{"image": img1, "mask": mask})
        img1, mask = sample["image"], sample["mask"]
    del img2
    img1 = img1 / 255.
    return img1, mask

def preprocess(path):
    img = Image.open(path)
    img1 = img.crop((0, 0, 256, 256)).resize((INPUT_SIZE, INPUT_SIZE))
    img2 = img.crop((256, 0, 512, 256)).resize((INPUT_SIZE, INPUT_SIZE))
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    return img1, img2

def aug_fn(image, mask):
    data = {"image":image, "mask": mask}
    aug_data = train_transforms(**data)
    aug_img = aug_data["image"]
    aug_mask = aug_data["mask"]
    aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_mask = tf.cast(aug_mask/255.0, tf.float32)  
    return aug_img, aug_mask

def process_data(image, mask):
    aug_img, aug_mask = tf.numpy_function(func=aug_fn, inp=[image, mask], Tout=[tf.float32, tf.float32])
    return aug_img, aug_mask

def valid_aug(image, mask):
    aug_img = tf.cast(image/255.0, tf.float32)
    aug_mask = tf.cast(mask/255.0, tf.float32)
    return aug_img, aug_mask


train_folder = "/content/split_data/train/Images"
val_folder = "/content/split_data/val/Images"

# Get list of file paths for training images
train_raw_images = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.png')]

# Get list of file paths for validation images
val_raw_images = [os.path.join(val_folder, f) for f in os.listdir(val_folder) if f.endswith('.png')]


#train_raw_images = glob("../input/cityscapes-image-pairs/cityscapes_data/train/*.jpg")
#val_raw_images = glob("../input/cityscapes-image-pairs/cityscapes_data/val/*.jpg")

x_train_images = np.zeros((len(train_raw_images), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
x_train_masks = np.zeros((len(train_raw_images), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)

x_valid_images = np.zeros((len(val_raw_images), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
x_valid_masks = np.zeros((len(val_raw_images), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)

for i in trange(len(train_raw_images)):
    image, mask = preprocess(train_raw_images[i])
    x_train_images[i, ...] = image
    x_train_masks[i, ...] = mask

for i in trange(len(val_raw_images)):
    image, mask = preprocess(val_raw_images[i])
    x_valid_images[i, ...] = image
    x_valid_masks[i, ...] = mask