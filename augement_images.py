import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# Folders
DATASET_DIR = "dataset"
AUGMENTED_DIR = "dataset_augmented"

# Create augmented dataset folder if not exists
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

def augment_images_for_class(class_name, num_augmented=150):
    input_dir = os.path.join(DATASET_DIR, class_name)
    output_dir = os.path.join(AUGMENTED_DIR, class_name)
    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(input_dir)
    total_generated = 0

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        prefix = os.path.splitext(img_name)[0]

        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=output_dir,
                                  save_prefix=prefix,
                                  save_format='jpg'):
            i += 1
            total_generated += 1
            if i >= (num_augmented // len(images)):
                break

    print(f"Augmented {total_generated} images for {class_name}")

# Example usage
# augment_images_for_class('bottle')
# augment_images_for_class('phone')
augment_images_for_class('background')
