import glob
import os

from PIL import Image


if __name__ == "__main__":
    for image_path in glob.glob('tiny-imagenet-crabs/cleaned_crabs/*'):
        if image_path.endswith('.png') or image_path.endswith('.PNG'):
            print(f"Converting {image_path}...")
            im = Image.open(image_path)
            try:
                rgb_im = im.convert('RGB')
                im.save(image_path[:-3] + 'jpg', quality=95)
            except:
                print(f"Failed to convert {image_path}...")
            os.remove(image_path)
