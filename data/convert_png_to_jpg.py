import glob
import os

from PIL import Image


if __name__ == "__main__":
    for image_path in glob.glob('cleaned_crabs/*'):
        if image_path.endswith('.png') or image_path.endswith('.PNG'):
            print(f"Converting {image_path}...")
            im = Image.open(image_path)
            try:
                rgb_im = im.convert('RGB')
                im.save(image_path[:-3] + 'JPEG', quality=95)
            except:
                print(f"Failed to convert {image_path}...")
            os.remove(image_path)
        elif not image_path.endswith('.JPEG'):
            try:
                im = Image.open(image_path)
            except OSError:
                print(f"Discarding {image_path}")
            else:
                print(f"Saving .jpg as .JPEG...")
                im.save(os.path.splitext(image_path)[0] + '.JPEG', quality=95)
            finally:
                os.remove(image_path)
