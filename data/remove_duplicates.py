import argparse
import glob
import shutil
import os

import imagehash
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str)
    args = parser.parse_args()
    table = {}

    filetypes = ["JPG", "jpg", "jpeg", "JPEG", "png", "PNG"]
    new_paths = []
    for filetype in filetypes:
        new_paths.extend(glob.glob(args.path + f"/*/*.{filetype}"))

    keep_path = os.path.join(args.path, "keep")
    if not os.path.exists(keep_path):
        os.makedirs(keep_path)
    toss_path = os.path.join(args.path, "toss")
    if not os.path.exists(toss_path):
        os.makedirs(toss_path)

    known_paths = []
    for filetype in filetypes:
        known_paths.extend(glob.glob(keep_path + f"/*/*.{filetype}"))

    # fill table with hashes of all images we've already kept
    for known_path in known_paths:
        image = Image.open(known_path)
        hash = str(imagehash.dhash(image))
        table[hash] = 1


    # move unique images into keep directory
    for path in new_paths:
        try:
            image = Image.open(path)
        except:
            print(f"Error opening {path}")
            continue
        hash = str(imagehash.dhash(image))
        try:
            table[hash]
        except KeyError:
            table[hash] = 1
            try:
                shutil.move(path, os.path.join(args.path, 'keep'))
            except shutil.Error:
                continue
        else:
            print(f"Tossing out {path}")
            try:
                shutil.move(path, os.path.join(args.path, 'toss'))
            except shutil.Error:
                continue



