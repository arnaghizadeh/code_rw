# reading TIFF is tricky! 1- we can conver it by diving to 255 only for the first channel, the other channels are invisible 2- if we map it, other channels become visible but this needs upper bound and each upper bound is different from the others
# this can create problems for the data, this could be ok to separate the images between boundary and non boundary but not in actual training of images.
# to detect we have to the converted images but later on to get thei real values we have to use the real forms of the images not the converted ones

import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import natsort
import tifffile as tiff
import statistics
import colorsys
from skimage import io
from skimage.transform import resize, rescale
from nd2reader import ND2Reader


def shownp(arr):
    Image.fromarray(arr).show()


def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    if not (0 <= lower_bound < 2 ** 16) and lower_bound is not None:
        raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
    if not (0 <= upper_bound < 2 ** 16) and upper_bound is not None:
        raise ValueError(
            '"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError(
            '"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2 ** 16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


class ImageIncellTIFF(Dataset):  # it is img, channel and z number
    def __init__(self, main_dir, ch_max=5, z_max=5):
        self.main_dir = main_dir
        self.ch_max = ch_max
        self.z_max = z_max
        self.image_paths = self.get_file_names()

    def get_file_names(self):
        valid_images = [".jpg", ".gif", ".png", ".tga", ".tif"]

        names = []
        names_ch = []
        names_z = []
        path = self.main_dir
        walk = natsort.natsorted(os.listdir(path))
        counter = 0
        for f in walk:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            names_z.append(f)
            counter += 1

            if counter % self.z_max == 0:
                names_ch.append(names_z)
                names_z = []
            if len(names_ch) > 0 and len(names_ch) % self.ch_max == 0:
                # print("f", names_ch)
                names.append(names_ch)
                names_ch = []

        return names

    def getitem_with_ub(self, idx=0, size=256):
        imgs = []
        for ch in range(self.ch_max):
            imgs_z = []
            for z in range(self.z_max):
                # print(self.main_dir,self.image_paths_lvl1[idx],self.image_paths_lvl2[idx][ch][z])
                image_path = os.path.join(self.main_dir, self.image_paths[idx][ch][
                    z])  # os.path.join(self.main_dir, self.image_paths_lvl1[idx], self.image_paths_lvl2[idx][ch][z])  # self.image_paths[ch][z])
                # print(image_path,self.main_dir, self.image_paths[ch][z])
                x = tiff.imread(image_path)
                upper_bound = np.max(x)
                x = map_uint16_to_uint8(x, lower_bound=0, upper_bound=upper_bound)
                # x = Image.fromarray(x)
                # x = x.resize((size, size))
                imgs_z.append(x)
            imgs.append(imgs_z)
        # x.show()
        imgs = np.asarray(imgs)
        return imgs

    def getitem_no_ub(self, idx=0, size=256):
        # print(self.main_dir, self.image_paths_lvl1[idx],len(self.image_paths_lvl1))
        imgs = []
        for ch in range(self.ch_max):
            imgs_z = []
            for z in range(self.z_max):
                image_path = os.path.join(self.main_dir, self.image_paths[idx][ch][
                    z])  # os.path.join(self.main_dir, self.image_paths_lvl1[idx], self.image_paths_lvl2[idx][ch][z])  # self.image_paths[ch][z])
                """print("idx:", self.image_paths[idx])
                print("idxch:", self.image_paths[idx][ch])
                print("idxchz:", self.image_paths[idx][ch][z])
                print(image_path,self.image_paths,self.image_paths[idx][ch][z],self.image_paths[idx][ch])
                sys.exit(1)"""
                x = tiff.imread(image_path)
                # x = np.dstack([x, x, x])
                imgs_z.append(x)
            imgs.append(imgs_z)
        imgs = np.asarray(imgs)
        return imgs

    def __getitem__(self, index):
        x = self.getitem_no_ub(idx=index)
        return x

    def __len__(self):
        return len(self.image_paths)


class ImageNikonTIFF(Dataset):  # it is img, channel and z number
    def __init__(self, main_dir, ch_max=5, z_max=5):
        self.main_dir = main_dir
        self.ch_max = ch_max
        self.z_max = z_max
        self.image_paths = self.get_file_names()

    def get_file_names(self):
        valid_images = [".jpg", ".gif", ".png", ".tga", ".tif"]
        names = []
        path = self.main_dir
        walk = natsort.natsorted(os.listdir(path))

        num = len(walk) // (self.ch_max * self.z_max)
        idx = 0
        for i in range(num):
            start_z = 0
            names_ch = []
            for ch in range(self.ch_max):
                names_z = []
                for z in range(self.z_max):
                    # print(idx+start_z+z, end=" ")
                    sel = idx + start_z + z
                    names_z.append(walk[sel])
                    start_z += self.z_max - 1
                # print()
                names_ch.append(names_z)
                start_z = ch + 1
            idx += self.ch_max * self.z_max
            names.append(names_ch)
        return names

    def getitem_with_ub(self, idx=0, size=256):
        imgs = []
        for ch in range(self.ch_max):
            imgs_z = []
            for z in range(self.z_max):
                # print(self.main_dir,self.image_paths_lvl1[idx],self.image_paths_lvl2[idx][ch][z])
                image_path = os.path.join(self.main_dir, self.image_paths[idx][ch][
                    z])  # os.path.join(self.main_dir, self.image_paths_lvl1[idx], self.image_paths_lvl2[idx][ch][z])  # self.image_paths[ch][z])
                # print(image_path,self.main_dir, self.image_paths[ch][z])
                x = tiff.imread(image_path)
                upper_bound = np.max(x)
                x = map_uint16_to_uint8(x, lower_bound=0, upper_bound=upper_bound)
                # x = Image.fromarray(x)
                # x = x.resize((size, size))
                imgs_z.append(x)
            imgs.append(imgs_z)
        # x.show()
        imgs = np.asarray(imgs)
        return imgs

    def getitem_no_ub(self, idx=0, size=256):
        # print(self.main_dir, self.image_paths_lvl1[idx],len(self.image_paths_lvl1))
        imgs = []
        for ch in range(self.ch_max):
            imgs_z = []
            for z in range(self.z_max):
                image_path = os.path.join(self.main_dir, self.image_paths[idx][ch][
                    z])  # os.path.join(self.main_dir, self.image_paths_lvl1[idx], self.image_paths_lvl2[idx][ch][z])  # self.image_paths[ch][z])
                """print("idx:", self.image_paths[idx])
                print("idxch:", self.image_paths[idx][ch])
                print("idxchz:", self.image_paths[idx][ch][z])
                print(image_path,self.image_paths,self.image_paths[idx][ch][z],self.image_paths[idx][ch])
                sys.exit(1)"""
                x = tiff.imread(image_path)
                # x = np.dstack([x, x, x])
                imgs_z.append(x)
            imgs.append(imgs_z)
        imgs = np.asarray(imgs)
        return imgs

    def __getitem__(self, index):
        x = self.getitem_no_ub(idx=index)
        return x

    def __len__(self):
        return len(self.image_paths)


class ImageNikonND2(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.image_paths = self.get_file_names()

    def get_file_names(self):
        valid_images = [".jpg", ".gif", ".png", ".tga", ".tif", ".nd2"]
        names = []
        path = self.main_dir
        walk = natsort.natsorted(os.listdir(path))
        for f in walk:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            names.append(f)
        return names

    def getitem_with_ub(self, idx=0):
        file = os.path.join(self.main_dir, self.image_paths[idx])
        with ND2Reader(file) as images:
            imgs = [[] for c in range(images.sizes['c'])]
            for c in range(images.sizes['c']):
                for z in range(images.sizes['z']):
                    image = images.get_frame_2D(c=c, t=0, z=z)
                    upper_bound = np.max(image)
                    image = map_uint16_to_uint8(image, lower_bound=0, upper_bound=upper_bound)
                    imgs[c].append(image)
        imgs = np.asarray(imgs)
        return imgs

    def getitem_no_ub(self, idx=0):
        file = os.path.join(self.main_dir, self.image_paths[idx])
        with ND2Reader(file) as images:
            imgs = [[] for c in range(images.sizes['c'])]
            for c in range(images.sizes['c']):
                for z in range(images.sizes['z']):
                    image = images.get_frame_2D(c=c, t=0, z=z)
                    imgs[c].append(image)
        imgs = np.asarray(imgs)
        return imgs

    def __getitem__(self, index):
        x = self.getitem_no_ub(idx=index)
        return x

    def __len__(self):
        return len(self.image_paths)

"""
file = '/media/an499/Liu-Lab/Bryan Tsao/CAR T 10ng strep488 split/p3 car t_10ng_kappa_biotin_af488-strep_xy01.nd2'
with ND2Reader(file) as images:
    imgs = [[] for c in range(images.sizes['c'])]
    for c in range(images.sizes['c']):
        for z in range(images.sizes['z']):
            image = images.get_frame_2D(c=c, t=0, z=z)
            upper_bound = np.max(image)
            image = map_uint16_to_uint8(image, lower_bound=0, upper_bound=upper_bound)
            imgs[c].append(image)

imgs = np.asarray(imgs)
shownp(imgs[4,0,:,:])

print(np.shape(imgs))

f = '/mnt/Volume D/AllDatasets/houston/nd2Houston'
dsets = ImageNikonND2(f)
f = dsets.getitem_with_ub(0)
shownp(f[0,0,:,:])
f = dsets.getitem_no_ub(0)
shownp(f[0,0,:,:])

f = "/mnt/Volume D/exctracted_images/Bryan Tsao/210804 Kappa CAR NK92MI lipid bilayer/8bit/2021_08_09_2G_6ug-1.tif.frames"
f = "D:\\exctracted_images\\Bryan Tsao\\210804 Kappa CAR NK92MI lipid bilayer\\"
ibt = ImageBT(f)
ibt.change_dir(16)
print(len(ibt[1]),len(ibt[1][0]))
"""
