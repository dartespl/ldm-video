import torch as th
import numpy as np
import torchvision
import cv2
from PIL import Image

def q_sample_frames(paths, t):
    transformed = []
    for i in range(t.shape[0]):
        frame = extractFrame(paths[i], t[i].item(), 128)
        img = transformImage(frame)
        transformed.append(img)

    return th.tensor(transformed)#.to(device=dist_util.dev())

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def transformImage(img):
    img = th.tensor(img.copy())
    img = np.transpose(img, [2, 0, 1])
    img = torchvision.transforms.functional.to_pil_image(img)

    arr = center_crop_arr(img, 128)

    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])


def extractFrame(pathIn, frame_num, num_frames):
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    seconds = int(frames / fps)

    vidcap.set(cv2.CAP_PROP_POS_MSEC, (frame_num * (seconds * (1000 / num_frames))))  # added this line
    success, image = vidcap.read()
    if image is None:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, 0)  # added this line
        success, image = vidcap.read()
    return image[..., ::-1]