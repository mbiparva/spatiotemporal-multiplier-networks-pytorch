import collections
import numbers
import random

import torch
from torchvision.transforms import functional
import numpy as np
from PIL import Image, ImageChops
try:
    import accimage
except ImportError:
    accimage = None

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mode):
        mode, _, axis = mode.partition('_')
        for t in self.transforms:
            if isinstance(t, Normalize) and mode == 'flow':
                img = t(img, mean=[127], std=[1])
            elif isinstance(t, RandomHorizontalFlip) and mode == 'flow' and axis == 'u' and t.random_value < t.p:
                img = ImageChops.invert(t(img))
            else:
                img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


# -------------------------------------------------
# functional re-implementation
# -------------------------------------------------
def to_tensor(pic, norm_value):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        norm_value (Integer): The value by which the normalization is done.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(norm_value)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(norm_value)
    else:
        return img


def random_corner_crop(img, crop_position, size, b_w=0, b_h=0):
    w, h = img.size
    c_h, c_w = size

    if b_w and b_h:
        b_w, b_h = int((w - c_w) * b_w), int((h - c_h) * b_h)

    if c_h > w or c_w > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size, (h, w)))

    if crop_position == 'center':
        return functional.center_crop(img, (c_h, c_w))
    elif crop_position == 'tl':
        return img.crop((b_w, b_h, b_w + c_w, b_h + c_h))
    elif crop_position == 'tr':
        return img.crop((w - c_w - b_w, b_h, w - b_w, b_h + c_h))
    elif crop_position == 'bl':
        return img.crop((b_w, h - c_h - b_h, b_w + c_w, h - b_h))
    elif crop_position == 'br':
        return img.crop((w - c_w - b_w, h - c_h - b_h, w - b_w, h - b_h))
    else:
        raise NotImplementedError


# -------------------------------------------------
# transform re-implementation
# -------------------------------------------------
class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic, norm_value=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic, norm_value if norm_value is not None else self.norm_value)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, mean=None, std=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return functional.normalize(tensor, self.mean if mean is None else mean, self.std if std is None else std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def randomize_parameters(self):
        pass


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        # noinspection PyTypeChecker
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size
        size_set = False
        if isinstance(self.size, int):
            size = self.size
            if w == size and h == size:
                size_set = True
        elif isinstance(self.size, collections.Iterable) and len(self.size) == 2:
            size = [i for i in self.size]
            if w == size[1] and h == size[0]:
                size_set = True
        else:
            raise NotImplementedError
        if size_set:
            return img
        else:
            return functional.resize(img, size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            # noinspection PyTypeChecker
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        assert min(img.size) == 256
        return functional.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def randomize_parameters(self):
        pass


class RandomCornerCrop(object):
    """Randomly crops the given PIL Image at the four corners or center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_position (sequence): Desired crop position of the crop. If it is None,
            the position is randomly selected. Default choices are
            ('center', 'tl', 'tr', 'bl', 'br').
        crop_scale (sequence): Desired list of scales in the range (0, image_size)
            to randomly crop from corners.
    """
    def __init__(self, size, crop_position=None, crop_scale=1.0, border=0):
        self.size, self.crop_size, self.border = size, size, border
        self.border_w, self.border_h = 0, 0
        self.randomize_corner, self.randomize_scale = True, True
        self.default_positions = ('center', 'tl', 'tr', 'bl', 'br')
        if crop_position is not None:
            self.randomize_corner, self.crop_position = False, crop_position
        if isinstance(crop_scale, tuple):
            self.crop_scale = crop_scale
        elif isinstance(crop_scale, float):
            self.randomize_scale = False
            self.crop_size = self.size * crop_scale
            self.crop_size = (int(self.crop_size), int(self.crop_size))
        else:
            raise Exception('NotDefined')

    def __call__(self, img):
        return random_corner_crop(img, self.crop_position, self.crop_size, self.border_w, self.border_h)

    def randomize_parameters(self):
        if self.randomize_corner:
            self.crop_position = random.choice(self.default_positions)
        if self.randomize_scale:
            if len(self.crop_scale) == 2 and self.crop_scale[0] < 2:
                self.crop_size = self.size * np.random.uniform(low=self.crop_scale[0], high=self.crop_scale[1])
            else:
                self.crop_size = np.random.choice(self.crop_scale)
            self.crop_size = (int(self.crop_size), int(self.crop_size))
        if self.border > 0:
            self.border_w, self.border_h = self.border*np.random.rand(), self.border*np.random.rand()


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p, self.random_value = p, None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.random_value < self.p:
            return functional.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    def randomize_parameters(self):
        self.random_value = random.random()


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.75, 1.0), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.r_scale, self.r_tl = None, None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        w, h = img.size
        r_tl_w, r_tl_h = self.r_tl

        min_length = min(w, h)
        crop_size = int(min_length * self.r_scale)

        i = r_tl_h * (h - crop_size)
        j = r_tl_w * (w - crop_size)

        return functional.resized_crop(img, i, j, crop_size, crop_size, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

    def randomize_parameters(self):
        self.r_scale = random.uniform(*self.scale)
        self.r_tl = [random.random(), random.random()]
