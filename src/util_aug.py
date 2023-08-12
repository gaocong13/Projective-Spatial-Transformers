import imgaug.augmenters as iaa
import random
import torch

def invert_fn(im):
    im_max = im.max()
    im = im_max - im

    return im

def noise_fn(im, device):
    im_min = im.min()
    im_max = im.max()

    im = (im - im_min) / (im_max - im_min)

    cur_noise_sigma = random.uniform(0.005, 0.01)
    im += torch.randn(im.shape, device=device) * cur_noise_sigma

    im = (im * (im_max - im_min)) + im_min

    return im

def gamma_fn(im):
    im_min = im.min()
    im_max = im.max()

    im = (im - im_min) / (im_max - im_min)

    gamma = random.uniform(0.7,1.3)
    im.pow_(gamma)

    im = (im * (im_max - im_min)) + im_min

    return im

def box_corrupt_fn(im, device):
    im_2d_shape = [im.shape[-2], im.shape[-1]]
    box_mean_dim = torch.Tensor([im_2d_shape[0] * 0.15, im_2d_shape[1] * 0.15])

    # Number of random corrupted boxes
    num_boxes = random.randint(1,5)

    for box_idx in range(num_boxes):
        box_valid = False

        while not box_valid:
            # First sample box dims
            box_dims = torch.round((torch.randn(2) * (box_mean_dim)) + box_mean_dim).long()

            if (box_dims[0] > 0) and (box_dims[1] > 0) and \
                    (box_dims[0] <= im_2d_shape[0]) and (box_dims[1] <= im_2d_shape[1]):
                # Next sample box location
                start_row = random.randint(0, im_2d_shape[0] - box_dims[0])
                start_col = random.randint(0, im_2d_shape[1] - box_dims[1])

                box_valid = True

        im_roi = im[0,start_row:(start_row+box_dims[0]),start_col:(start_col+box_dims[1])]

        sigma_noise = (im_roi.max() - im_roi.min()) * 0.2

        im_roi += torch.randn(im_roi.shape).to(device) * sigma_noise

    return im

def aug_torch_target(im, device):
    BATCH_SIZE = im.size()[0]

    for b in range(BATCH_SIZE):
        if random.random() < 0.5:
            im[b, :, :] = invert_fn(im[b, :, :])

        if random.random() < 0.5:
            im[b, :, :] = noise_fn(im[b, :, :], device)

        if random.random() < 0.5:
            im[b, :, :] = gamma_fn(im[b, :, :])

        if random.random() < 0.5:
            im[b, :, :] = box_corrupt_fn(im[b, :, :], device)

    return im

def aug_numpy_target_imgaug(target_arr, heavy_aug=False):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    contrast_seq = iaa.Sequential([
                    iaa.OneOf([
                        # Improve or worsen the contrast of images.
                        iaa.LinearContrast((0.5, 2.0)),
                        iaa.GammaContrast((0.5, 2.0)),
                        iaa.LogContrast(gain=(0.6, 1.4)),
                        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                        ])
            ])

    saltpepper_seq = iaa.Sequential([
            iaa.OneOf([
                iaa.ImpulseNoise(0.1),
                iaa.SaltAndPepper(0.1),
                iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)),
                ])
            ])

    heavy_seq = iaa.Sequential([
                #
                # Execute 0 to 2 of the following (less important) augmenters per
                # image. Don't execute all of them, as that would often be way too
                # strong.
                #

                iaa.SomeOf((1, 2),
                [
                    # Blur
                    sometimes(
                    iaa.OneOf([
                        # Blur each image with varying strength using
                        # gaussian blur (sigma between 0 and 3.0),
                        # average/uniform blur (kernel size between 2x2 and 7x7)
                        # median blur (kernel size between 3x3 and 11x11).
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(4, 6)),
                    ])
                    ),

                    # Additive Noise Injection
                    # sometimes(
                    # iaa.OneOf([
                    #     iaa.AdditiveLaplaceNoise(scale=(0, scale_val)),
                    #     iaa.AdditivePoissonNoise(scale_val),
                    # ])
                    # ),

                    # Dropout
                    sometimes(
                    iaa.OneOf([
                        # Either drop randomly 1 to 10% of all pixels (i.e. set
                        # them to black) or drop them on an image with 2-5% percent
                        # of the original size, leading to large dropped
                        # rectangles.
                        iaa.Dropout((0.1, 0.15)),
                        iaa.CoarseDropout(
                            (0.1, 0.15), size_percent=(0.1, 0.15)
                        ),
                    ])
                    ),

                    # Convolutional
                    sometimes(
                    iaa.OneOf([
                        # Sharpen each image, overlay the result with the original
                        # image using an alpha between 0 (no sharpening) and 1
                        # (full sharpening effect).
                        iaa.Sharpen(alpha=(0.5, 1.0), lightness=(1.0, 1.5)),

                        # Same as sharpen, but for an embossing effect.
                        iaa.Emboss(alpha=(0.5, 1.0), strength=(1.0, 2.0))
                    ])
                    ),

                    # Pooling
                    sometimes(
                    iaa.OneOf([
                        iaa.AveragePooling([2, 4]),
                        iaa.MaxPooling([2, 4]),
                        iaa.MinPooling([2, 4]),
                        iaa.MedianPooling([2, 4]),
                        ])
                    ),

                    # Multiply
                    sometimes(
                        iaa.OneOf([
                            # Change brightness of images (50-150% of original value).
                            iaa.Multiply((0.5, 1.5)),
                            iaa.MultiplyElementwise((0.5, 1.5))
                        ])
                    ),

                    # Replace 10% of all pixels with either the value 0 or max_val
                    # sometimes(
                    #     iaa.ReplaceElementwise(0.1, [0, scale_val])
                    # ),

                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    )
                ],
                # do all of the above augmentations in random order
                random_order=True
                )
            ],
            # do all of the above augmentations in random order
            random_order=True)

    img = contrast_seq(images=target_arr)
    img = saltpepper_seq(images=img)

    if heavy_aug:
        img = heavy_seq(images=img)

    return img
