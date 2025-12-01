"""
Image augmentation transforms for autonomous driving.
Based on SimLingo's augmentation strategy.
"""

import numpy as np
from imgaug import augmenters as ia


def get_image_augmenter(prob: float = 0.5, cutout: bool = False):
    """
    Create image augmentation pipeline matching SimLingo's strategy.
    
    Args:
        prob: Probability of applying each augmentation
        cutout: Whether to include cutout augmentation
        
    Returns:
        imgaug Sequential augmenter
    """
    augmentations = [
        ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
        ia.Sometimes(prob, ia.AdditiveGaussianNoise(loc=0, scale=(0., 0.05 * 255), per_channel=0.5)),
        ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),
        ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),  # Brightness
        ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),
        ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
    ]
    
    if cutout:
        augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False)))
    
    augmenter = ia.Sequential(augmentations, random_order=True)
    return augmenter


class ImagePreprocessor:
    """
    Image preprocessing for autonomous driving models.
    Matches SimLingo's preprocessing pipeline.
    """
    
    def __init__(
        self,
        target_size: tuple = (448, 448),
        cut_bottom_quarter: bool = True,
        augment: bool = True,
        augment_prob: float = 0.5,
    ):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            cut_bottom_quarter: Whether to remove bottom of image (car hood)
            augment: Whether to apply augmentation
            augment_prob: Probability of applying augmentations
        """
        self.target_size = target_size
        self.cut_bottom_quarter = cut_bottom_quarter
        self.augment = augment
        
        if augment:
            self.augmenter = get_image_augmenter(prob=augment_prob)
        else:
            self.augmenter = None
    
    def __call__(self, image: np.ndarray, apply_augment: bool = True) -> np.ndarray:
        """
        Preprocess an image.
        
        Args:
            image: Input image (H, W, C) in RGB format, uint8 [0, 255]
            apply_augment: Whether to apply augmentation (can override self.augment)
            
        Returns:
            Preprocessed image (H, W, C) in RGB format, uint8 [0, 255]
        """
        # Apply augmentation if enabled
        if self.augment and apply_augment and self.augmenter is not None:
            image = self.augmenter(image=image)
        
        # Cut bottom quarter to remove car hood
        # SimLingo removes 4.8/16 of the bottom
        if self.cut_bottom_quarter:
            cut_height = int(image.shape[0] * (4.8 / 16))
            image = image[:-cut_height, :, :]
        
        # Note: Resizing to target_size is typically done by the vision encoder
        # (e.g., InternVL2 handles this internally)
        # We keep the image at its current size here
        
        return image
    
    def preprocess_batch(self, images: list, apply_augment: bool = True) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images
            apply_augment: Whether to apply augmentation
            
        Returns:
            Stacked array of preprocessed images
        """
        processed = [self(img, apply_augment=apply_augment) for img in images]
        return np.stack(processed, axis=0)

