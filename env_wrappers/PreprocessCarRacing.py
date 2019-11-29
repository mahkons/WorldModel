import cv2
import gym
import numpy as np


class PreprocessCarRacing(gym.ObservationWrapper):
    @staticmethod
    def _resize(img):
        """ Simple downsampling to (64, 56)"""
        return cv2.resize(img, (64, 56))

    @staticmethod
    def _crop(img):
        """Remove unnecessary parts of image"""
        return img[:-12, :, :]

    @staticmethod
    def _to_float(img):
        """More memory to the god of the memory"""
        return np.asarray(img, dtype=np.float64) / 255.0

    def observation(self, img):
        img = self._crop(img)
        img = self._resize(img)
        img = self._to_float(img)
        return img

    def __init__(self, env):
        super().__init__(env)

        self.img_size = (56, 64)  # Eee, magic constants (see crop + downsampling)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(*self.img_size, 3), dtype=np.float64)
