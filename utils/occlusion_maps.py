"""
Reference:
    https://www.kaggle.com/code/blargl/simple-occlusion-and-saliency-maps/notebook#Occlusion-Map
"""

import numpy as np
from collections import defaultdict


class OcclusionMaps:
    def __init__(self, aImg):
        self.img = aImg
    
    def iter_occlusion(self, aSize=8):
        occlusion = np.full((aSize * 5, aSize * 5, 1), [0.5], np.float32)
        occlusion_center = np.full((aSize, aSize, 1), [0.5], np.float32)
        occlusion_padding = aSize * 2

        image_padded = np.pad(self.img, ((occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0) \
                                           ), 'constant', constant_values = 0.0)

        for y in range(occlusion_padding, self.img.shape[0] + occlusion_padding, aSize):

            for x in range(occlusion_padding, self.img.shape[1] + occlusion_padding, aSize):
                tmp = image_padded.copy()

                tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
                    x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
                    = occlusion

                tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center

                yield x - occlusion_padding, y - occlusion_padding, \
                    tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]

    def get_heatmap(self, aModel, aOcclusionSize, aCorrectClass, aImgSize):

        heatmap = np.zeros((aImgSize, aImgSize), np.float32)
        class_pixels = np.zeros((aImgSize, aImgSize), np.int16)

        counters = defaultdict(int)

        for n, (x, y, img_float) in enumerate(self.iter_occlusion(aSize=aOcclusionSize)):   

            X = img_float.reshape(1, aImgSize, aImgSize, 3)
            out = aModel.predict(X)

            heatmap[y:y + aOcclusionSize, x:x + aOcclusionSize] = out[0][aCorrectClass]
            class_pixels[y:y + aOcclusionSize, x:x + aOcclusionSize] = np.argmax(out)
            counters[np.argmax(out)] += 1

        return heatmap


