"""
Reference:
    https://github.com/eclique/RISE/blob/master/Easy_start.ipynb
"""

import numpy as np
from skimage.transform import resize
from tqdm import tqdm


class RISE:
    def __init__(self, aN, aS, aP1, aModel, aBatchSize, aInputSize):
        self.n = aN
        self.s = aS
        self.p1 = aP1
        self.model = aModel
        self.batch_size = aBatchSize
        self.input_size = aInputSize

    def generate_masks(self):
        cell_size = np.ceil(np.array(self.input_size) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(self.n, self.s, self.s) < self.p1
        grid = grid.astype('float32')

        masks = np.empty((self.n, *self.input_size))

        for i in tqdm(range(self.n), desc='Generating masks'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        masks = masks.reshape(-1, *self.input_size, 1)
        return masks
    
    def explain(self, aInp, aMasks):
        preds = []
        # Make sure multiplication is being done for correct axes
        masked = aInp * aMasks
        for i in tqdm(range(0, self.n, self.batch_size), desc='Explaining'):
            preds.append(self.model.predict(masked[i:min(i+self.batch_size, self.n)]))
        preds = np.concatenate(preds)
        sal = preds.T.dot(aMasks.reshape(self.n, -1)).reshape(-1, *self.input_size)
        sal = sal / self.n / self.p1
        return sal