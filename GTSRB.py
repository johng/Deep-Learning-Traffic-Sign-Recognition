import numpy as np


class gtsrb:
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3
    OUTPUT = 43
    nTestSamples = 200

    def __init__(self, batchsize=128):

        dataset = np.load('gtsrb_dataset.npz')

        self.trainData = dataset['X_{0:s}'.format('train')]
        self.trainLabels = dataset['y_{0:s}'.format('train')]
        self.testData = dataset['X_{0:s}'.format('test')]
        self.testLabels = dataset['y_{0:s}'.format('test')]

        self.nTrainSamples = len(self.trainLabels)
        self.nTestSamples = len(self.testLabels)

        self.batchSize = batchsize

        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)



        self.currentIndexTest = 0
        self.currentIndexTrain = 0

    def getTrainBatch(self, allowSmallerBatches=False):
        return self._getBatch('train', allowSmallerBatches)

    def getTestBatch(self, allowSmallerBatches=False):
        return self._getBatch('test', allowSmallerBatches)

    def reset(self):

        self.currentIndexTrain = 0
        self.currentIndexTest = 0
        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)

    def _getBatch(self, dataSet, allowSmallerBatches=False):

        D = np.array([])
        L = np.array([])

        if dataSet == 'train':
            train = True
            test = False
        elif dataSet == 'test':
            train = False
            test = True
        else:
            raise ValueError('_getBatch: Unrecognised set: ' + dataSet)

        while True:
            if train:
                r = range(self.currentIndexTrain,
                          min(self.currentIndexTrain + self.batchSize - L.shape[0], self.nTrainSamples))
                self.currentIndexTrain = r[-1] + 1 if r[-1] < self.nTrainSamples - 1 else 0
                (d, l) = (self.trainData[self.pTrain[r]][:], self.trainLabels[self.pTrain[r]][:])
            elif test:
                r = range(self.currentIndexTest,
                          min(self.currentIndexTest + self.batchSize - L.shape[0], self.nTestSamples))
                self.currentIndexTest = r[-1] + 1 if r[-1] < self.nTestSamples - 1 else 0
                (d, l) = (self.testData[self.pTest[r]][:], self.testLabels[self.pTest[r]][:])

            if D.size == 0:
                D = d
                L = l
            else:
                D = np.concatenate((D, d))
                L = np.concatenate((L, l))

            if D.shape[0] == self.batchSize or allowSmallerBatches:
                break

        return D, L
