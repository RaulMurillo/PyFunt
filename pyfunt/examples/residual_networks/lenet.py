from pyfunt import (SpatialConvolution, 
                    Sequential,
                    ReLU,
                    Linear,
                    Reshape,
                    SpatialMaxPooling,
                    LogSoftMax)


def lenet():
    model = Sequential()
    add = model.add
    add(SpatialConvolution(1, 6, 5, 5, 1, 1, 0, 0))
    add(ReLU())
    add(SpatialMaxPooling(2, 2, 2, 2))
    add(SpatialConvolution(6, 16, 5, 5, 1, 1, 0, 0))
    add(ReLU())
    add(SpatialMaxPooling(2, 2, 2, 2))
    add(Reshape(400))
    add(Linear(400, 120))
    add(ReLU())
    add(Linear(120, 84))
    add(ReLU())
    add(Linear(84, 10))
    add(LogSoftMax())
    return model