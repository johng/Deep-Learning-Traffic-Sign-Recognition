# Deep Learning Traffic Sign Recogniser

[Zhang et al. A Shallow Network with Combined Pooling for Fast Traffic Sign Recognition](http://www.mdpi.com/2078-2489/8/2/45/htm)

## Requirements

* Tensorflow 1.2
* Numpy
* GTSRB Dataset 'gtsrb_dataset.npz'

## Replication

By default all our improvements are disabled, so the replication can be run with

`python main.py`

## Improvement

To run our improvemed version the augmented dataset must first be generated

### Generating the Augmentated DataSet

To generate the augmented data we use OpenCV to manipulate the training data

####  Configuring OpenCV

```
curl https://raw.githubusercontent.com/mitsuhiko/pipsi/master/get-pipsi.py | python
pipsi install backports.shutil_get_terminal_size
pipsi install pew
pip install pipenv --user
pipenv --site-packages  # initialises virtualenv with access to system installation of Tensorflow
pipenv install  # installs packages listed in Pipfile.lock 

```

#### Executing the data augmentation

* First enter the pipenv environment: `pipenv shell`
* Generate augmented data: `python augment_data.py generate`

##### Optional
Viewing augmented data `python augment_data.py show 1000`
Where 1000 is the index of the image you want to view variants of. Environment variable `SCIPY_PIL_IMAGE_VIEWER` must be set to an image viewer.


### Running the improved version

`python main.py --multi-scale --crelu --use-augmented-data --dropout-keep-rate=0.7 --max-pools`


## Architecture

### Layers

**Layer**|**Name**|**Kernel**|**Stride**|**Padding**|**Output**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
1|Input|-|-|-|(32,32,3)
2|Convolution/Relu|5x5|1|2|(32,32,32)
3|Average Pool|3x3 |2|[0 1 0 1]|(32,16,16)
4|Convolution/Relu|5x5|1|2|(32,16,16)
5|Average Pool|3x3 |2|[0 1 0 1]|(32,8,8)
6|Convolution/Relu|5x5|1|2|(64,8,8)
7|Max Pool|3x3 |2|[0 1 0 1]|(64,4,4)
8|Convolution/Relu|4x4|1|2|(64,1,1)
9|Fully Connected|1x1|1|0|(64)
10|Softmax Loss| | | |(43)
