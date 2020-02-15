# coding=utf-8
import os
import pandas as pd
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() # interactive model 

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
# iloc : Purely integer-location based, access a group of rows and columns
img_name = landmarks_frame.iloc[n, 0] 
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 5 landmarks: {}'.format(landmarks[:5]))

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001) # pause a bit so that plots are updated

#plt.figure()
#show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
#               landmarks)
#plt.show()

#input()

'''
    torch.utils.data.Dataset is an abstract class representing a dataset. Your
    custom dataset should inherit `Dataset` and override the following methods:
    -- __len__
    -- __getitem__
'''

class FaceLandmarksDataset(Dataset):
    '''Face Landmarks Dataset'''

    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args:
            csv_file(string): Path to the csv file with annotations.
            root_dir(string): Directory with all images
            transform(callable, optional): Optional transform to be 
                applied on a sample
        '''
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, 
                                self.landmarks_frame.iloc[index, 0]) 
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[index, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

'''
    Let's instantiate this iterate through the data samples.
'''
face_dataset = FaceLandmarksDataset(
        csv_file='data/faces/face_landmarks.csv',
        root_dir='data/faces/')

plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample) # why use two asterisks ?

    if i == 3:
        plt.show()
        break
#input()

'''
    One issue we can see from the above is that the samples are not of the same
    size. Most neural networks expect the images of a fixed size.
    -- Rescale: to scale the image
    -- RandomCrop: to crop from image randomly. This is data augmentation.
    -- ToTensor: to convert the numpy images to torch images
    
    We will write them as callable classes instead of simple functions so that
    parameters of the transform need not be passed everytime it's called.
'''

class Rescale(object):
    '''Rescale the image in a sample to a given size
       
    Args:
        output_size(tuple or int): Desired output size. If tuple, output is 
        matched to output_size. If int, samller of image edges is matched to
        output_size keeping aspect ratio the same.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # X and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    '''Crop randomly the image in a sample

    Args:
        output_size(tuple or int): Desired output size. If int, square crop 
        is made
    '''
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        landmarks = torch.from_numpy(landmarks)
        return {'image': image, 'landmarks': landmarks}
       

# Apply each of the above transforms on sample.
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), 
                               RandomCrop(224)])

fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
#input()

'''
    Iterating through the dataset

    Let's put this all together to create a dataset with composed transform.
    To summarize, every time this dataset is sampled:
    - An image is reaad from the file on the fly
    - Transforms are applied on the read image
    - Since one of the transforms is random, data is augmented on sampling

    We can iterate over the created dataset with a ``for i in range`` 
    loop as before
'''

transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', 
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()])
                                           )

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

'''
    However, we are losing a lot of features by using a simple ``for`` loop to
    iterate over the data. In particular, we are missing out on:

    - Batching the data
    - Shuffling the data
    - Load the data in parallel using ``multiprocessing`` workers.

    ``torch.utils.data.DataLoader`` is an iterator which provides all these
    features. Parameters used below should be clear. One parameter of interest is 
    ``collate_fn``. You can specify how exactly the samples need to be batched 
    using ``collate_fn``. However, default collate should work fine for most
    use cases.
'''

dataloader = DataLoader(transformed_dataset, batch_size=4, 
                        shuffle=True, num_workers=4)

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    '''Show image with landmarks for a batch of samples.'''
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i+1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')
        plt.title('Batch from dataloader')


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    #observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
input()


'''
    torchvision package provides some common datasets and transforms.
'''
#data_transform = transforms.Compose([
#        transforms.RandomSizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        ])
#
#hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#        transform=data_transform)
#
#datasets_loader = DataLoader(hymenoptera_dataset, batch_size=4, shuffle=True,
#        num_workers=4)
#print(type(datasets_loader))





