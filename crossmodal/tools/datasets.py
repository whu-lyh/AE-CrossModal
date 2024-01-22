'''
Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

We thank the Nanne repo https://github.com/Nanne/pytorch-NetVlad for inspiration
into the design of the dataloader
'''


import torchvision.transforms as transforms


def input_transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.3),
                transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1))], p=0.5),
                # transforms.RandomHorizontalFlip(p=0.5),
                #transforms.Resize(resize), # TODO extra performance to be validated
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.396197, 0.452953, 0.490031],
                                 std=[0.315778, 0.343629, 0.369563]),
            # pytorch statistic information
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                       std=[0.229, 0.224, 0.225]),
            # kitti360 dataset
            # transforms.Normalize(mean=[0.056020, 0.064157, 0.067195],
            # std=[0.172305, 0.192164, 0.204675]),
        ])
    else:
        return transforms.Compose([
            #transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.396197, 0.452953, 0.490031],
                                 std=[0.315778, 0.343629, 0.369563]),
            # pytorch statistic information
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                       std=[0.229, 0.224, 0.225]),
            # kitti360 dataset
            # transforms.Normalize(mean=[0.056020, 0.064157, 0.067195],
            # std=[0.172305, 0.192164, 0.204675]),
    ])


def configure_transform(image_dim, meta):
	normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
	transform = transforms.Compose([
		transforms.Resize(image_dim),
		transforms.ToTensor(),
		normalize,
	])
	return transform