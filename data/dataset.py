import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


# code modified from https://github.com/atuannguyen/DIRT/blob/main/domain_gen_rotatedmnist/mnist_loader.py
class RotationDataset(data_utils.Dataset):
    def __init__(self,
                 list_train_domains,
                 root,
                 dataset,
                 train=True,
                 mnist_subset='med',
                 transform=None,
                 download=True,
                 list_test_domains=None):

        """
        :param list_train_domains: all domains we observe in the training
        :param root: data directory
        :param train: whether to load MNIST training data
        :param mnist_subset: 'max' - for each domain, use 60000 MNIST samples, 'med' - use 10000 MNIST samples, index from 0-9 - use 1000 MNIST samples
        :param transform: ...
        :param download: ...
        :param list_test_domains: whether to load unseen domains
        """

        self.list_train_domains = list_train_domains
        self.dataset = dataset
        self.mnist_subset = mnist_subset
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.download = download
        self.list_test_domains = list_test_domains

        # self.not_eval = not_eval  # load test MNIST dataset

        if not self.list_test_domains:
            print('Running domain translation task')
            self.data, self.labels, self.domain, self.angles = self._get_data()
        else:
            print('Running domain generalization task')
            assert self.train is True, 'Use train set for new domains' # if want to use test set, remove this line
            self.data, self.labels, self.domain = self._get_data()

    def load_inds(self):
        '''
        If specifyign a subset, load 1000 mnist samples with balanced class (100 samples
        for each class). If not, load 10000 mnist samples.
        :return: indices of mnist samples to be loaded
        '''
        if self.mnist_subset == 'med':
            fullidx = np.array([])
            for i in range(10):
                fullidx = np.concatenate(
                    (fullidx, np.load(os.path.join(self.root, f'{self.dataset}/supervised_inds_{i}.npy'))))
            return fullidx
        else:
            return np.load(os.path.join(self.root, f'{self.dataset}/supervised_inds_{self.mnist_subset}.npy'))

    def _get_data(self):
        if not self.list_test_domains:
            if self.train:
                bs = 60000
            else:
                bs = 10000
                self.mnist_subset = 'max'  # always use full set for test data as we don't have saved indices for MNIST test set

            train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                      train=self.train,
                                                                      download=self.download,
                                                                      transform=transforms.ToTensor()),
                                                       batch_size=bs,
                                                       shuffle=False)

            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y

            if self.mnist_subset != 'max':
                # Get labeled examples
                print(f'use MNIST subset {self.mnist_subset}!')
                sup_inds = self.load_inds()
                mnist_labels = mnist_labels[sup_inds]
                mnist_imgs = mnist_imgs[sup_inds]
            else:
                print('use all MNIST data!')

            num_samples = mnist_imgs.shape[0]

            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            # Run transforms
            mnist_0_img = torch.zeros((num_samples, 28, 28))
            mnist_15_img = torch.zeros((num_samples, 28, 28))
            mnist_30_img = torch.zeros((num_samples, 28, 28))
            mnist_45_img = torch.zeros((num_samples, 28, 28))
            mnist_60_img = torch.zeros((num_samples, 28, 28))
            mnist_75_img = torch.zeros((num_samples, 28, 28))

            for i in range(len(mnist_imgs)):
                mnist_0_img[i] = to_tensor(to_pil(mnist_imgs[i]))

            for i in range(len(mnist_imgs)):
                mnist_15_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 15))

            for i in range(len(mnist_imgs)):
                mnist_30_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 30))

            for i in range(len(mnist_imgs)):
                mnist_45_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 45))

            for i in range(len(mnist_imgs)):
                mnist_60_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 60))

            for i in range(len(mnist_imgs)):
                mnist_75_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 75))

            # Choose subsets that should be included into the training
            training_list_img = []
            training_list_labels = []
            train_angles = []
            for domain in self.list_train_domains:
                if domain == '0':
                    training_list_img.append(mnist_0_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(0)
                if domain == '15':
                    training_list_img.append(mnist_15_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(15)
                if domain == '30':
                    training_list_img.append(mnist_30_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(30)
                if domain == '45':
                    training_list_img.append(mnist_45_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(45)
                if domain == '60':
                    training_list_img.append(mnist_60_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(60)
                if domain == '75':
                    training_list_img.append(mnist_75_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(75)

                    # Stack
            train_imgs = torch.cat(training_list_img)
            train_labels = torch.cat(training_list_labels)

            # Create domain labels
            train_domains = torch.zeros(train_labels.size())
            for i in range(len(self.list_train_domains)):
                train_domains[i * num_samples:(i + 1) * num_samples] += i

            # Shuffle everything one more time
            inds = np.arange(train_labels.size()[0])
            np.random.shuffle(inds)
            train_imgs = train_imgs[inds]
            train_labels = train_labels[inds]
            train_domains = train_domains[inds].long()

            return train_imgs.unsqueeze(1), train_labels, train_domains, train_angles

        else:
            if not self.list_test_domains:
                if self.train:
                    bs = 60000
                else:
                    bs = 10000
                    self.mnist_subset = 'max'  # always use full set for test data as we don't have saved indices for MNIST test set
            train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                      train=self.train,
                                                                      download=self.download,
                                                                      transform=transforms.ToTensor()),
                                                       batch_size=bs,
                                                       shuffle=False)

            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y

            if self.mnist_subset != 'max':
                # Get labeled examples
                print(f'use MNIST subset {self.mnist_subset}!')
                sup_inds = self.load_inds()
                mnist_labels = mnist_labels[sup_inds]
                mnist_imgs = mnist_imgs[sup_inds]
            else:
                print('use all MNIST data!')

            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            # Get angle
            rot_angle = int(self.list_test_domain[0])

            # Resize
            num_samples = int(mnist_imgs.shape[0])
            mnist_imgs_rot = torch.zeros((num_samples, 28, 28))

            for i in range(len(mnist_imgs)):
                mnist_imgs_rot[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), rot_angle))

            # Create domain labels
            test_domain = torch.zeros(mnist_labels.size()).long()

            return mnist_imgs_rot.unsqueeze(1), mnist_labels, test_domain

    def __len__(self):
        if not self.list_test_domains:
            return len(self.labels)
        else:
            return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        d = self.domain[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y, d



