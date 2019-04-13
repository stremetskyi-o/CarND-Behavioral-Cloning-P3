from dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset('data')
    dataset.load()
    print('Dataset size =', len(dataset.imgs))
    dataset.augment()
    print('Dataset size after augmentation =', len(dataset.imgs))
    dataset.validation_split(0.2)
    print('Train size=', len(dataset.x_train))
    print('Validation size=', len(dataset.x_valid))
