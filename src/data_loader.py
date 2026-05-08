import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BirdSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_classes=None):
        self.root_dir = root_dir
        self.transform = transform

        all_classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
        ])

        if max_classes is not None:
            self.classes = all_classes[:max_classes]
        else:
            self.classes = all_classes

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.png'):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(root_dir, batch_size=32, is_train=True, max_classes=None):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = BirdSpectrogramDataset(root_dir=root_dir, transform=transform, max_classes=max_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return loader, len(dataset.classes)
