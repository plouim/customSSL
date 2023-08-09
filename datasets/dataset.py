from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from timm.data.readers.class_map import load_class_map
from pathlib import Path

# class ImageDataset(ImageFolder):
#     def __init__(self, root:str, class2idx:dict, transform=None, target_transform=None) -> None:
#         super().__init__(root=root, transform=transform, target_transform=target_transform)
#         samples = self.make_dataset(root, class2idx, self.extensions)

#         self.imgs=[s[0] for s in samples]
#         self.targets=[s[1] for s in samples]

#     def __getitem__(self, index):
#         img = Image.open(self.imgs[index])
#         target = self.targets[index]
#         if self.transforms:
#             img = self.transforms(img)
#         return img, target
    

class SSLDataset(Dataset):
    def __init__(self, root:str, class_map:str, transform=None, target_transform=None, has_label=True) -> None:
        IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp",
                          ".JPG", ".JPEG", ".PNG", ".PPM", ".BMP", ".PGM", ".TIF", ".TIFF", ".WEBP",
            )
        super().__init__()
        self.root = root
        self.class_map = class_map
        self.tranform=transform
        self.target_transforms=target_transform
        
        img_files = []
        for extension in IMG_EXTENSIONS:        
            img_files.extend(list(Path(self.root).rglob('*' + extension)))
        self.img_files = img_files
        
        if self.class_map:
            self.class2idx = load_class_map(self.class_map)
        else:
            self.class2idx = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        if self.class2idx:
            label = self.class2idx[self.img_files[index].parent.name]
        else:
            label=0

        if self.tranform:
            img = self.tranform(img)
        if self.target_transforms:
            label = self.target_transforms(label)

        return img, label