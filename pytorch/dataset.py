import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        for label, cls in enumerate(["real", "fake"]):
            class_path = os.path.join(root_dir, cls)
            for img in os.listdir(class_path):
                self.samples.append(
                    (os.path.join(class_path, img), label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
