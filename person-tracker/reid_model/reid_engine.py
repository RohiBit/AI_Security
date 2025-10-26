import torchreid
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class ReIDEngine:
    def __init__(self, model_name='osnet_x1_0'):
        # ✅ Automatically downloads pretrained weights
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            loss='softmax',
            pretrained=True
        )
        self.model.eval()

        # Optional: move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def extract_features(self, img):
        if isinstance(img, np.ndarray):
            img = self.transform(img).unsqueeze(0)
        else:
            raise ValueError("Input should be a NumPy array (OpenCV image)")

        img = img.to(self.device)
        with torch.no_grad():
            features = self.model(img)
        return features.squeeze(0).cpu().numpy()
