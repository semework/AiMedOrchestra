"""
aimedorchestra/agents/imaging/agent.py
------------------------------------
Fully-functional demo ImagingAgent with graceful OpenCV fallback.

Install for full features:
    pip install torch torchvision pydicom opencv-python-headless pillow numpy
"""

from pathlib import Path
import numpy as np

# ─── Try OpenCV; fall back to Pillow if missing ─────────────────────────
try:
    import cv2
    _CV_OK = True
except ModuleNotFoundError:
    _CV_OK = False
    from PIL import Image
    print("⚠️  OpenCV not found – ImagingAgent will load images via Pillow "
          "and return a dummy heat-map.  "
          "Install opencv-python-headless for full functionality.")

import pydicom
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models


class ImagingAgent:
    """
    ImagingAgent
    ------------
    • Backbones: ResNet-50 + EfficientNet-B3 (ImageNet-pre-trained)
    • Explainability: Grad-CAM (ResNet-50 head)
    • I/O:  DICOM or PNG/JPG → (diagnosis_label, heatmap_overlay as np.ndarray)
    """

    # ------------------------------------------------------------------
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load backbones
        self.resnet    = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().to(self.device)
        self.efficient = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT).eval().to(self.device)

        # Grad-CAM helpers
        self.grad_layer  = self.resnet.layer4[-1]
        self.grad_layer.register_forward_hook(self._save_activation)
        self.grad_layer.register_backward_hook(self._save_gradient)
        self.gradients   = None
        self.activations = None

        # Pre-process 224×224
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.idx_to_class = models.ResNet50_Weights.DEFAULT.meta["categories"]

    # ------------------------------------------------------------------
    def analyze(self, path: str | Path) -> tuple[str, np.ndarray]:
        """
        Returns
        -------
        label : str
            Top-1 ImageNet class (proxy pathology label).
        heatmap : np.ndarray  H×W×3 BGR
            Grad-CAM overlay (zeros if OpenCV missing).
        """
        img_bgr = self._load_image(path)                     # uint8 BGR
        if img_bgr is None:
            return "Image-load-failed", np.zeros((224, 224, 3), dtype=np.uint8)

        inp = self.tf(img_bgr[:, :, ::-1]).unsqueeze(0).to(self.device)  # to RGB

        with torch.no_grad():
            logits_r = self.resnet(inp)
            logits_e = self.efficient(inp)

        probs   = F.softmax(logits_r, dim=1) * 0.5 + F.softmax(logits_e, dim=1) * 0.5
        top_idx = int(probs.argmax(dim=1))
        label   = self.idx_to_class[top_idx]

        # ---- Grad-CAM (only meaningful if cv2 present) -----------------
        if _CV_OK:
            self.resnet.zero_grad()
            logits_r[0, top_idx].backward(retain_graph=True)

            heatmap = self._make_cam(img_bgr.shape[:2])
            overlay = self._overlay_heatmap(img_bgr, heatmap)
        else:
            overlay = np.zeros_like(img_bgr)  # dummy

        return label, overlay

    # ───────────────────────── internal helpers ─────────────────────────
    def _load_image(self, path):
        path = Path(path)
        try:
            if path.suffix.lower() == ".dcm":
                ds  = pydicom.dcmread(str(path))
                img = ds.pixel_array
                if img.ndim == 2:                     # grayscale → 3-ch BGR
                    img = (
                        cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        if _CV_OK else
                        np.stack([img]*3, axis=-1).astype(np.uint8)
                    )
            else:
                if _CV_OK:
                    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                else:
                    img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
            return img
        except Exception as e:
            print(f"⚠️  Failed to load image {path}: {e}")
            return None

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _make_cam(self, size_hw):
        pooled_grads = torch.mean(self.gradients, dim=(0, 2, 3))
        act = self.activations[0]
        for i in range(act.shape[0]):
            act[i] *= pooled_grads[i]

        heatmap = act.sum(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-8
        if _CV_OK:
            heatmap = cv2.resize(heatmap, size_hw[::-1])
        else:  # Pillow resize fallback
            heatmap = np.array(Image.fromarray((heatmap*255).astype(np.uint8)).resize(size_hw[::-1])) / 255.0
        return heatmap

    def _overlay_heatmap(self, img_bgr, heatmap, alpha=0.35):
        if _CV_OK:
            h_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8),
                                        cv2.COLORMAP_JET)
            return cv2.addWeighted(img_bgr, 1 - alpha, h_color, alpha, 0)
        # Pillow fallback (returns original if OpenCV absent)
        return img_bgr
