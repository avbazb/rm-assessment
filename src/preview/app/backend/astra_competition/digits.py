"""Weak character evidence for lamp pairing; never an independent armor detector."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "models/competition-digits-v2"


def letterbox(gray, size=64):
    scale = min(size / gray.shape[1], size / gray.shape[0])
    width, height = max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))
    output = np.full((size, size), 114, np.uint8)
    x, y = (size-width)//2, (size-height)//2
    output[y:y+height, x:x+width] = cv2.resize(gray, (width, height))
    return output


def character_mask(gray, color=None):
    """Central connected glyph. Blank, solid panels and edge lamps are not glyphs."""
    height, width = gray.shape
    center = gray[:, round(width*.12):round(width*.88)]
    if center.size == 0 or float(np.percentile(center, 98)-np.percentile(center, 10)) < 12:
        return None, "low_contrast"
    _, binary = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    options = []
    colored_spill = False
    if color is not None:
        # Strongly saturated lamp spill can become a high-similarity blob in
        # grayscale. A real numeral may be bright and tinted by team lighting;
        # require dominant-channel intensity over twice the opposite channel,
        # as well as lamp-like brightness/chroma, before withholding support.
        crop = color[:, round(width*.12):round(width*.88)].astype(np.int16)
        b, g, r = cv2.split(crop)
        chromatic = (((b-r >= 25) & (b-g >= 12) & (b > 2*r)) |
                     ((r-b >= 25) & (r-g >= 12) & (r > 2*b))) & (crop.max(axis=2) >= 85)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        fill = area/max(w*h, 1)
        cx, cy = centroids[i]
        if not (.20*height <= h <= .98*height and .06 <= w/h <= 1.4 and .09 <= fill <= .86):
            continue
        if abs(cx-center.shape[1]/2) > center.shape[1]*.32 or abs(cy-height/2) > height*.35:
            continue
        # A clipped component is likely another lamp / neighboring plate.
        if x == 0 or x+w >= center.shape[1] or y == 0 or y+h >= height:
            continue
        if color is not None and np.count_nonzero(chromatic & (labels == i)) > area/2:
            colored_spill = True
            continue
        component = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        normalized = np.zeros((48, 32), np.uint8)
        scale = min(28/w, 44/h)
        nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
        left, top = (32-nw)//2, (48-nh)//2
        normalized[top:top+nh,left:left+nw] = cv2.resize(component,(nw,nh),interpolation=cv2.INTER_NEAREST)
        rank = area / (1+abs(cx-center.shape[1]/2))
        options.append((rank, normalized))
    if not options:
        return None, "colored_lamp_spill" if colored_spill else "no_centered_glyph"
    return max(options, key=lambda item:item[0])[1], "centered_glyph"


def plate_transform(image, corners, valid_source=None):
    """One shared warp and clipping decision for gray and color evidence."""
    q = np.asarray(corners, np.float32)
    length = (np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
    gap = (np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2
    if length < 8 or gap < 10:
        return None, None, "too_small"
    # The size hypothesis affects this character crop only, never a metric pose.
    width, height = (148 if gap/length > 3.2 else 90), 84
    destination = np.array(((0,24),(width-1,24),(width-1,60),(0,60)),np.float32)
    transform = cv2.getPerspectiveTransform(q,destination)
    if valid_source is None:
        valid_source = np.full(image.shape[:2],255,np.uint8)
    valid = cv2.warpPerspective(valid_source,transform,(width,height))
    if np.mean(valid[:,width//5:4*width//5] == 255) < .95:
        return None, None, "crop_clipped"
    return transform, (width,height), "available"


def rectify_plate(image, corners, *, grayscale=True):
    """Include the number above/below the luminous endpoints, in TL/TR/BR/BL order."""
    transform, size, reason = plate_transform(image, corners)
    if transform is None:
        return None, reason
    source = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if grayscale else image
    return cv2.warpPerspective(source,transform,size), reason


class FrameDigitEvidence:
    """Caches belong to one detect call, never to a shared detector or video buffer."""
    def __init__(self, evidence, image):
        self.evidence, self.image = evidence, image
        self.gray = None
        self.valid = None

    def pair(self, image, corners):
        if image is not self.image:
            raise ValueError("frame evidence used with a different image")
        if self.valid is None:
            self.valid = np.full(image.shape[:2],255,np.uint8)
        transform, size, reason = plate_transform(image,corners,self.valid)
        if transform is None:
            return self.evidence.classify(None,reason)
        if self.gray is None:
            self.gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = cv2.warpPerspective(self.gray,transform,size)
        color = cv2.warpPerspective(image,transform,size)
        return self.evidence.classify(gray,reason,color)

    def box(self, image, bbox):
        if image is not self.image:
            raise ValueError("frame evidence used with a different image")
        x1,y1,x2,y2 = np.rint(bbox).astype(int)
        x1,y1 = max(0,x1),max(0,y1)
        x2,y2 = min(image.shape[1],x2),min(image.shape[0],y2)
        if x2-x1<10 or y2-y1<8:
            return self.evidence.classify(None,"too_small")
        if self.gray is None:
            self.gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return self.evidence.classify(self.gray[y1:y2,x1:x2])


class DigitEvidence:
    def __init__(self, method="template", threshold=.25, assets=ASSETS):
        if method not in ("template","model") or not 0 <= threshold <= 1:
            raise ValueError("invalid digit method or threshold")
        self.method, self.threshold = method, threshold
        assets = Path(assets)
        if not (assets/'manifest.json').is_file():
            raise ValueError("未提供数字识别资源，请使用 digit_method='off' 或提供完整资源目录")
        self.manifest = json.loads((assets/"manifest.json").read_text(encoding="utf-8"))
        self.labels = self.manifest["classes"]
        name = "templates.npz" if method == "template" else "weights.pt"
        payload = (assets/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.manifest["files"][name]:
            raise ValueError("digit asset hash mismatch")
        if method == "template":
            with np.load(assets/name,allow_pickle=False) as bank:
                self.templates = bank["masks"].astype(np.float32).reshape(-1,48*32)
                self.template_labels = bank["labels"]
            self.areas = self.templates.sum(axis=1)
        else:
            # This branch is never entered by the pure traditional mode.
            import torch
            import torch.nn as nn
            self.torch = torch
            self.net = nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1024,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,7))
            self.net.load_state_dict(torch.load(assets/name,map_location="cpu",weights_only=True))
            self.net.eval()

    def classify(self, gray, reason="available", color=None):
        result = dict(label=None,score=None,method=self.method,supported=False,status=reason,
                      score_kind="template_similarity" if self.method=="template" else "class_confidence",
                      threshold=.48 if self.method=="template" else self.threshold)
        if gray is None:
            return result
        mask, structure = character_mask(gray,color)
        result["status"] = structure
        if mask is None:
            return result
        if self.method == "template":
            flat = mask.astype(np.float32).reshape(-1)
            # Tiny vector products should not launch a multithreaded BLAS team per candidate.
            similarities = 2*np.einsum('ij,j->i',self.templates,flat,optimize=False)/np.maximum(self.areas+flat.sum(),1)
            i = int(similarities.argmax())
            label, score = int(self.template_labels[i]),float(similarities[i])
        else:
            tensor = self.torch.from_numpy(letterbox(gray).astype(np.float32)[None,None]/255)
            with self.torch.inference_mode():
                probabilities = self.net(tensor).softmax(1).numpy()[0]
            label,score = int(probabilities.argmax()),float(probabilities.max())
        result.update(label=self.labels[label],score=score,supported=score>=result["threshold"])
        result["status"] = "weak_support" if result["supported"] else "uncertain_character"
        return result

    def for_frame(self,image):
        return FrameDigitEvidence(self,image)

    def pair(self,image,corners):
        gray,reason = rectify_plate(image,corners)
        color = rectify_plate(image,corners,grayscale=False)[0] if gray is not None else None
        return self.classify(gray,reason,color)

    def box(self,image,bbox):
        x1,y1,x2,y2 = np.rint(bbox).astype(int)
        x1,y1 = max(0,x1),max(0,y1)
        x2,y2 = min(image.shape[1],x2),min(image.shape[0],y2)
        if x2-x1<10 or y2-y1<8:
            return self.classify(None,"too_small")
        return self.classify(cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY))
