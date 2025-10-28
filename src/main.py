import os
import argparse
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
try:
    from ultralytics import YOLO
except Exception:
    raise ImportError("ultralytics is required for YOLO detection. Install with: pip install ultralytics")

IMG_H = 32
IMG_W = 128
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青台港澳蒙桂藏宁新使领学警武海空北南广成挂临"

char_to_idx = {c: i+1 for i, c in enumerate(CHARS)}
idx_to_char = {i+1: c for i, c in enumerate(CHARS)}

transform_infer = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CRNN(nn.Module):
    def __init__(self, img_h=IMG_H, num_classes=len(CHARS)+1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        cnn_out_h = img_h // 4
        rnn_in = 256 * cnn_out_h
        self.rnn = nn.LSTM(input_size=rnn_in, hidden_size=256, num_layers=2,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256*2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2).contiguous()
        conv = conv.view(b, w, c * h)
        rnn_out, _ = self.rnn(conv)
        logits = self.fc(rnn_out)
        logits = logits.permute(1, 0, 2)
        return logits

@torch.no_grad()
def ctc_greedy_decode(logits: torch.Tensor) -> List[str]:
    probs = logits.softmax(dim=2)
    preds = probs.argmax(dim=2).transpose(0,1).cpu().numpy()
    out_texts = []
    for seq in preds:
        prev = -1
        out = []
        for p in seq:
            if p != prev and p != 0:
                out.append(int(p))
            prev = p
        out_texts.append("".join(idx_to_char[i] for i in out if i in idx_to_char))
    return out_texts

class PlateRecognizer:
    def __init__(self,yolo_weights:str=r"models/best.pt",
                 crnn_weights:str=r"models/best_crnn.pt",
                 device:Optional[str]=None,
                 Yolo_conf:float=0.25):
        self.device= device or ("cuda" if torch.cuda.is_available() else"cpu")
        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
        self.detector=YOLO(yolo_weights)
        self.yolo_conf=Yolo_conf
        if not  os.path.exists(crnn_weights):
            raise FileNotFoundError(f"CRNN weights not found: {crnn_weights}")
        self.crnn =CRNN(img_h=IMG_H,num_classes=len(CHARS)+1).to(self.device)
        state=torch.load(crnn_weights,map_location=self.device)
        if isinstance(state,dict) and 'model_state' in state:
            self.crnn.load_state_dict(state['model_state'])
        else:
            self.crnn.load_state_dict(state)
        self.crnn.eval()

    def _detect_plate_boxes(self,img:np.ndarray):
        """Run YOLO detection and return boxes in xyxy, confidences, and class ids."""
        results = self.detector.predict(img, conf=self.yolo_conf, verbose=False)
        if not results:
            return []
        r=results[0]
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((0,))
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((0,))
        return [(xyxy[i], float(conf[i]), int(cls[i])) for i in range(len(xyxy))]

    def _crop_plate(self,pil_img:Image.Image,box_xyxy:np.ndarray)->Image.Image:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(pil_img.width, x2)
        y2 = min(pil_img.height, y2)
        crop = pil_img.crop((x1, y1, x2, y2))
        return crop

    @torch.no_grad()
    def recognize(self, image_path: str, return_crop: bool = False) -> Tuple[Optional[str], Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
        """
        Recognize plate text from an image.
        Returns (text, crop_image or None, bbox_xyxy or None)
        """
        if not  os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        pil_img = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_img)  # YOLO expects numpy RGB
        boxes = self._detect_plate_boxes(img_np)
        # filter class 0 (LicensePlate) and choose highest conf
        boxes = [b for b in boxes if b[2] == 0]
        if len(boxes) == 0:
            print("[INFO] No plate detected.")
            return None, None, None
        boxes.sort(key=lambda  b:b[1],reverse=True)
        box,conf,cls_id =boxes[0]
        crop = self._crop_plate(pil_img, box)
        # preprocess for CRNN
        img_tensor = transform_infer(crop).unsqueeze(0).to(self.device)  # [1,1,H,W]
        logits = self.crnn(img_tensor)  # [T,B,C]
        pred_texts = ctc_greedy_decode(logits)
        pred = pred_texts[0] if len(pred_texts) > 0 else ""
        if return_crop:
            return pred, crop, (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        else:
            return pred, None, None


# Utility: find next available numeric index in output dir
def next_available_index(out_dir: str) -> int:
    """
    中文:
    - 清空输出文本目录中的所有 .txt 文件，并将编号重置为 0。
    - 每次调用都会执行清理，以确保新的结果从 0 开始编号。

    English:
    - Clear all .txt files in the output text directory and reset the index to 0.
    - Each call performs cleanup to ensure new results start numbering from 0.
    """
    text_dir = os.path.join(out_dir, "texts")
    os.makedirs(text_dir, exist_ok=True)
    for name in os.listdir(text_dir):
        if name.lower().endswith(".txt"):
            try:
                os.remove(os.path.join(text_dir, name))
            except Exception:
                pass
    return 0

def run_on_image(recognizer:PlateRecognizer,image_path:str,save_crop:bool=True,out_dir:str="outputs",index:Optional[int]=None):
    text, crop, bbox = recognizer.recognize(image_path, return_crop=save_crop)
    loss=0
    print(f"Image: {image_path}\nPlate: {text}")
    # 创建子目录
    text_dir = os.path.join(out_dir, "texts")
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(text_dir, exist_ok=True)
    if save_crop:
        os.makedirs(crops_dir, exist_ok=True)
    # 自动递增编号并保存文本
    idx = next_available_index(out_dir) if index is None else index
    txt_path = os.path.join(text_dir, f"{idx}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        if text is None:
            f.write("")
            loss=1
        else:
            f.write(text)
    print(f"Saved text: {txt_path}")
    if save_crop and crop is not None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        crop_path = os.path.join(crops_dir, f"{base}_plate.jpg")
        crop.save(crop_path)
        print(f"Saved crop: {crop_path}")
    return loss


def get_all_images(root_dir, exts=None):
    """ get all the dir of pictures"""
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_list = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                img_list.append(os.path.join(root, f))
    return sorted(img_list)



def main():

    current_dir = Path(__file__).parent.parent
    print(current_dir)
    parser = argparse.ArgumentParser(description="License Plate OCR Pipeline")
    parser.add_argument("--image", type=str, help="input single image path")
    parser.add_argument("--dir", type=str, help="input image directory (jpg/png/bmp)")
    parser.add_argument("--yolo", type=str, default=current_dir/"models"/"best.pt", help="YOLO model path")
    parser.add_argument("--crnn", type=str, default=current_dir/"models"/"best_crnn.pt", help="CRNN model path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--save-crop",default=True, action="store_true", help="save cropped plates")
    parser.add_argument("--out-dir", type=str, default=current_dir/"outputs", help="output directory")
    args = parser.parse_args()

    if not args.image and not args.dir:
        inp = input("请输入图片路径或文件夹路径: ").strip()
        if os.path.isfile(inp):
            args.image = inp
        elif os.path.isdir(inp):
            args.dir = inp
        else:
            print(f"无效输入，访问默认位置{current_dir/'input'/'pictures'}")
            args.dir = current_dir/"input"/"pictures"
            
    recognizer = PlateRecognizer(
        yolo_weights=args.yolo,
        crnn_weights=args.crnn,
        device=args.device,
        Yolo_conf=args.conf
    )

    base_idx = next_available_index(args.out_dir)
    print(f"{os.path.join(args.out_dir, 'texts')}，set index to 0")
    loss=0
    if args.image:
        # 单张图片
        loss=run_on_image(recognizer, args.image, save_crop=args.save_crop, out_dir=args.out_dir, index=base_idx)
        if loss==1:
            print("识别失败")
        else:
            print("识别成功")
    else:
        # 扫描整个文件夹
        all_imgs = get_all_images(args.dir)
        if not all_imgs:
            print(f"未找到任何图片文件于 {args.dir}")
            return
        for i, img_path in enumerate(all_imgs):
            loss+=run_on_image(recognizer, img_path, save_crop=args.save_crop, out_dir=args.out_dir, index=base_idx + i)
        print(f"成功率为{all_imgs.__len__()-loss}/{all_imgs.__len__()}")
if __name__ == "__main__":
    main()