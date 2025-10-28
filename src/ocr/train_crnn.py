#config
import os
from typing import Dict, List
import torch
from torch import nn
from torchvision import transforms
from torch import optim
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from pathlib import Path
import editdistance
DataRoot = Path(__file__).parent.parent.parent/"data"/"ocr_dataset"
TrainImgDir=os.path.join(DataRoot,"images","train")
ValImgDir=os.path.join(DataRoot,"images","val")
TrainLabelDir=os.path.join(DataRoot,"txt","train.txt")
ValLabelDir=os.path.join(DataRoot,"txt","val.txt")
OutputDir=os.path.join(Path(__file__).parent.parent.parent,"outputs")
OutputDir=os.path.join(OutputDir,"train")
os.makedirs(OutputDir,exist_ok=True)

IMG_H = 32
IMG_W = 128
BATCH_SIZE = 64
NUM_WORKERS = 8
LR = 1e-3
EPOCHS = 60
PRINT_EVERY = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青台港澳蒙桂藏宁新使领学警武海空北南广成挂临"
char_to_idx: Dict[str,int] = {c: i+1 for i, c in enumerate(CHARS)}
idx_to_char: Dict[int,str] = {i+1: c for i, c in enumerate(CHARS)}

def text_to_labels(text:str)->List[int]:
    seq=[]
    for ch in text:
        if ch in char_to_idx:
            seq.append(char_to_idx[ch])
        else:
            print(f"[WARN] unknown char '{ch}' in '{text}' -> skipped")
    return seq

def labels_to_text(labels:List[int])->str:
    return "".join(idx_to_char[n] for n in labels if n in idx_to_char)


transform_train =transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_H,IMG_W)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)],p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class PlateOCRDataset(Dataset):
    def __init__(self,img_dir:str,label_dir:str,transform=None):
        super().__init__()
        self.img_dir=img_dir
        self.transform=transform
        self.samples = []
        with open(label_dir,'r',encoding='utf-8')as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                parts=line.split()
                if len(parts)<2:
                    continue
                img_name=parts[0]
                plate=parts[1]
                img_path=os.path.join(self.img_dir,img_name)
                if not os.path.exists(img_path):
                    continue
                self.samples.append((img_name,plate))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_name,plate=self.samples[idx]
        img_path=os.path.join(self.img_dir,img_name)
        img=Image.open(img_path).convert("RGB")
        if self.transform:
            img=self.transform(img)
        labels=text_to_labels(plate)
        labels=torch.LongTensor(labels)
        return img,labels,plate

def collate_fn(batch):
    #using plateOCRDataset to convert
    imgs = torch.stack([b[0] for b in batch], dim=0)  # [B, C, H, W]
    label_list = [b[1] for b in batch]
    raw_texts = [b[2] for b in batch]

    if len(label_list)==0:
        targets=torch.LongTensor([])
        targets_lengths=torch.LongTensor([])
    else:
        targets = torch.cat(label_list).long()
        targets_lengths = torch.LongTensor([len(i) for i in label_list])

    return  imgs,targets,targets_lengths,raw_texts

class CRNN(nn.Module):
    def __init__(self,img_h=IMG_H,num_classes=len(CHARS)+1):
        super().__init__()

        self.cnn=nn.Sequential(
            nn.Conv2d(1,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(2,2),
            #block2
            nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.MaxPool2d(2, 2),
            #block3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        cnn_out_h=img_h//4
        rnn_in=256*cnn_out_h
        self.rnn=nn.LSTM(input_size=rnn_in,hidden_size=256,num_layers=2,
                         batch_first=True,bidirectional=True)
        self.fc=nn.Linear(256*2,num_classes)

    def forward(self,x):
        conv =self.cnn(x)
        b,c,h,w=conv.size()

        conv =conv.permute(0,3,1,2).contiguous()
        conv =conv.view(b,w,c*h)
        rnn_out,_=self.rnn(conv)
        logits =self.fc(rnn_out)
        logits =logits.permute(1,0,2)
        return  logits

def ctc_greedy_decode(logits:torch.Tensor)->List[str]:
    """
        logits: [T, B, C] (raw logits)
        returns list of strings length B
    """
    probs=logits.softmax(dim=2)
    preds = probs.argmax(dim=2).transpose(0, 1).cpu().numpy()
    out_strs=[]
    for seq in preds:
        prev=-1
        out=[]
        for p in seq:
            if p!=prev and p!=0:
                out.append(int(p))
            prev=p
        out_strs.append("".join(idx_to_char[i] for i in out if i in idx_to_char))
    return out_strs


def train_one_epoch(model:nn.Module,loader:DataLoader,criterion,optimizer:optim.Adam,scaler:torch.amp.GradScaler,device):
    model.train()
    total_loss=0.0
    for i ,(imgs,targets,target_lengths,_) in enumerate(loader):
        imgs=imgs.to(device)
        if targets.numel() >0:
            targets =targets.to(device)
        else:
            targets =targets.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits =model(imgs)
            Tt,B,C =logits.shape
            input_lengths =torch.full((B,),fill_value=Tt,dtype=torch.long).to(device)
            log_probs=logits.log_softmax(2)
            loss=criterion(log_probs,targets,input_lengths,target_lengths.to(device))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss +=loss.item()
        if (i+1)%PRINT_EVERY ==0:
            print(f"[train]step{i+1}/{len(loader)} loss={total_loss/(i+1):.4f}")

    return total_loss/len(loader)

def validate(model,loader,criterion,device):
    model.eval()
    total_loss = 0.0
    total_chars = 0
    total_errors = 0
    with torch.no_grad():
        for imgs, targets, target_lengths, texts in loader:
            imgs =imgs.to(device)
            if targets.numel()>0:
                targets = targets.to(device)
            logits = model(imgs)
            Tt, B, C = logits.shape
            input_lengths = torch.full((B,), fill_value=Tt, dtype=torch.long).to(device)
            log_probs = logits.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths.to(device))
            total_loss += loss.item()

            preds=ctc_greedy_decode(logits.cpu())
            for p,t in zip(preds,texts):
                total_errors+=editdistance.eval(p,t)
                total_chars +=max(1,len(t))

    avg_loss = total_loss / len(loader)
    cer = total_errors / total_chars if total_chars > 0 else 0.0
    return avg_loss, cer

def main():
    print("Device:" ,device)
    train_ds = PlateOCRDataset(TrainImgDir, TrainLabelDir, transform=transform_train)
    val_ds = PlateOCRDataset(ValImgDir, ValLabelDir, transform=transform_val)
    print("Dataset sizes:",len(train_ds),len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            collate_fn=collate_fn, prefetch_factor=2, persistent_workers=True)
    model = CRNN(img_h=IMG_H, num_classes=len(CHARS) + 1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_cer = 1.0
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}-----------")
        train_loss=train_one_epoch(model,train_loader,criterion,optimizer,scaler,device)
        val_loss,val_cer = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_CER: {val_cer:.4f}")

        ckpt={
            "epoch":epoch,
            "model_state":model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "val_cer":val_cer
        }
        torch.save(ckpt,os.path.join(OutputDir,f"crnn_epoch{epoch:03d}_cer{val_cer:.4f}.pt"))

        if val_cer<best_cer:
            best_cer=val_cer
            torch.save(model.state_dict(),os.path.join(OutputDir,"best_crnn.pt"))
            print("Saved best_crnn.pt")

    print("Training finished. Best CER:", best_cer)

if __name__=="__main__":
    main()