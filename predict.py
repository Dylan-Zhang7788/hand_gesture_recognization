import torch
from torchvision import transforms

val_data_transforms = transforms.Compose([
    # transforms.ToTensor(),  # 转化为张量并归一化至[0-1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img,net):
    with torch.no_grad():
        img=img/255
        img = val_data_transforms(img)
        output = net(img.unsqueeze(dim=0))
        # pre_lab = torch.argmax(output, 1)+1
        probability, pre_lab=torch.max(output,1)
        pre_lab=pre_lab[0].cpu().numpy()+1
        return probability,pre_lab