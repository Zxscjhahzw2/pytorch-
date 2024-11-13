import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter



img="test_img/img.png"
img=Image.open(img)
print(img)
# image=image.convert('RGB')
device=torch.device("cuda")
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])
img=transform(img)
img1=img
class Neuro(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(64,10)

        )

    def forward(self,x):
        x=self.model(x)
        return x

model=torch.load("neuro_39time.pth")
print(model)
img=torch.reshape(img,(1,3,32,32))
img=img.to(device)
model.eval()
with torch.no_grad():
    output=model(img)
print(output)
print(output.argmax(1))
print("1111")

writer=SummaryWriter("logs_test2")
writer.add_image("test",img1)
writer.close()

