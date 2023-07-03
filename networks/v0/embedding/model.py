import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision

class Embedding(nn.Module):
    def __init__(self, is_train):
        super(Embedding, self).__init__()
        # self.resnet = InceptionResnetV1(pretrained='vggface2')
        self.resnet = torchvision.models.resnet50(pretrained=False)
        if is_train == True:
            modelpath = "/home/zhananjin/.cache/torch/resnet50-19c8e357.pth"
            self.resnet.load_state_dict(torch.load(modelpath))
        self.resnet.train()
        self.mlp = MLP(1000, 1024, 100)

    def forward(self, img):
        out = self.resnet(img)
        out = self.mlp(out)
        return out

class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        self.hidden0 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, 512)
        self.hidden2 = torch.nn.Linear(512, 256)
        self.predict = torch.nn.Linear(256, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden0(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.predict(x)
        return x