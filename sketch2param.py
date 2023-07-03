import sys
import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm
import openmesh as om
from networks.v0.embedding.model import Embedding


class Sketch2Param:
    def __init__(self):
        self.to_tensor = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.coeff_database = np.load('./networks/v0/embedding/training_dataset/coeff_database.npz',allow_pickle=True)
        pcamat = np.load('./networks/v0/embedding/training_dataset/pcamat.npy')
        coeffdic = np.load('./networks/v0/embedding/training_dataset/coeffdic.npy',allow_pickle=True).item()
        mesh = om.read_polymesh('./networks/v0/embedding/training_dataset/mean.obj')
        self.mean_points =  mesh.points()
        self.faces = mesh.face_vertex_indices()
        maxmin = np.load('./networks/v0/embedding/training_dataset/maxmin.npy',allow_pickle=True)
        maxmin = maxmin.T
        maxmin = maxmin[:100, [1, 0]]
        c = maxmin[:, 0:1]
        norm_maxmin = maxmin - c
        r = norm_maxmin[:, 1:]
        self.c = c.reshape(-1)
        self.r = r.reshape(-1)
        self.pcamat = pcamat[:100, :]

        self.embedding = Embedding(is_train=False)
        model_CKPT = torch.load("./networks/v0/embedding/checkpoints/dicts/latest.pth")
        self.embedding.load_state_dict({k.replace('module.', ''):v for k,v in model_CKPT.items()})
        self.embedding.cuda().eval()

        self.targetvec = {}
        self.data_type = {
            "human",
            "bear",
            "horse/deer",
            "mouse",
            "dog",
            "cow",
            "pig",
            "monkey",
            "rabbit",
            "hippo",
            "sheep",
            "elephant",
            "fox/wolf",
            "cat",
            "tiger/lion/leopard"
        }
        for target_type in self.data_type:
            targetvec = self.coeff_database[target_type]
            targetvec = targetvec/np.linalg.norm(targetvec,2,axis=1).reshape(-1,1)
            self.targetvec[target_type] = targetvec

    def predict(self, sketch):
        sketch = self.to_tensor(Image.fromarray(sketch).convert('RGB'))
        img_embedding = self.embedding(sketch.unsqueeze(0).cuda())
        coeff = img_embedding.detach().cpu().numpy().reshape(-1)
        vec = coeff.copy()
        coeff = coeff * self.r + self.c
        project = np.dot(self.pcamat.T, coeff)
        newpoints = project.reshape(-1,3) + self.mean_points

        recommend_list = self.get_recommend(vec, is_norm=True)

        return newpoints, self.faces, recommend_list

    def get_by_key(self, target_type, index):
        coeff = self.coeff_database[target_type][index]
        coeff = coeff * self.r + self.c
        project = np.dot(self.pcamat.T, coeff)
        newpoints = project.reshape(-1,3) + self.mean_points
        return newpoints

    def get_nearest(self, sketch, target_type="human", k_n=16):
        sketch = self.to_tensor(Image.fromarray(sketch).convert('RGB'))
        img_embedding = self.embedding(sketch.unsqueeze(0).cuda())
        vec = img_embedding.detach().cpu().numpy().reshape(-1)
        vecnorm = np.linalg.norm(vec)
        vec = vec/vecnorm
        
        targetvec = self.targetvec[target_type]
        # print(targetvec.shape)
        cosine =  np.dot(vec, targetvec.T).reshape(-1)
        # print(cosine.shape)
        # maxindex = np.argmax(cosine)

        indices = cosine.argsort()[-k_n:][::-1]
        maxindex = indices[0]
        recommend_list = ["renders/" + target_type.replace("/", "+") + "/" + str(index) + ".png" for index in indices[1:]]
        # print(maxindex)
        coeff = self.coeff_database[target_type][maxindex]
        coeff = coeff * self.r + self.c
        project = np.dot(self.pcamat.T, coeff)
        newpoints = project.reshape(-1,3) + self.mean_points
        return newpoints, self.faces, recommend_list
    
    def get_recommend(self, vec, is_norm=False):
        # sketch = self.to_tensor(Image.fromarray(sketch).convert('RGB'))
        # img_embedding = self.embedding(sketch.unsqueeze(0).cuda())
        # vec = img_embedding.detach().cpu().numpy().reshape(-1)
        if is_norm:
            vecnorm = np.linalg.norm(vec)
            vec = vec/vecnorm
        recommend_list = []
        for target_type in self.data_type:
            targetvec = self.targetvec[target_type]
            # print(targetvec.shape)
            cosine =  np.dot(vec, targetvec.T).reshape(-1)
            # print(cosine.shape)
            maxindex = np.argmax(cosine)
            # print(maxindex)
            # coeff = self.coeff_database[target_type][maxindex]
            # coeff = coeff * self.r + self.c
            # project = np.dot(self.pcamat.T, coeff)
            # newpoints = project.reshape(-1,3) + self.mean_points
            recommend_list.append("renders/" + target_type.replace("/", "+")  + "/" + str(maxindex) + ".png")
        return recommend_list


if __name__ == '__main__':
    s2p = Sketch2Param()
    root = "networks/v0/embedding/training_dataset/datasets_final1"
    for f in os.listdir(root):
        if "_plus_" in f:
            continue
        # key = f.replace("#U", "\\u").encode("utf-8").decode("unicode_escape")[0:-12]
        # coeff = coeffdic[key][:100]
        # project = np.dot(pcamat.T, coeff)
        # newpoints = project.reshape(-1,3) + points
        # newmesh = om.PolyMesh(points=newpoints,face_vertex_indices=faces)
        # om.write_mesh(os.path.join("networks/v0/embedding/outputs", f.replace(".png", "_pca.obj")), newmesh)

        img = Image.open(os.path.join(root, f))
        img.save("networks/v0/embedding/outputs/" + f)
        img = np.array(Image.open(os.path.join(root, f)))
        vertices, faces = s2p.get_nearest(img, target_type="cat")
        newmesh = om.PolyMesh(points=vertices,face_vertex_indices=faces)
        om.write_mesh(os.path.join("networks/v0/embedding/outputs", f.replace(".png", "_pred.obj")), newmesh)
        break
    '''
    root = "./networks/v0/embedding/training_dataset/NORMAL"
    for f in os.listdir(root):
        img = np.array(Image.open(os.path.join(root, f)))
        vertices, faces = n2p.predict(img)
        newmesh = om.PolyMesh(points=vertices,face_vertex_indices=faces)
        om.write_mesh(os.path.join("networks/v0/embedding/outputs", f.replace(".png", "_pred.obj")), newmesh)
        break
    '''