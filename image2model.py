from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import openmesh as om
import torch
import os
import shutil

from networks.v0.shape_engine.model import Embedding
from networks.v0.shape_engine.partSMPLModel import partSMPLModel


import cv2
import pdb
def save_obj(filename,points,faces):
    with open(filename,'w') as fp:
        for v in points: 
            fp.write('v %f %f %f\n' % (v[0].item(), v[1].item(), v[2].item())) 
        for f in faces + 1: 
            fp.write('f %d %d %d %d\n' % (f[0], f[1], f[2],f[3]))

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r


def img_process(image):  #process and convert to Img
    #find white
    mask = image.mean(-1)
    w,h = mask.shape
    mask = (mask>=250)
    masky = mask.sum(0)==mask.shape[0]
    maskx = mask.sum(1)==mask.shape[1]
    x1,x2,y1,y2 = -1,-1,-1,-1

    for i in range(len(maskx)):
        val = maskx[i]
        if not val and x1==-1: # val is false
            x1 = max(i-1,0)
    for i in range(len(maskx)):
        val = maskx[len(maskx)-1-i]
        if not val and x2==-1:
            x2 = min(len(maskx)-i,len(maskx)-1)

    for i in range(len(masky)):
        val = masky[i]
        if not val and y1==-1: # val is false
            y1 = max(i-1,0)
    for i in range(len(masky)):
        val = masky[len(masky)-1-i]
        if not val and y2==-1:
            y2 = min(len(masky)-i,len(masky)-1)
    
    
    image = image[x1:x2,y1:y2,:]
    #print(y1,x1,y2,x2)
    dx = abs(x2-x1)
    dy = abs(y2-y1)
    dif = abs(dx-dy)
    if dx>dy:
        image = cv2.copyMakeBorder(image, 0, 0, int(dif//2), int(dif-dif//2), cv2.BORDER_CONSTANT, value=(255,255,255))
    else:
        image = cv2.copyMakeBorder(image, int(dif//2), int(dif-dif//2), 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    #print(image.shape)
    image = cv2.resize(image,(512,512))
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    return image

#load some info
mesh = om.read_polymesh('./networks/v0/shape_engine/training_dataset/part_smpl/mean.obj')
faces = mesh.face_vertex_indices()
coeffdic_list = []
for x in ['0_1','2_3','4','5']: 
    coeffdic = np.load(os.path.join("./networks/v0/shape_engine/training_dataset","part_smpl","betas",str(x),"coeffdic.npy") , allow_pickle=True).item()
    coeffdic_list.append(coeffdic)

to_tensor = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#load model
embedding = Embedding(shape_dim=200,backbone='resnet50').cuda()
model_CKPT = torch.load(os.path.join("networks/v0/shape_engine/checkpoints","final_jitter.pth"))
embedding.load_state_dict(model_CKPT)
embedding.eval()
normsmpl = partSMPLModel(norm=True) #(1,4,50)/(1,24,3)/(1,3)

class Image2Model:
    def __init__(self):
        pass
    def predict(self, img):
        with torch.no_grad():
            # img = cv2.imread("networks/v0/shape_engine/demo_img/20211114_picture_猪_26.jpg")
            img = img_process(img)
            img = to_tensor(img)
            img = img.cuda()
            pred_beta,pred_pose,trans = embedding(img.unsqueeze(0))
            pred_pose = pred_pose.reshape(1,24,3)
            trans = torch.zeros(trans.shape).cuda()
            zero_pose = torch.ones_like(pred_pose)*0.5
            zero_pose = zero_pose.cuda()
            vertices,pred_joints = normsmpl(beta=pred_beta, pose=zero_pose, trans=trans)
            vertices = vertices.reshape(-1,3).detach().cpu().numpy()
            vertices, _, _  = normalize(vertices)
            return vertices, faces

if __name__ == '__main__':
    i2m = Image2Model()
    img = cv2.imread("networks/v0/shape_engine/demo_img/20211114_picture_猪_26.jpg")
    vertices, faces = i2m.predict(img)
    save_obj("pred.obj", vertices, faces)

