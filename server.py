
import re
import io
import os
import time
import json
import trimesh
import base64
import random
import shutil
import numpy as np
from PIL import Image
from flask import Flask, send_from_directory, make_response, jsonify, request
import openmesh as om
import DracoPy
import cv2
from sketch2param import Sketch2Param
from image2uv2 import Image2UV2
from sketch2texture_stylegan import Sketch2TextureStyle
from pose_builder import PoseBuilder
import requests
from skimage import io
from SMPLModel_cat_eye_simple import SMPLModel_eye
import torch

app = Flask(__name__, static_folder='', static_url_path='')
s2p = Sketch2Param()
# s2p_q = Sketch2Param_QUAD()
s2ts = Sketch2TextureStyle()
i2uv2 = Image2UV2()
pb = PoseBuilder()
eb = SMPLModel_eye(beta_norm=True, theta_norm=False)
template_mesh = om.read_polymesh('template/m.obj')
beta = torch.ones((1,200)).cuda()*0.5
theta = torch.zeros((1,72)).cuda()
trans = torch.zeros((1,3)).cuda()

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r

@app.route('/')
def root():
    return send_from_directory("./", "index.html")

'''
for qt
'''
@app.route('/generate_shape', methods=["POST"])
def generate_shape():
    data = request.get_data()
    json_data = json.loads(data)
    image = np.array(json_data["front_sketch"], dtype=np.uint8).reshape(800, 800, 3)
    predict_type = json_data["predict_type"]
    target_category = json_data["target_category"]
    image = 255 - image
    image[image<255] = 0
    name = str(time.time())
    image = Image.fromarray(image).resize((512, 512))
    sketch_path = "gallery/sketch/" + name + ".png"
    image.save(sketch_path)

    start = time.time()
    obj_path = "gallery/obj/" + name + "_p.obj"
    if predict_type == "B":
        if target_category == "none":
            vertices, faces, recommend_list = s2p.predict(np.array(image))
        else:
            vertices, faces, recommend_list = s2p.get_nearest(np.array(image), target_category)
    else:
        # vertices, faces, recommend_list = s2p_q.predict(np.array(image))
        if target_category == "none":
            vertices, faces, recommend_list = s2p_q.predict(np.array(image))
        else:
            vertices, faces, recommend_list = s2p_q.get_nearest(np.array(image), target_category)
    end = time.time()
    print("shape:", end - start)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.vertices, _, _ = normalize(mesh.vertices)
    mesh.export(obj_path)
    newmesh = om.PolyMesh(points=vertices,face_vertex_indices=faces)
    newmesh.request_vertex_normals()
    newmesh.update_vertex_normals()
    vertex_normals = newmesh.vertex_normals()
    # print(recommend_list)
    return {"vertices_list": [mesh.vertices.tolist(), vertex_normals.tolist()], "faces_list": [template_mesh.face_vertex_indices().tolist()], "recommend_list": recommend_list}

@app.route('/generate_texture', methods=["POST"])
def generate_texture():
    name = str(time.time())
    data = request.get_data()
    json_data = json.loads(data)
    image = np.array(json_data["front_sketch"], dtype=np.uint8).reshape(800, 800, 3)
    Image.fromarray(image).save("temp.png")
    vertices_list = json_data["vertices_list"]
    image = Image.fromarray(image).resize((512, 512))
    sketch_path = "gallery/painting/" + name + ".png"
    image.save(sketch_path)

    in_uv_coords = np.array(vertices_list[1])
    vertex_normals = np.array(vertices_list[2])
    indices = np.where(vertex_normals[:, 2] < 0)
    in_uv_coords[indices] = 0
    in_texture = io.imread("temp.png") / 255.
    image = i2uv2.transfer_single_uv(in_uv_coords=in_uv_coords, in_uv_faces=template_mesh.face_vertex_indices().copy(), in_texture=in_texture, side=512, new_path="gallery/part_texture/" + name + ".png")
    cv2.imwrite("gallery/part_texture/" + name + ".png", image)
    
    texture_path = "gallery/texture/" + name + ".png"
    start = time.time()
    pred_texture = s2ts.predict(Image.fromarray( cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    end = time.time()
    print("texture:", end - start)
    kernel = np.ones((5,5),np.uint8) 
    pred_texture = cv2.dilate(np.array(pred_texture), kernel, iterations=1)
    Image.fromarray(pred_texture).save(texture_path)
    pred_texture = np.array(pred_texture)
    return {"texture_list": [pred_texture.reshape(-1).tolist()]}

@app.route('/generate_pose', methods=["POST"])
def generate_pose():
    name = str(time.time())
    data = request.get_data()
    json_data = json.loads(data)
    predict_type = json_data["predict_type"]
    vertices_list = json_data["vertices_list"]
    T_vertices = np.array(vertices_list[0])
    key = json_data["filename"]
    if "none" in key or predict_type == "Q":
        pose = theta
    else:
        key = json_data["filename"][0:-4]
        pose = pb.build(T_vertices=T_vertices, k=key).reshape((1,72))
        pose = torch.Tensor(pose).cuda()


    body_mesh_points = torch.Tensor(T_vertices).reshape(1, -1, 3).cuda()
    body_mesh_points, eyes = eb(beta=None, pose=pose, trans=trans, TposeModel=body_mesh_points)
    
    body_mesh_points = body_mesh_points.reshape(-1,3).detach().cpu().numpy()
    newmesh = om.PolyMesh(points=body_mesh_points, face_vertex_indices=template_mesh.face_vertex_indices())
    newmesh.request_vertex_normals()
    newmesh.update_vertex_normals()
    vertex_normals = newmesh.vertex_normals()
    vertices = newmesh.points()

    eye_mesh = eyes[0]
    eye_mesh.request_vertex_normals()
    eye_mesh.update_vertex_normals()
    eye_vertices = eye_mesh.points()
    eye_vertex_normals = eye_mesh.vertex_normals()

    return {"vertices_list": [vertices.tolist(), vertex_normals.tolist()], "eye_vertices_list": [eye_vertices.tolist(), eye_vertex_normals.tolist()]}
    

@app.route('/generate_by_key', methods=["POST"])
def generate_by_key():
    name = str(time.time())
    data = request.get_data()
    json_data = json.loads(data)
    predict_type = json_data["predict_type"]
    # vertices_list = json_data["vertices_list"]
    # T_vertices = np.array(vertices_list[0])
    key = json_data["key"]
    name_list = key.split("/")
    target_type = name_list[1].replace("+", "/")
    index = int(name_list[2][:-4])
    # print(target_type, index)
    if predict_type == "Q":
        vertices = s2p_q.get_by_key(target_type, index)
    else: 
        vertices = s2p.get_by_key(target_type, index)
    vertices, _, _ = normalize(vertices)
    newmesh = om.PolyMesh(points=vertices, face_vertex_indices=template_mesh.face_vertex_indices())
    newmesh.request_vertex_normals()
    newmesh.update_vertex_normals()
    vertex_normals = newmesh.vertex_normals()
    return {"vertices_list": [vertices.tolist(), vertex_normals.tolist()]}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)