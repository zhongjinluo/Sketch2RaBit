# -*- coding: utf-8 -*-
import os
import shutil
import cv2
import numpy as np
from mesh import load_obj_mesh, save_obj_mesh, save_obj_mesh_with_rgb
from mesh_util import save_obj_mesh_with_color, save_obj_mesh_with_color_uv
import trimesh
from tqdm import tqdm
from quad2trimesh import tranf_quad2tri
import argparse

# Todo: making a mask to save all original part in case of affecting details. Then, dilating around.

# def get_mask(size=2048):

#     mask_img = cv2.imread('m.bmp')
#     img_size = mask_img.shape[0]
#     if img_size != size:
#         mask_img = cv2.resize(mask_img, (size, size), interpolation=cv2.INTER_CUBIC)
#     mask_img[mask_img > 0] = 255
#     # mask_img = mask_img[0]
    
#     return np.where(mask_img)

# def dilate_img(img, is_mask=True):

#     mask_idx = np.where(img)

#     kernel = np.ones((5,5),np.uint8) 
#     # kernel2 = np.ones((3, 3), np.uint8)

#     dilate_img = cv2.dilate(img, kernel, iterations=2)

#     if is_mask:
#         dilate_img[mask_idx] = img[mask_idx]
    
#     return dilate_img

def get_mask(size=2048):

    mask_img = cv2.imread('m.bmp')
    img_size = mask_img.shape[0]
    if img_size != size:
        mask_img = cv2.resize(mask_img, (size, size), interpolation=cv2.INTER_CUBIC)
    mask_img[mask_img > 0] = 255
    # mask_img = mask_img[0]
    kernel = np.ones((2,2),np.uint8) 
    # kernel2 = np.ones((3, 3), np.uint8)

    # e_img = cv2.erode(mask_img, kernel, iterations=1)
    e_img = mask_img
    return np.where(e_img)

def dilate_img(mask_idx, size, img_path, output_path):

    img = cv2.imread(img_path)

    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    
    
#     mask_idx = np.where(img)

    kernel = np.ones((10,10),np.uint8) 
    # kernel2 = np.ones((3, 3), np.uint8)

    dilate_img = cv2.dilate(img, kernel, iterations=1)

    # dilate_img[mask_idx] = img[mask_idx]
    
    new_img = dilate_img - img
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    cv2.imwrite(output_path, dilate_img)

def dilate_save_img(is_mask, size, img_path, output_path):

#     img = cv2.imread(img_path)

#     if img.shape[0] != size:
#         img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

#     mask_idx = np.where(img)

#     kernel = np.ones((5,5),np.uint8) 
#     # kernel2 = np.ones((3, 3), np.uint8)

#     dilate_img = cv2.dilate(img, kernel, iterations=1)

#     if is_mask:
#         dilate_img[mask_idx] = img[mask_idx]
    
# #     new_img = dilate_img - img
# #     new_img[new_img < 0] = 0
# #     new_img[new_img > 255] = 255
# #     show(new_img)
#     cv2.imwrite(output_path, dilate_img)

    mask_idx = get_mask(size)

    dilate_img(mask_idx[:2], size, img_path, output_path)


def reflect_result(uvPath='./data/mm.bmp', objPath='./data/m.obj', mesh_path = 'data/ms.obj'):

    # load obj file and params
    # print("objPath:",objPath)
    verts, faces, uvs, face_uvs = load_obj_mesh(objPath, with_texture=True)

    # side = 2048

    uv_img = cv2.imread(uvPath)
    height, width, channels = uv_img.shape
    assert height == width, 'uv should be a square!'
    side = width
    # cv2.imwrite(uv_render_path, uv_img)
    uv_img = cv2.cvtColor(uv_img, cv2.COLOR_BGR2RGB)
    color_uv = np.zeros((int(face_uvs.max())+1, 3))
    # for index, value in enumerate(color_uv):
    for index in range(uvs.shape[0]):
        x = int(uvs[index][0] * side)
        y = int((1-uvs[index][1]) * side)
        color_uv[index] = uv_img[y][x] / 255.0

    color = np.zeros((int(faces.max())+1, 3))
    for i in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            color[faces[i,j]] = color_uv[face_uvs[i,j]]

    # save new obj file with texture
    save_obj_mesh_with_color_uv(mesh_path, verts, faces, uvs, color)

def load_obj_mesh_color(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            if len(values) > 4:
                v = list(map(float, values[1:7]))
            else:
                v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1
    
    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    return vertices, faces


def interpolation(uvPath='./data/s2.bmp', objPath='./solve/mc.obj', meshPath = 'solve/ms.obj', outPath = 'solve/ms1.obj'):

    # load obj file and params
    verts, faces, uvs, face_uvs = load_obj_mesh_color(objPath, with_texture=True)
    points = []
    
    for i, p in enumerate(verts):
        if p[4] == 0 and p[5] == 0 and p[3] == 0:
            points.append(i)
    
    # load mesh using trimesh, calculate its laplacian matrix
    mesh = trimesh.load(meshPath, process=False)
    # print(mesh.vertices.shape)
    matrix = trimesh.smoothing.laplacian_calculation(mesh)
    # print(matrix.shape)
    m_v, _, _, _ = load_obj_mesh_color(meshPath, with_texture=True)
    # print(m_v.shape)
    colors = m_v[:, 3:]
    # print(colors.shape)
    ncolors = matrix.dot(colors)
    # print(ncolors)
#     for i in range(len(ncolors)):
#         if i in points:
#             colors[i] = ncolors[i]
    
    # save new obj file with texture
    v, f, vu, fu = load_obj_mesh(meshPath, with_texture=True)
    save_obj_mesh_with_color_uv(outPath, v, f, vu, ncolors)

def interpolation4all(meshPath = 'solve/ms.obj', outPath = 'solve/ms1.obj'):

    # load mesh using trimesh, calculate its laplacian matrix
    mesh = trimesh.load(meshPath, process=False)
    matrix = trimesh.smoothing.laplacian_calculation(mesh)
    m_v, _, _, _ = load_obj_mesh_color(meshPath, with_texture=True)
    colors = m_v[:, 3:]
    ncolors = matrix.dot(colors)

    # save new obj file with texture
    v, f, vu, fu = load_obj_mesh(meshPath, with_texture=True)
    save_obj_mesh_with_color_uv(outPath, v, f, vu, ncolors)

def after_process(args):

    current_dir = './'
    # right_obj_path = os.path.join(current_dir, 'right.obj')
    right_obj_path = os.path.join(current_dir, 'right2.obj') # new uv from back
    verts_r, faces_r, norms_r, face_norms_r, uvs_r, face_uvs_r = load_obj_mesh(right_obj_path, with_texture=True, with_normal=True)

    if False: # args.folder == 'no': # processing single image
        dilate_save_img(args.is_mask, args.size, args.input, args.output)
    else:
        obj_root = args.test_mesh_path
        uv_root = os.path.join(args.exp_dir, 'inference_results')
        file_list = os.listdir(uv_root)
        file_list = tqdm(file_list)
        for i in file_list:
            uv_path = os.path.join(uv_root, i)
            name = i[:-4]
            file_list.set_description("Processing {}".format(name))

            # prepare
            outdir = os.path.join(args.exp_dir, 'colored_mesh')
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            obj_path = os.path.join(outdir, name + '.obj')
            if args.is_quad:
                # tranf_quad2tri('{}{}_pred.obj'.format(obj_root, name), obj_path, uvs_r, faces_r, face_uvs_r, norms_r, face_norms_r)
                tranf_quad2tri('{}{}.obj'.format(obj_root, name), obj_path, uvs_r, faces_r, face_uvs_r, norms_r, face_norms_r)
            else:
                shutil.copy('{}{}_pred.obj'.format(obj_root, name), obj_path)

            # dilate uv
            if not os.path.isdir('tmp'):
                os.mkdir('tmp')
            dilate_save_img(True, args.size, uv_path, 'tmp/tmp.bmp')

            # reflect texture on mesh
            reflect_result(uvPath='tmp/tmp.bmp', objPath=obj_path, mesh_path = obj_path)

            # interpolation
            interpolation4all(obj_path, obj_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=2048)
    parser.add_argument('-i', '--input', type=str, default='m.jpg')
    parser.add_argument('-o', '--output', type=str, default='mm.jpg')
    parser.add_argument('-f', '--folder', type=str, default='no')
    parser.add_argument('-ob', '--obj_root', type=str, default='no')
    parser.add_argument('-od', '--outdir', type=str, default='/home/shengcai/process/combine_jg_sc/colored_mesh')
    parser.add_argument('-m', '--is_mask', type=bool, default=True)
    parser.add_argument('-q', '--is_quad', type=bool, default=True)
    args = parser.parse_args()

    current_dir = './'
    right_obj_path = os.path.join(current_dir, 'right.obj')
    verts_r, faces_r, norms_r, face_norms_r, uvs_r, face_uvs_r = load_obj_mesh(right_obj_path, with_texture=True, with_normal=True)

    if args.folder == 'no': # processing single image
        dilate_save_img(args.is_mask, args.size, args.input, args.output)
    else:
        # obj_root = '/home/shengcai/process/wild_challenge/res50_wild_3d_mix/output_results/res50_wild_3d_balanced_wild/'
        obj_root = args.obj_root
        file_list = os.listdir(args.folder)
        file_list = tqdm(file_list)
        for i in file_list:
            uv_path = os.path.join(args.folder, i)
            name = i[:-4]
            file_list.set_description("Processing {}".format(name))

            # prepare
            if not os.path.isdir(args.outdir):
                os.mkdir(args.outdir)
            outdir = args.outdir
            obj_path = os.path.join(outdir, name + '.obj')
            in_obj_path = os.path.join(obj_root, name, 'body.obj')
            if args.is_quad:
                tranf_quad2tri(in_obj_path, obj_path, uvs_r, faces_r, face_uvs_r, norms_r, face_norms_r)
            else:
                shutil.copy('{}{}_pred.obj'.format(obj_root, name), obj_path)

            # dilate uv
            if not os.path.isdir('tmp'):
                os.mkdir('tmp')
            dilate_save_img(True, args.size, uv_path, 'tmp/tmp.bmp')

            # reflect texture on mesh
            reflect_result(uvPath='tmp/tmp.bmp', objPath=obj_path, mesh_path = obj_path)

            # interpolation
            interpolation4all(obj_path, obj_path)




