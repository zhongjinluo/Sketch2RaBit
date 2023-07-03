import os
import numpy as np
import cv2
from skimage import io
from mesh import load_obj_mesh, save_obj_mesh, save_obj_mesh_with_rgb
from mesh_util import save_obj_mesh_with_color
from face3d import mesh
import argparse

def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs, face_uvs):
    file = open(mesh_path, 'w')

    file.write('mtllib m.mtl\n')
    file.write('usemtl defaultMat\n')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    
    for vt in uvs:
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    # for f in faces:
    #     f_plus = f + 1
    #     file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))

    for idx, f in enumerate(faces):
        f_plus = f + 1
        f_plus_uv = face_uvs[idx] + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus_uv[0],
                                              f_plus[1], f_plus_uv[1],
                                              f_plus[2], f_plus_uv[2]))
    file.close()

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


def process_uv(uv_coords, uv_h=256, uv_w=256, add_z=True):
    # uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    # uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    if add_z:
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
    return uv_coords

def get_uv(objPath, uvPath):

    verts, faces, uvs, face_uvs = load_obj_mesh_color(objPath, with_texture=True)

    side = 2048
    color_uv = verts[:, 3:]
    uv_coords = process_uv(uvs, side, side)
    uv_coords_image = mesh.render.render_colors(uv_coords, face_uvs, color_uv, side, side, c=3)
    uv_coords_image = (uv_coords_image*255).astype(np.uint8)
    io.imsave(uvPath, np.squeeze(uv_coords_image))

    save_obj_mesh_with_uv(objPath, verts[:, :3], faces, uvs, face_uvs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--obj', type=str, default='m.obj')
    parser.add_argument('-u', '--uv', type=str, default='tmp/t.bmp')
    args = parser.parse_args()

    get_uv(args.obj, args.uv)