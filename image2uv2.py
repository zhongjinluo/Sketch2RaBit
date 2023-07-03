import sys
sys.path.append("transfer_uv/")
import os
import numpy as np
import cv2
from skimage import io
from mesh import load_obj_mesh, save_obj_mesh_with_uv_tri, save_obj_mesh_with_uv_qurd
import argparse
from face3d import mesh
from tqdm import tqdm



class Image2UV2:
    def __init__(self):
        pass
    
    def process_uv_inverse(self, uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] / (uv_w - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        uv_coords[:, 1] = uv_coords[:, 1] / (uv_h - 1)
        return uv_coords[:, 0:2]

    def get_old_uv(self):
        verts, faces, norms, face_norms, uvs, face_uvs = load_obj_mesh("xiaojin_test/source.obj", with_texture=True, with_normal=True)
        v_color = np.zeros((int(faces.max())+1, 3))
        indices = np.where(norms[:, 2] > 0)
        v_color[indices] = np.array([0.9, 0.1, 0.1])
        return v_color, 512

    def process_uv(self, uv_coords, uv_h=256, uv_w=256, add_z=True):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        if add_z:
            uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
        return uv_coords

    def transfer_single_uv(self, in_uv_coords, in_uv_faces, in_texture, side, new_path=None, new_obj_path=None):
        verts_r, faces_r, uvs_r, face_uvs_r = load_obj_mesh("transfer_uv/xiaojin_test2/target_template.obj", with_texture=True)
        out_uv_coords = self.process_uv(uvs_r, side, side)
        uv_coords_image = mesh.render.render_texture(out_uv_coords, face_uvs_r, in_texture, in_uv_coords, in_uv_faces, side, side, c = 3, mapping_type = 'bilinear')
        uv_coords_image = (uv_coords_image*255).astype(np.uint8)
        io.imsave(new_path, np.squeeze(uv_coords_image))
        result_img = cv2.imread(new_path, 1)
        return result_img
        
        # verts_r, faces_r, norms_r, face_norms_r, uvs_r, face_uvs_r = load_obj_mesh(right_obj, with_texture=True, with_normal=True)

        # save_obj_mesh_with_uv_tri(new_obj_path, verts_r, faces_r, uvs_r, face_uvs_r, norms_r, face_norms_r)
    


if __name__ == '__main__':
    i2uv = Image2UV2()
    # output
    old_path = 'xiaojin_test2/source.bmp'
    obj_path = 'xiaojin_test2/source.obj'
    new_path = 'new_path.png'

    def process_uv(uv_coords, uv_h=256, uv_w=256, add_z=True):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        if add_z:
            uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
        return uv_coords

    in_texture = io.imread(old_path) / 255.
    verts, faces, uvs, in_uv_faces = load_obj_mesh(obj_path, with_texture=True)
    in_uv_coords = process_uv(uvs, 512, 512)
    result_img = i2uv.transfer_single_uv(in_uv_coords=in_uv_coords, in_uv_faces=in_uv_faces, in_texture=in_texture, side=512, new_path=new_path)
    cv2.imwrite(new_path, result_img)