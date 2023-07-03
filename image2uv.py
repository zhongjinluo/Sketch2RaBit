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



class Image2UV:
    def __init__(self):
        pass
    
    def process_uv_inverse(self, uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] / (uv_w - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        uv_coords[:, 1] = uv_coords[:, 1] / (uv_h - 1)
        return uv_coords[:, 0:2]

    def get_old_uv(self):
        verts, faces, norms, face_norms, uvs, face_uvs = load_obj_mesh("transfer_uv/xiaojin_test/source.obj", with_texture=True, with_normal=True)
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

    def transfer_single_uv(self, v_color, side, new_path=None, new_obj_path=None):
        '''
        # new uv parsing
        '''
        right_obj = 'transfer_uv/xiaojin_test2/target_template.obj'
        verts_r, faces_r, norms_r, face_norms_r, uvs_r, face_uvs_r = load_obj_mesh(right_obj, with_texture=True, with_normal=True)
        new_uv_color = np.zeros((int(face_uvs_r.max())+1, 3))
        for index, face in enumerate(faces_r):
            face_uv = face_uvs_r[index]
            for i in range(3):
                new_uv_color[face_uv[i]] = v_color[face[i]]
        uv_coords = self.process_uv(uvs_r, side, side)
        uv_coords_image = mesh.render.render_colors(uv_coords, face_uvs_r, new_uv_color, side, side, c=3)
        uv_coords_image = (uv_coords_image * 255).astype(np.uint8)
        io.imsave(new_path, np.squeeze(uv_coords_image))

        # resize and save the result img
        result_img = cv2.imread(new_path, 1)
        kernel = np.ones((5,5),np.uint8) 
        result_img = cv2.dilate(result_img, kernel, iterations=1)
        # result_img = cv2.resize(result_img, (2048, 2048), interpolation=cv2.INTER_CUBIC)
        if new_path:
            cv2.imwrite(new_path, result_img)
        
        return result_img
        
        # verts_r, faces_r, norms_r, face_norms_r, uvs_r, face_uvs_r = load_obj_mesh(right_obj, with_texture=True, with_normal=True)

        # save_obj_mesh_with_uv_tri(new_obj_path, verts_r, faces_r, uvs_r, face_uvs_r, norms_r, face_norms_r)
    


if __name__ == '__main__':
    def get_old_uv():
        verts, faces, norms, face_norms, uvs, face_uvs = load_obj_mesh("transfer_uv/xiaojin_test/source.obj", with_texture=True, with_normal=True)
        v_color = np.zeros((int(faces.max())+1, 3))
        indices = np.where(norms[:, 2] > 0)
        v_color[indices] = np.array([0.9, 0.1, 0.1])
        return v_color, 512
    
    i2uv = Image2UV()
    # output
    new_path = 'm.png'
    # new_obj_path = 'm.obj' 
    v_color, side = get_old_uv()
    result_img = i2uv.transfer_single_uv(v_color, side, new_path)
    cv2.imwrite(new_path, result_img)