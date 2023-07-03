# -*- coding: utf-8 -*-
import os
import shutil
from mesh import load_obj_mesh
import argparse

def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs, face_uvs, norms, face_norms):
    file = open(mesh_path, 'w')

    file.write('mtllib m.mtl\n')
    file.write('usemtl defaultMat\n')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    
    for vt in uvs:
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for vn in norms:
        file.write('vn %.4f %.4f %.4f\n' % (vn[0], vn[1], vn[2]))
    # for f in faces:
    #     f_plus = f + 1
    #     file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))

    for idx, f in enumerate(faces):
        f_plus = f + 1
        f_plus_uv = face_uvs[idx] + 1
        f_plus_n = face_norms[idx] + 1
        file.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (f_plus[0], f_plus_uv[0], f_plus_n[0],
                                              f_plus[1], f_plus_uv[1], f_plus_n[1],
                                              f_plus[2], f_plus_uv[2], f_plus_n[2]))
    file.close()

def tranf_quad2tri(obj_path, out_obj_path, uvs_r, faces_r, face_uvs_r, norms_r, face_norms_r):
    """
        Transfer quad to tri mesh
    """
    # print("Transfering {}".format(obj_path))
    verts, faces = load_obj_mesh(obj_path)
    save_obj_mesh_with_uv(out_obj_path, verts, faces_r, uvs_r, face_uvs_r, norms_r, face_norms_r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--date', type=str, default='20210906')
    args = parser.parse_args()
        
    current_dir = './'

    right_obj_path = os.path.join(current_dir, 'right.obj')
    verts_r, faces_r, norms_r, face_norms_r, uvs_r, face_uvs_r = load_obj_mesh(right_obj_path, with_texture=True, with_normal=True)

    group_name = '{}_design'.format(args.date)
    dir_path = os.path.join(current_dir, group_name)

    output_dir = os.path.join(current_dir, 'raw_samples_{}'.format(args.date))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    num = 0
    file_list = os.listdir(dir_path)
    for animal in file_list:
        new_dir = os.path.join(dir_path, animal)
        if os.path.isdir(new_dir):
            m_list = os.listdir(new_dir)
            for id in m_list:
                m_dir = os.path.join(new_dir, id)
                if os.path.isdir(m_dir):
                    name = group_name + '_' + animal + '_' + id
                    num += 1
                    print("Processing {}".format(name))
                    obj_path = os.path.join(m_dir, 'm.OBJ')

                    out_dir_path = os.path.join(output_dir, name)
                    if not os.path.isdir(out_dir_path):
                        os.mkdir(out_dir_path)
                    check_dir = os.path.join(out_dir_path, 'check')
                    print(check_dir)
                    if not os.path.isdir(check_dir):
                        os.mkdir(check_dir)
                    out_obj_path = os.path.join(out_dir_path, 'm.obj')
                    shutil.copy(m_dir+'/m.BMP', out_dir_path+'/m.bmp')
                    shutil.copy(m_dir+'/m.mtl', out_dir_path+'/m.mtl')
                    shutil.copy(m_dir+'/check/m.png', check_dir+'/m.png')
                    tranf_quad2tri(obj_path, out_obj_path, uvs_r, faces_r, face_uvs_r, norms_r, face_norms_r)
    print("Total number is {}".format(num))







