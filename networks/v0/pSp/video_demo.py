import os
from show_on_mesh import show_on_mesh
from tqdm import tqdm
from multiprocessing import Process

def parse_name(source):
    name = source.split('_shape=')[0]
    return name[name.find('_') + 1 : ]

def video_demo(input_path, output_path, uv_root='/home/jinguodong/data/inverseGAN/1007'):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    menu = os.listdir(input_path)

    for i in tqdm(menu):
        k = i.split('=')[-1]
        if k == '0':
            continue
        obj_path = os.path.join(input_path, i, 'body.obj')
        name = parse_name(i)
        uv_path = os.path.join(uv_root, name, '{}.png'.format(k))
        out_obj = os.path.join(output_path, i + '.obj')

        show_on_mesh(
            uv_path, 
            input_obj=obj_path, 
            output_obj=out_obj, 
            is_quad=True, 
            size=1024
        )

if __name__ == '__main__':
    root = '/mnt/sharedisk/ruiboming/reconstruction/body_add_eye/demo/2_svr/1_raw_mesh/result'

    processes = [Process(target=video_demo, args=(os.path.join(root, str(x)), os.path.join('/home/jinguodong/data/inverseGAN/1007/obj', str(x)))) for x in range(30)]

    # 并行多进程处理
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('\nDone!')

    # for i in range(5):
    #     j = str(i)
    #     video_demo(, )