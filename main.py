
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

import DracoPy
from networks.v0.sketch2norm import Sketch2Norm
from networks.v0.sketch2model import Sketch2Model

from networks.v0.sdfrenderer import MyRenderer
from networks.v0.sketch2keypoint import Sketch2Keypoint


s2n = Sketch2Norm()
s2m = Sketch2Model()
s2k = Sketch2Keypoint()
r = MyRenderer(s2m)

def generate_mesh():

    root = "sketch_database"
    for f in os.listdir(root):
        shutil.rmtree("networks/v0/renders")
        os.makedirs("networks/v0/renders")

        sketch = np.array(Image.open(os.path.join(root, f)).convert('RGB'))
        s, n = s2n.predict(sketch)

        r.render(s2m, n)
        count = 0
        while count < 6:
            count = 0
            for _ in os.listdir("networks/v0/renders"):
                count += 1
        landmarks = s2k.predict("", "")

        obj_path = os.path.join("outs", f.replace(".png", ".obj"))
        vertices, faces = s2m.predict_with_template_2(n, obj_path, landmarks)
        print(f)

    return True


if __name__ == "__main__":
    generate_mesh()