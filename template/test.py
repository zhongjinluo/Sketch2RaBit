import trimesh

uvs = []
uv_faces = []
with open("m.obj") as file_in_template:
    # 
    for line in file_in_template.readlines():
        line = line.replace('\n', '')
        if line.startswith('#'): 
            continue
        values = line.split()
        if len(values) == 0:
            continue
        elif values[0] == "usemtl":
            # file_out.write(line + "\n")
            # usemtl = line + "\n"
            pass
        elif values[0] == "mtllib":
            # file_out.write(line + "\n")
            # mtllib = line + "\n"
            pass
        elif values[0] == "v":
            pass
        elif values[0] == "vt":
            # file_out.write(line + "\n")
            items = line.replace("\n", "").split(" ")
            uvs.append([float(items[1]),float(items[2])])
            # vertex_texture_lines.append(line + "\n")
        elif values[0] == "vn":
            # file_out.write(line + "\n")
            # vertex_normal_lines.append(line + "\n")
            pass
        elif values[0] == "f" or values[0] == "g":
            items = line.replace("\n", "").split(" ")
            uv_face = []
            for item in items:
                if item == "f":
                    continue
                elif 'g' in item or "G" in item:
                    continue
                else:
                    splits = item.split("/")
                    uv_face.append(int(splits[1]))
            uv_faces.append(uv_face)
print(uv_faces)