import sys
import os
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib/build"))
import lib_neural_acd as lib

os.makedirs("out", exist_ok=True)


def load_obj(path):
    vertices = []
    triangles = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # faces may be v, v/vt, or v/vt/vn — take only the vertex index
                indices = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                triangles.append(indices)
    return vertices, triangles


def main():
    vertices, triangles = load_obj("data/cow.obj")

    mesh = lib.Mesh()
    mesh.vertices = lib.VecArray3d(vertices)
    mesh.triangles = lib.make_vecarray3i(np.array(triangles, dtype=np.int32))

    parts = lib.process(mesh, concavity=0.03, num_parts=32, stats_file="out/stats.csv")

    print(f"Decomposed into {len(parts)} parts.")

    scene = trimesh.Scene()
    for i, part in enumerate(parts):
        verts = np.array(list(part.vertices), dtype=np.float64)
        faces = np.array(list(part.triangles), dtype=np.int32)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        scene.add_geometry(tm, node_name=f"part_{i:03d}")

    out_path = "out/decomposition.glb"
    scene.export(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
