import sys
import os
import numpy as np
import trimesh

import visacd


os.makedirs("out", exist_ok=True)


def main():
    tm = trimesh.load("data/cow.obj", force="mesh")

    mesh = visacd.Mesh()
    mesh.vertices = visacd.VecArray3d(tm.vertices.tolist())
    mesh.triangles = visacd.make_vecarray3i(np.array(tm.faces, dtype=np.int32))

    result = visacd.process(mesh, concavity=0.04, num_parts=40)

    print(f"Decomposed into {result.num_parts} parts (concavity={result.concavity:.4f}).")

    scene = trimesh.Scene()
    for i, part in enumerate(result.parts):
        verts = np.array(list(part.vertices), dtype=np.float64)
        faces = np.array(list(part.triangles), dtype=np.int32)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        scene.add_geometry(tm, node_name=f"part_{i:03d}")

    out_path = "out/decomposition.glb"
    scene.export(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
