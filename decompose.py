import argparse
import os
import numpy as np
import trimesh

import visacd


def main():
    parser = argparse.ArgumentParser(description="Approximate convex decomposition via VisACD.")
    parser.add_argument("mesh", help="Path to input mesh file")
    parser.add_argument("-o", "--output", help="Output GLB path (default: <mesh_name>_decomposed.glb)")
    parser.add_argument("--concavity", type=float, default=0.04)
    parser.add_argument("--num-parts", type=int, default=40)
    args = parser.parse_args()

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.mesh))[0]
        out_path = os.path.join(os.path.dirname(args.mesh) or ".", f"{base}_decomposed.glb")

    tm = trimesh.load(args.mesh, force="mesh")

    mesh = visacd.Mesh()
    mesh.vertices = visacd.VecArray3d(tm.vertices.tolist())
    mesh.triangles = visacd.make_vecarray3i(np.array(tm.faces, dtype=np.int32))

    result = visacd.process(mesh, concavity=args.concavity, num_parts=args.num_parts)
    print(f"Decomposed into {result.num_parts} parts (concavity={result.concavity:.4f}).")

    scene = trimesh.Scene()
    for i, part in enumerate(result.parts):
        verts = np.array(list(part.vertices), dtype=np.float64)
        faces = np.array(list(part.triangles), dtype=np.int32)
        scene.add_geometry(trimesh.Trimesh(vertices=verts, faces=faces, process=False), node_name=f"part_{i:03d}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    scene.export(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
