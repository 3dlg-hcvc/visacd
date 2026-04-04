# VisACD: Visibility-Based GPU-Accelerated Approximate Convex Decomposition

[Project Page](https://3dlg-hcvc.github.io/visacd) | Paper (coming soon)

![Teaser](docs/visuals/teaser.png)

VisACD is a visibility-based, GPU-accelerated algorithm for intersection-free approximate convex decomposition (ACD). It is rotation-equivariant, making it robust to variations in input mesh orientation. Compared to prior work, VisACD produces decompositions that more closely approximate the original geometry while using fewer parts, and does so with significantly improved efficiency.

At the core of the approach is a **visibility edge** concavity metric: edges between pairs of vertices that lie outside the mesh without intersecting it. A convex mesh contains no such edges, while increasingly concave geometry produces more. The best cutting plane is the one that intersects the largest total length of visibility edges — simple, efficient, and interpretable. The algorithm is fully parallelized using NVIDIA OptiX and CUDA.

## Installation

**Requirements:** CUDA-capable GPU, NVIDIA OptiX 8.0, Python ≥ 3.8. Tested with CUDA 12.2.

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/3dlg-hcvc/visacd
cd visacd
```

### 2. Install NVIDIA OptiX

Download [OptiX 8.0](https://developer.nvidia.com/designworks/optix/downloads/legacy) and install it, then set (OptiX_INSTALL_DIR should contain include/ subfolder):

```bash
export OptiX_INSTALL_DIR=/path/to/optix
```

### 3. Set CUDA compiler (if not on PATH)

```bash
export CUDACXX=/path/to/cuda/bin/nvcc
```

### 4. Install (might take a while)

```bash
pip install .
```

## Usage

```python
import numpy as np
import trimesh
import visacd

# load any mesh trimesh supports
tm = trimesh.load("mesh.obj", force="mesh")

mesh = visacd.Mesh()
mesh.vertices = visacd.VecArray3d(tm.vertices.tolist())
mesh.triangles = visacd.make_vecarray3i(np.array(tm.faces, dtype=np.int32))

# optionally configure
visacd.config.score_mode = "concavity"
visacd.set_seed(42)

# decompose
result = visacd.process(mesh, concavity=0.04, num_parts=32)
print(f"Parts: {result.num_parts}, concavity: {result.concavity:.4f}")

# export as GLB
scene = trimesh.Scene()
for part in result.parts:
    scene.add_geometry(trimesh.Trimesh(
        vertices=np.array(list(part.vertices)),
        faces=np.array(list(part.triangles)),
        process=False,
    ))
scene.export("decomposition.glb")
```

See [decompose.py](decompose.py) for a full example that loads an OBJ and exports the decomposition as a GLB.

## Citation

```bibtex
@inproceedings{fokin2026visacd,
  title={VisACD: Visibility-Based GPU-Accelerated Approximate Convex Decomposition},
  author={Fokin, Egor and Savva, Manolis},
  booktitle={47th Annual Conference of the European Association for Computer Graphics,
                  Eurographics 2026 - Short Papers},
  year={2026}
}
```
