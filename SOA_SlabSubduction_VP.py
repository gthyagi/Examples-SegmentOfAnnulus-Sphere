# ### Slab Subduction in Segment of Annulus

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy
import gmsh
import os
import time

# not necessary once mesh generation is defined in meshing module
import math
from typing import Tuple
from enum import Enum
# -

from underworld3.cython import petsc_discretisation
from underworld3 import timing

# plotting options
if uw.mpi.size == 1:
    import pyvista as pv
    import vtk
    render = True
else:
    render = False

# setting specific parameter values
os.environ["UW_TIMING_ENABLE"] = "1"

# +
# some useful global variable
test_no = 1

# number of steps
nsteps = 3
# -

# creating output directory
meshes_dir = os.path.join(os.path.abspath("./"), "meshes/")
output_dir = os.path.join(os.path.abspath("./output/"), f"SOA_SlabSubduction_Test{test_no}/")
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)


def SegmentOfAnnulusEdit(radiusOuter: float = 1.0,
                         radiusInner: float = 0.3,
                         angleExtent: float = 45,
                         cellSize: float = 0.1,
                         centre: bool = False,
                         degree: int = 1,
                         qdegree: int = 2,
                         filename=None,
                         verbosity=0,
                        ):
    """
    Creates Segment of Annulus
    """
    
    class boundaries(Enum):
        InnerArc = 1
        OuterArc = 2
        Left = 3
        Right = 4
        Centre = 10

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbosity)
    gmsh.model.add("SegmentOfAnnulus")

    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, meshSize=cellSize)

    # angle Extent in radian
    angleExtentRadian = np.deg2rad(angleExtent)
    theta1 = (np.pi - angleExtentRadian) / 2
    theta2 = theta1 + angleExtentRadian

    loops = []

    if radiusInner > 0.0:
        p1 = gmsh.model.geo.addPoint(radiusInner * np.cos(theta1), radiusInner * np.sin(theta1), 0.0, meshSize=cellSize)
        p4 = gmsh.model.geo.addPoint(radiusInner * np.cos(theta2), radiusInner * np.sin(theta2), 0.0, meshSize=cellSize)

    p2 = gmsh.model.geo.addPoint(radiusOuter * np.cos(theta1), radiusOuter * np.sin(theta1), 0.0, meshSize=cellSize)
    p3 = gmsh.model.geo.addPoint(radiusOuter * np.cos(theta2), radiusOuter * np.sin(theta2), 0.0, meshSize=cellSize)


    if radiusInner > 0.0:
        l_right = gmsh.model.geo.addLine(p1, p2)
        l_left = gmsh.model.geo.addLine(p3, p4)
        c_outer = gmsh.model.geo.addCircleArc(p2, p0, p3)
        c_inner = gmsh.model.geo.addCircleArc(p4, p0, p1)
        loops = [c_inner, l_right, c_outer, l_left]
    else:
        l_right = gmsh.model.geo.addLine(p0, p2)
        l_left = gmsh.model.geo.addLine(p3, p0)
        c_outer = gmsh.model.geo.addCircleArc(p2, p0, p3)
        loops = [l_right, c_outer, l_left]

    loop = gmsh.model.geo.addCurveLoop(loops)
    s = gmsh.model.geo.addPlaneSurface([loop])

    # gmsh.model.mesh.embed(0, [p0], 2, s) # not sure use of this line

    if radiusInner > 0.0:
        gmsh.model.addPhysicalGroup(1, [c_inner], boundaries.InnerArc.value, name=boundaries.InnerArc.name)
    else:
        gmsh.model.addPhysicalGroup(0, [p0], tag=boundaries.Centre.value, name=boundaries.Centre.name)

    gmsh.model.addPhysicalGroup(1, [c_outer], boundaries.OuterArc.value, name=boundaries.OuterArc.name)
    gmsh.model.addPhysicalGroup(1, [l_left], boundaries.Left.value, name=boundaries.Left.name)
    gmsh.model.addPhysicalGroup(1, [l_right], boundaries.Right.value, name=boundaries.Right.name)
    gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

    return 

# #### Create Mesh

# creating mesh in gmsh
if uw.mpi.rank == 0:
    SegmentOfAnnulusEdit(radiusOuter=6371.0/6371.0, radiusInner=1-(2891/6371), 
                         angleExtent=90, cellSize=0.07,
                         filename=meshes_dir+f"soa_mesh_test{test_no}.msh")

# start timing
timing.reset()
timing.start()

# creating uw mesh
mesh = uw.discretisation.Mesh(meshes_dir+f"soa_mesh_test{test_no}.msh", 
                              simplex=True, qdegree=3, markVertices=False, 
                              useRegions=True, useMultipleTags=True, )

# stop timing
timing.stop()
uw.timing.print_table()

if uw.mpi.size == 1:
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [900, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "trame"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh.vtk(output_dir+f"soa_mesh_test{test_no}.vtk")
    pvmesh = pv.read(output_dir+f"soa_mesh_test{test_no}.vtk")

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Blue", "wireframe", opacity=0.5,)

    pl.show(cpos="xy")

# creating swarm
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.SwarmVariable("M", swarm, size=1, proxy_continuous=False, proxy_degree=0)
swarm.populate(fill_param=1)

if uw.mpi.size == 1:

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "trame"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    pvmesh = pv.read(output_dir+f"soa_mesh_test{test_no}.vtk")

    pl = pv.Plotter()

    with swarm.access():
        points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm.particle_coordinates.data[:, 0]
        points[:, 1] = swarm.particle_coordinates.data[:, 1]

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl.add_points(point_cloud, cmap="viridis", render_points_as_spheres=False,
                  point_size=5, opacity=0.66,)
    pl.add_mesh(pvmesh, "Blue", "wireframe")

    pl.show(cpos="xy")



mesh1 = uw.meshing.SegmentOfAnnulus(radiusOuter = 1.0, radiusInner = 0.5, 
                                    angleExtent = 90, cellSize = 0.1,)

if uw.mpi.size == 1:
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [900, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "trame"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh1.vtk(output_dir+f"soa_mesh1_test{test_no}.vtk")
    pvmesh = pv.read(output_dir+f"soa_mesh1_test{test_no}.vtk")

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Blue", "wireframe", opacity=0.5,)

    pl.show(cpos="xy")

mesh1.CoordinateSystem.X


