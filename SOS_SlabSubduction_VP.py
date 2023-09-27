# ### Slab Subduction in Segment of Sphere

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

if uw.mpi.size == 1:
    import pyvista as pv
    import vtk

# setting specific parameter values
os.environ["UW_TIMING_ENABLE"] = "1"

# some useful global variable
test_no = 1

# creating output directory
meshes_dir = os.path.join(os.path.abspath("./"), "meshes/")
output_dir = os.path.join(os.path.abspath("./output/"), f"SOS_SlabSubduction_Test{test_no}/")
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)


def SegmentOfSphereEdit(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.5,
    longitudeExtent: float = 90.0,
    latitudeExtent: float = 90.0,
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    regular: bool = False,
    filename=None,
    refinement=None,
    verbosity=0,
    centroid: Tuple = (0.0, 0.0, 0.0),
):
    """
    Generates a segment of sphere.

    Parameters
    ----------
    radiusOuter:
        Float specifying radius of the outer surface.
    radiusInner:
        Float specifying radius of the inner surface.
    LongitudeExtent:
        Angle (float) specifying model extent in longitude direction.
    LatitudeExtent:
        Angle (float) specifying model extent in latitude direction. 

    """

    class boundaries(Enum):
        Inner = 11
        Outer = 12
        East = 13
        West = 14
        South = 15
        North = 16
        
    def getSphericalXYZ(point):
        """
        Perform Cubed-sphere projection on coordinates.
        Converts (radius, lon, lat) in spherical region to (x, y, z) in spherical region.

        Parameters
        ----------
        Input: 
            Coordinates in rthetaphi format (radius, lon, lat) 
        Output
            Coordinates in XYZ format (x, y, z)
        """
        
        (x,y) = (math.tan(point[1]*math.pi/180.0), math.tan(point[2]*math.pi/180.0))
        d = point[0] / math.sqrt( x**2 + y**2 + 1)
        coordX, coordY, coordZ = centroid[0] + d*x, centroid[1] + d*y, centroid[2] + d
        
        return (coordX, coordY, coordZ)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbosity)
    gmsh.model.add("SegmentOfSphere")

    p0 = gmsh.model.geo.addPoint(centroid[0], centroid[1], centroid[2], meshSize=cellSize)
    
    # Create segment of sphere
    dim = 3

    long_half = longitudeExtent/2
    lat_half = latitudeExtent/2

    pt1 = getSphericalXYZ((radiusInner, -long_half, -lat_half))
    pt2 = getSphericalXYZ((radiusInner, long_half, -lat_half))
    pt3 = getSphericalXYZ((radiusInner, long_half, lat_half))
    pt4 = getSphericalXYZ((radiusInner, -long_half, lat_half))
    pt5 = getSphericalXYZ((radiusOuter, -long_half, -lat_half))
    pt6 = getSphericalXYZ((radiusOuter, long_half, -lat_half))
    pt7 = getSphericalXYZ((radiusOuter, long_half, lat_half))
    pt8 = getSphericalXYZ((radiusOuter, -long_half, lat_half))
    

    p1 = gmsh.model.geo.addPoint(pt1[0], pt1[1], pt1[2], meshSize=cellSize)
    p2 = gmsh.model.geo.addPoint(pt2[0], pt2[1], pt2[2], meshSize=cellSize)
    p3 = gmsh.model.geo.addPoint(pt3[0], pt3[1], pt3[2], meshSize=cellSize)
    p4 = gmsh.model.geo.addPoint(pt4[0], pt4[1], pt4[2], meshSize=cellSize)
    p5 = gmsh.model.geo.addPoint(pt5[0], pt5[1], pt5[2], meshSize=cellSize)
    p6 = gmsh.model.geo.addPoint(pt6[0], pt6[1], pt6[2], meshSize=cellSize)
    p7 = gmsh.model.geo.addPoint(pt7[0], pt7[1], pt7[2], meshSize=cellSize)
    p8 = gmsh.model.geo.addPoint(pt8[0], pt8[1], pt8[2], meshSize=cellSize)

    l1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
    l2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
    l3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
    l4 = gmsh.model.geo.addCircleArc(p4, p0, p1)
    l5 = gmsh.model.geo.addCircleArc(p5, p0, p6)
    l6 = gmsh.model.geo.addCircleArc(p6, p0, p7)
    l7 = gmsh.model.geo.addCircleArc(p7, p0, p8)
    l8 = gmsh.model.geo.addCircleArc(p8, p0, p5)
    l9 = gmsh.model.geo.addLine(p5, p1)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p7, p3)
    l12 = gmsh.model.geo.addLine(p4, p8)

    cl = gmsh.model.geo.addCurveLoop((l1, l2, l3, l4))
    inner = gmsh.model.geo.addSurfaceFilling([cl], tag=boundaries.Inner.value, sphereCenterTag=p0)

    cl = gmsh.model.geo.addCurveLoop((l5, l6, l7, l8))
    outer = gmsh.model.geo.addSurfaceFilling([cl], tag=boundaries.Outer.value, sphereCenterTag=p0)

    cl = gmsh.model.geo.addCurveLoop((l10, l6, l11, -l2)) 
    east = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.East.value)

    cl = gmsh.model.geo.addCurveLoop((l9, -l4, l12, l8))
    west = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.West.value)

    cl = gmsh.model.geo.addCurveLoop((l1, l10, -l5, l9))
    south = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.South.value)

    cl = gmsh.model.geo.addCurveLoop((-l3, -l11, l7, -l12))
    north = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.North.value)

    sloop = gmsh.model.geo.addSurfaceLoop(
        [south, east, north, outer, west, inner]
    )
    volume = gmsh.model.geo.addVolume([sloop])

    gmsh.model.geo.synchronize()

    # Add Physical groups
    for b in boundaries:
        tag = b.value
        name = b.name
        gmsh.model.addPhysicalGroup(2, [tag], tag)
        gmsh.model.setPhysicalName(2, tag, name)

    gmsh.model.addPhysicalGroup(3, [volume], 99999)
    gmsh.model.setPhysicalName(3, 99999, "Elements")

    # Generate Mesh
    gmsh.model.mesh.generate(dim)
    gmsh.write(filename)
    gmsh.finalize()

    return

# creating mesh in gmsh
if uw.mpi.rank == 0:
    SegmentOfSphereEdit(radiusOuter=6371.0/6371.0, radiusInner=1-(2891/6371), 
                        longitudeExtent=90, latitudeExtent=90, cellSize=0.07,
                        filename=meshes_dir+f"sos_mesh_test{test_no}.msh")

# start timing
timing.reset()
timing.start()

# creating uw mesh
mesh = uw.discretisation.Mesh(meshes_dir+f"sos_mesh_test{test_no}.msh", 
                              simplex=True, qdegree=3, markVertices=False, useRegions=True, useMultipleTags=True, )

# stop timing
timing.stop()
uw.timing.print_table()

if uw.mpi.size == 1:
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "trame"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh.vtk(output_dir+f"sos_mesh_test{test_no}.vtk")
    pvmesh = pv.read(output_dir+f"sos_mesh_test{test_no}.vtk")

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Blue", "wireframe", opacity=0.5,)

    pl.show(cpos="yx")

# creating swarm
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.SwarmVariable("M", swarm, size=1, proxy_continuous=False, proxy_degree=0)
swarm.populate(fill_param=1)

# +
# printing mesh and swarm info
print(mesh.data.shape)

with swarm.access(material):
    print(material.data.shape, '\n', material.swarm.data)
# -

if uw.mpi.size == 1:

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "trame"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    pvmesh = pv.read(output_dir+f"sos_mesh_test{test_no}.vtk")

    pl = pv.Plotter()

    with swarm.access():
        points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm.particle_coordinates.data[:, 0]
        points[:, 1] = swarm.particle_coordinates.data[:, 1]
        points[:, 2] = swarm.particle_coordinates.data[:, 2]

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl.add_points(point_cloud, cmap="viridis", render_points_as_spheres=False,
                  point_size=5, opacity=0.66,)
    pl.add_mesh(pvmesh, "Blue", "wireframe")

    pl.show(cpos="xy")

# +
# # saving data
# swarm.save(output_dir+f"sos_swarm_test{test_no}.h5")
# material.save(output_dir+f"sos_mat_test{test_no}.h5")
# -


