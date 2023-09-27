import gmsh
import sys
import numpy as np
import math
from typing import Tuple
from enum import Enum


def SegmentOfAnnulus(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.3,
    angleExtent: float = 45,
    cellSize: float = 0.1,
    centre: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    verbosity=0,
):
    class boundaries(Enum):
        InnerArc = 1
        OuterArc = 2
        Left = 3
        Right = 4
        Centre = 10

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = (
            f"uw_SegmentOfAnnulus_ro{radiusOuter}_ri{radiusInner}_extent{angleExtent}_csize{cellSize}.msh"
        )
    else:
        uw_filename = filename

    if uw.mpi.rank == 0:
        import gmsh

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
        gmsh.write(uw_filename)
        gmsh.finalize()

    new_mesh = Mesh(
        uw_filename,
        degree=degree,
        qdegree=qdegree,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
    )

    return new_mesh


def SegmentOfAnnulusEdit(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.3,
    angleExtent: float = 45,
    cellSize: float = 0.1,
    centre: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    verbosity=0,
):
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

SegmentOfAnnulusEdit(radiusOuter=1.0, radiusInner=1-(2891/6371), angleExtent=90, filename='test_segofann.vtk', )

# +
# # initialise
# gmsh.initialize()

# # model name
# gmsh.model.add("SegmentOfAnnulus")

# # parameters
# radiusOuter = 1.0
# radiusInner = 0.55
# angleExtend = 180 # Max. accepted value is 180
# cellSize = 0.08

# # Origin point
# Origin = gmsh.model.geo.addPoint(0, 0, 0, cellSize, 1)

# # angle Extent in radian
# angleExtendRadian = np.deg2rad(angleExtend)
# theta = (np.pi - angleExtendRadian) / 2

# # Inner arc end points
# Inner_EP_right = gmsh.model.geo.addPoint(radiusInner * np.cos(theta), radiusInner * np.sin(theta), 0.0, 
#                                          cellSize, 2)
# Inner_EP_left = gmsh.model.geo.addPoint(radiusInner * np.cos(theta+angleExtendRadian), 
#                                         radiusInner * np.sin(theta+angleExtendRadian), 0.0, cellSize, 3)

# # Outer arc end points
# Outer_EP_right = gmsh.model.geo.addPoint(radiusOuter * np.cos(theta), radiusOuter * np.sin(theta), 0.0, 
#                                          cellSize, 4)
# Outer_EP_left = gmsh.model.geo.addPoint(radiusOuter * np.cos(theta+angleExtendRadian), 
#                                         radiusOuter * np.sin(theta+angleExtendRadian), 0.0, cellSize, 5)

# # Inner and Outer Arcs
# InnerArc = gmsh.model.geo.addCircleArc(Inner_EP_right, Origin, Inner_EP_left, 1)
# OuterArc = gmsh.model.geo.addCircleArc(Outer_EP_right, Origin, Outer_EP_left, 2)

# # Side lines
# RightLine = gmsh.model.geo.addLine(Inner_EP_right, Outer_EP_right, 3)
# LeftLine = gmsh.model.geo.addLine(Inner_EP_left, Outer_EP_left, 4)

# # Curve loop
# CurveLoop = gmsh.model.geo.addCurveLoop([1, 4, -2, -3], 1)

# # Plane Surface
# PlaneSurface = gmsh.model.geo.addPlaneSurface([1], 1)

# # Physical Curve and Physical Surface
# PhysicalCurve = gmsh.model.geo.addPhysicalGroup(1, [1, 4, -2, -3], 1)
# PhysicalSurface = gmsh.model.geo.addPhysicalGroup(2, [1], 1)

# # synchronize
# gmsh.model.geo.synchronize()

# # We can then generate a 2D mesh...
# gmsh.model.mesh.generate(2)

# # save as vtk
# gmsh.write("SegmentOfAnnulus.vtk")

# # This should be called when you are done using the Gmsh Python API:
# gmsh.finalize()
# -


