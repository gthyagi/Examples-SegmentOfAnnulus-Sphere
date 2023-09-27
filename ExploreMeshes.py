# ### Explore UW3 mesh options

import underworld3 as uw
import os

# +
# output directory 
output_dir = './output/'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# +
# Annnulus mesh
annulus_mesh = uw.meshing.Annulus(radiusOuter = 1.0,
                                  radiusInner = 0.3,
                                  cellSize = 0.05,
                                  degree = 1,
                                  qdegree = 2,)

# save the mesh into a vtk file
annulus_mesh.vtk(output_dir+'AnnulusMesh.vtk')

# +
# Annulus with internal boundary mesh
annulus_internal_boundary_mesh = uw.meshing.AnnulusInternalBoundary(radiusOuter = 1.5,
                                                     radiusInternal = 1.0,
                                                     radiusInner = 0.5,
                                                     cellSize = 0.1,
                                                     cellSize_Outer = 0.05,
                                                     degree = 1,
                                                     qdegree = 2,)

# save the mesh into a vtk file
annulus_internal_boundary_mesh.vtk(output_dir+'AnnulusInternalBoundaryMesh.vtk')

# +
# Cubedsphere mesh
cubedsphere_mesh = uw.meshing.CubedSphere(radiusOuter = 1.0,
                                          radiusInner = 0.3,
                                          numElements = 5, # controls no.of elements in the radial direction
                                          degree = 1,
                                          qdegree = 2,)

# save the mesh into a vtk file
cubedsphere_mesh.vtk(output_dir+'CubedSphereMesh.vtk')

# +
# Quarter Annulus (angle 0) mesh
quarter_annulus_0_mesh = uw.meshing.QuarterAnnulus(radiusOuter = 1.0,
                                                 radiusInner = 0.3,
                                                 angle = 0,
                                                 cellSize = 0.1,
                                                 degree = 1,
                                                 qdegree = 2,)

# save the mesh into a vtk file
quarter_annulus_0_mesh.vtk(output_dir+'QuarterAnnulus0Mesh.vtk')

# Quarter Annulus (angle 20) mesh
quarter_annulus_20_mesh = uw.meshing.QuarterAnnulus(radiusOuter = 1.0,
                                                 radiusInner = 0.3,
                                                 angle = 20,
                                                 cellSize = 0.1,
                                                 degree = 1,
                                                 qdegree = 2,)

# save the mesh into a vtk file
quarter_annulus_20_mesh.vtk(output_dir+'QuarterAnnulus20Mesh.vtk')

# Quarter Annulus (angle 40) mesh
quarter_annulus_40_mesh = uw.meshing.QuarterAnnulus(radiusOuter = 1.0,
                                                 radiusInner = 0.3,
                                                 angle = 40,
                                                 cellSize = 0.1,
                                                 degree = 1,
                                                 qdegree = 2,)

# save the mesh into a vtk file
quarter_annulus_40_mesh.vtk(output_dir+'QuarterAnnulus40Mesh.vtk')

# Quarter Annulus (angle 70) mesh
quarter_annulus_70_mesh = uw.meshing.QuarterAnnulus(radiusOuter = 1.0,
                                                 radiusInner = 0.3,
                                                 angle = 70,
                                                 cellSize = 0.1,
                                                 degree = 1,
                                                 qdegree = 2,)

# save the mesh into a vtk file
quarter_annulus_70_mesh.vtk(output_dir+'QuarterAnnulus70Mesh.vtk')

# Quarter Annulus (angle 90) mesh
quarter_annulus_90_mesh = uw.meshing.QuarterAnnulus(radiusOuter = 1.0,
                                                 radiusInner = 0.3,
                                                 angle = 90,
                                                 cellSize = 0.1,
                                                 degree = 1,
                                                 qdegree = 2,)

# save the mesh into a vtk file
quarter_annulus_90_mesh.vtk(output_dir+'QuarterAnnulus90Mesh.vtk')

# Quarter annulus
quarter_annulus_mesh = uw.meshing.QuarterAnnulus()

# save the mesh into a vtk file
quarter_annulus_mesh.vtk(output_dir+'QuarterAnnulusMesh.vtk')
# -


