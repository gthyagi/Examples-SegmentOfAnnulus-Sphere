# ### Slab Subduction in Segment of Annulus

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
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt

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
test_no = 2
inner_radius = (6371.-1000)/6371.
outer_radius = 6371./6371.
angle_extent = 36

# linear or nonlinear version
linear = False ### False for NL version

# number of steps
nsteps = 3

# # Recycle rate of particles
# recycle_rate = 0

# cohesion value 
cohesion = 0.0096

# +
# creating output directory
meshes_dir = os.path.join("./", "meshes/")
output_dir = os.path.join("./output/", f"SOA_SlabSubduction_Test{test_no}/")

reload = False

if False: #uw.mpi.rank == 0:
    if os.path.exists(output_dir):
        reload = True
        restart_step = 205
        pv_mesh_data = pv.XdmfReader(output_dir + f'step_0{restart_step}.xdmf').read()
        time_d =  pv_mesh_data['time_time'][0]
else:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)
    step = 0
    time = 0.
    restart_step = -1
# -

output_dir


def rtheta2xy(data):
    """
    converts (r, theta) to (x, y) coordinates
    """
    newcoords 		= np.zeros((len(data[:,0]),2))
    newcoords[:,0] 	= data[:,0]*np.cos(data[:,1]*np.pi/180.0)
    newcoords[:,1] 	= data[:,0]*np.sin(data[:,1]*np.pi/180.0)
    return newcoords


def xy2rtheta(data):
    """
    converts (x, y) to (r, theta) coordinates
    """
    newcoords 		= np.zeros((len(data[:,0]),2))
    newcoords[:,0] 	= np.sqrt(data[:,0]**2 + data[:,1]**2)
    newcoords[:,1] 	= np.arctan2(data[:,1],data[:,0]) * 180 / np.pi
    return newcoords


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

# #### Scaling setup



# #### Create Mesh

# creating mesh in gmsh
if uw.mpi.rank == 0:
    SegmentOfAnnulusEdit(radiusOuter=outer_radius, radiusInner=inner_radius, 
                         angleExtent=angle_extent, cellSize=0.005,
                         filename=meshes_dir+f"soa_mesh_test{test_no}.msh")

# start timing
timing.reset()
timing.start()

# creating uw mesh
mesh = uw.discretisation.Mesh(meshes_dir+f"soa_mesh_test{test_no}.msh", 
                              simplex=True, qdegree=3, markVertices=False, useRegions=True, useMultipleTags=True, )

# stop timing
timing.stop()
uw.timing.print_table()



if uw.mpi.size == 1:
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1500, 1000]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh.vtk(output_dir+f"soa_mesh_test{test_no}.vtk")
    pvmesh = pv.read(output_dir+f"soa_mesh_test{test_no}.vtk")

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Blue", "wireframe", opacity=0.5,)

    pl.show(cpos="xy")

# #### Create Stokes Object

# +
# Mesh Variables
v = uw.discretisation.MeshVariable('U', mesh, mesh.dim, degree=2 )
p = uw.discretisation.MeshVariable('P', mesh, 1, degree=1,  continuous=True)

strain_rate_inv2 = uw.discretisation.MeshVariable("SR", mesh, 1, degree=2)
dev_stress_inv2 = uw.discretisation.MeshVariable("stress", mesh, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("viscosity", mesh, 1, degree=1)

timeField      = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)
materialField  = uw.discretisation.MeshVariable("material", mesh, 1, degree=1)
# -

# Stokes setup
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(v)

# #### Swarm Setup

# ##### What is recycle_rate and it's use? 

# creating swarm
swarm = uw.swarm.Swarm(mesh=mesh)
# swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=recycle_rate)
# materialVariable = uw.swarm.SwarmVariable("M", swarm, size=1, proxy_continuous=False, proxy_degree=0)
materialVariable  = uw.swarm.IndexSwarmVariable("M", swarm, indices=5)
swarm.populate(fill_param=10)

# +
# printing mesh and swarm info
print(mesh.data.shape)

with swarm.access(materialVariable):
    print(materialVariable.data.shape, '\n', materialVariable.swarm.data.shape)
# -

if uw.mpi.size == 1:

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1500, 1000]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "static"
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
        point_cloud.point_data["M"] = materialVariable.data.copy()

    pl.add_points(point_cloud, cmap="viridis", render_points_as_spheres=False,
                  point_size=1, opacity=0.66,)
    pl.add_mesh(pvmesh, "Blue", "wireframe")

    pl.show(cpos="xy")

# #### Creating Slab Layers

# no.of points forming layers
num_lay_pt = 200
slab_theta1 = 78
slab_theta2 = 98

# points to create upper part points
slab_upper_top_theta = np.linspace(slab_theta1, slab_theta2, num=num_lay_pt, endpoint=True)
slab_upper_top_radius = 1.0
slab_upper_top_rtheta = np.zeros((len(slab_upper_top_theta)+1, 2))
slab_upper_top_rtheta[:len(slab_upper_top_theta),0] = slab_upper_top_radius
slab_upper_top_rtheta[:len(slab_upper_top_theta),1] = slab_upper_top_theta
pt1 = ((1-(75.0/6371.0)), 99.6)   # depth and distance of pt1 is calculated from the slope
slab_upper_top_rtheta[-1][0] = pt1[0]
slab_upper_top_rtheta[-1][1] = pt1[1]
slab_upper_top = rtheta2xy(slab_upper_top_rtheta)

# points to create upper part points
pt2 = (pt1[0]-(30/6371.), pt1[1])
slab_upper_bot_theta = np.linspace(slab_theta2, slab_theta1+0.5, num=num_lay_pt, endpoint=True)
slab_upper_bot_radius = (6371.-30)/6371.
slab_upper_bot_rtheta = np.zeros((len(slab_upper_bot_theta)+1, 2))
slab_upper_bot_rtheta[0][0] = pt2[0]
slab_upper_bot_rtheta[0][1] = pt2[1]
slab_upper_bot_rtheta[1:,0] = slab_upper_bot_radius
slab_upper_bot_rtheta[1:,1] = slab_upper_bot_theta
slab_upper_bot = rtheta2xy(slab_upper_bot_rtheta)

# points to create lower part points
pt3 = (pt1[0]-(30/6371.)-(40/6371), pt1[1])
slab_lower_top_theta = np.linspace(slab_theta2, slab_theta1+1.0, num=num_lay_pt, endpoint=True)
slab_lower_top_radius = (6371.-70)/6371.
slab_lower_top_rtheta = np.zeros((len(slab_lower_top_theta)+1, 2))
slab_lower_top_rtheta[0][0] = pt3[0]
slab_lower_top_rtheta[0][1] = pt3[1]
slab_lower_top_rtheta[1:,0] = slab_lower_top_radius
slab_lower_top_rtheta[1:,1] = slab_lower_top_theta
slab_lower_top = rtheta2xy(slab_lower_top_rtheta)

# points to create lower part points
pt4 = (pt1[0]-(30/6371.)-(40/6371)-(30/6371.), pt1[1])
slab_lower_bot_theta = np.linspace(slab_theta2, slab_theta1+1.5, num=num_lay_pt, endpoint=True)
slab_lower_bot_radius = (6371.-100)/6371.
slab_lower_bot_rtheta = np.zeros((len(slab_lower_bot_theta)+1, 2))
slab_lower_bot_rtheta[0][0] = pt4[0]
slab_lower_bot_rtheta[0][1] = pt4[1]
slab_lower_bot_rtheta[1:,0] = slab_lower_bot_radius
slab_lower_bot_rtheta[1:,1] = slab_lower_bot_theta
slab_lower_bot = rtheta2xy(slab_lower_bot_rtheta)

# **Plot the initial positions for the particle swarm and colour by material type**
#
# We are using a vis store object to keep all our figures together and allow them to be revisualised later so create this here and pass it to all the figures created later. We'll also name this figure to make it easy to find later when revisualising.

# combining points to form polygon. Note: it should either in clockwise or anti-clockwise order
reversed_slab_upper_bot = slab_upper_bot[::-1]
reversed_slab_lower_top = slab_lower_top[::-1]
reversed_slab_lower_bot = slab_lower_bot[::-1]
slabUpper = mpltPath.Path( np.concatenate((slab_upper_top, slab_upper_bot), axis=0) )
slabCore  = mpltPath.Path( np.concatenate((reversed_slab_upper_bot, slab_lower_top), axis=0) )
slabLower = mpltPath.Path( np.concatenate((reversed_slab_lower_top, slab_lower_bot), axis=0) )

# **Allocate materials to particles**

# +
# initialise the 'materialVariable' data to represent two different materials. 
upperMantleIndex = 0
lowerMantleIndex = 1
upperSlabIndex   = 2
lowerSlabIndex   = 3
coreSlabIndex    = 4

# initialise everying to be upper mantle material
with swarm.access(materialVariable):
    materialVariable.data[...] = upperMantleIndex
    
    #indexing lower mantle material
    rtheta_coord = xy2rtheta(swarm.data)
    indices = np.argwhere(rtheta_coord[:,0] <= (6371.0-660.0)/6371.0)
    materialVariable.data[indices] = lowerMantleIndex
    
    # change matieral index if the particle is not upper mantle
    for index in range( len(swarm.data) ):
        coord = swarm.data[index][:]
        if rtheta_coord[:,0][index] > (6371.0-660.0)/6371.0:
            if slabCore.contains_point(tuple(coord)):
                    materialVariable.data[index] = coreSlabIndex
            if slabUpper.contains_point(tuple(coord)):
                    materialVariable.data[index] = upperSlabIndex
            if slabLower.contains_point(tuple(coord)):
                    materialVariable.data[index] = lowerSlabIndex
# -

if uw.mpi.size == 1:

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1500, 1000]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "static"
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
        point_cloud.point_data["M"] = materialVariable.data.copy()

    pl.add_points(point_cloud, cmap=plt.get_cmap('tab10', 5), render_points_as_spheres=False,
                  point_size=1.5, opacity=0.76,)
    pl.add_mesh(pvmesh, "Blue", "wireframe")

    pl.show(cpos="xy")

# #### Additional Mesh Variables

# +
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
nodal_strain_rate_inv2.smoothing = 0.
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
nodal_visc_calc.smoothing = 0.
nodal_visc_calc.petsc_options.delValue("ksp_monitor")


nodal_tau_inv2 = uw.systems.Projection(mesh, dev_stress_inv2)
nodal_tau_inv2.uw_function = 2. * stokes.constitutive_model.Parameters.shear_viscosity_0 * stokes._Einv2
nodal_tau_inv2.smoothing = 0.
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

# matProj = uw.systems.Projection(mesh, materialField)
# matProj.uw_function = materialVariable.sym[0]
# matProj.smoothing = 0.
# matProj.petsc_options.delValue("ksp_monitor")


### create function to update fields
def updateFields(time):
    
    with mesh.access(timeField):
        timeField.data[:,0] = time

    nodal_strain_rate_inv2.solve()

    # matProj.uw_function = materialVariable.sym[0] 
    # matProj.solve(_force_setup=True)

    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
    nodal_visc_calc.solve(_force_setup=True)

    nodal_tau_inv2.uw_function = 2. * stokes.constitutive_model.Parameters.shear_viscosity_0 * stokes._Einv2
    nodal_tau_inv2.solve(_force_setup=True)


# -

# #### Boundary Conditions

# +
sol_vel = sympy.Matrix([0.,0.])

### No slip left & right
stokes.add_dirichlet_bc( sol_vel, ["Left", "Right"],  [0, 1] )  # left/right: components, function, markers
### No slip top and bottom
stokes.add_dirichlet_bc( sol_vel, ["Top", "Bottom"],  [0, 1] )  # top/bottom: components, function, markers 
# -

# #### Setup Density and Viscosity of Materials

# defining fn to convert (x, y) to (r, theta)
radius_fn = sympy.sqrt(mesh.X.dot(mesh.X))  # normalise by outer radius if not 1.0
unit_rvec = mesh.X / (radius_fn)

# +
mantleDensity = 0.0
slabDensity   = 1.0 

density_fn = materialVariable.createMask([mantleDensity, 
                                          mantleDensity, 
                                          slabDensity, 
                                          slabDensity, 
                                          slabDensity])

stokes.bodyforce = -1 * (density_fn * unit_rvec)

# +
# # it is working only component wise not as a vector (issue)
# uw.function.evaluate(den_fn[1], mesh.data)

# +
upperMantleViscosity =    1.0
lowerMantleViscosity =  100.0
slabViscosity        =  500.0
coreViscosity        =  500.0

# linear solve
viscosity_L_fn = materialVariable.createMask([upperMantleViscosity, 
                                              lowerMantleViscosity, 
                                              slabViscosity, 
                                              slabViscosity, 
                                              coreViscosity])

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_L_fn
stokes.saddle_preconditioner = 1.0 / viscosity_L_fn
# -

# #### Initial Linear Solve

# +
# Set solve options here (or remove default values
# stokes.petsc_options["ksp_monitor"] = None

if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.tolerance = 1.0e-5
# -

stokes.solve(zero_init_guess=False)

# #### Introduce NL Viscosity

# +
vonMises = 0.5 * cohesion / (stokes._Einv2 + 1.0e-18)

# The upper slab viscosity is the minimum of the 'slabViscosity' or the 'vonMises' 
eta_min = upperMantleViscosity
slabYieldVisc = sympy.Max(eta_min, sympy.Min(vonMises, slabViscosity))

viscosity_mat_fn = materialVariable.createMask([upperMantleViscosity, 
                                                lowerMantleViscosity, 
                                                slabYieldVisc, 
                                                slabViscosity, 
                                                coreViscosity])

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_mat_fn
stokes.saddle_preconditioner = 1.0 / viscosity_mat_fn


# -

# #### Save Mesh to h5/xdmf file

def saveData(step, outputPath):

    ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
    mesh.petsc_save_checkpoint(index=step, 
                               meshVars=[v, p, strain_rate_inv2, node_viscosity, timeField], 
                               outputPath=outputPath)
    
    
    #### save the swarm and selected variables
    swarm.petsc_save_checkpoint('swarm', step, outputPath)

# #### Solver Loop for Multiple Iterations

while step < nsteps:
    
    ### save loop
    if step % 1 == 0 and step != restart_step:
        if uw.mpi.rank==0:
            print(f'\n\nSave data: \n\n')
        ### update fields first
        updateFields(time = time)
        ### save mesh variables
        saveData(step, output_dir)

    ### solve stokes 
    stokes.solve(zero_init_guess=False)
    ### estimate dt
    dt = 0.5 * stokes.estimate_dt()

    ### advect the swarm
    swarm.advection(stokes.u.sym, dt, corrector=False, evalf=True)
        
    step+=1
    time+=dt


def plot_field(_mesh_vtk, _mesh_data, _field, _cmap='Viridis', _cb_min=0, _cb_max=1, _log_scale=False,
               _show_edges=False):

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'static'
    pv.global_theme.smooth_shading = True

    pvmesh = pv.read(_mesh_vtk)

    with mesh.access(_field):
        pvmesh.point_data["data"] = _field.rbf_interpolate(_mesh_data, nnn=1)

    field_data = pvmesh.point_data.values()[0]
    
    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh, cmap=_cmap, scalars=field_data, edge_color="Grey",
                show_edges=_show_edges, use_transparency=False, clim=[_cb_min, _cb_max],
                opacity=1.0, log_scale=_log_scale)

    pl.show(cpos="xy")


def plot_vel():
    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'static'
    pv.global_theme.smooth_shading = True
    
    pvmesh = pv.read(mesh_file)
        
    with mesh.access():
        v_sol = v.rbf_interpolate(mesh.data, nnn=1)
        mesh_coords = mesh.data.copy()  
        
    arrow_loc = np.zeros((mesh_coords.shape[0], 3))
    arrow_loc[:, 0:2] = mesh_coords[...]
    
    arrow_length = np.zeros((mesh_coords.shape[0], 3))
    arrow_length[:, 0:2] = v_sol[...]
    
    def get_vel_mag(_vel):
        return np.sqrt(_vel[:,0]**2 + _vel[:,1]**2)
        
    u_mag = get_vel_mag(v_sol)
    
    pl = pv.Plotter(notebook=True)
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=False, 
                scalars=u_mag, use_transparency=False, opacity=1.0, clim=[5e-5, 5e-4])
    
    pl.add_arrows(arrow_loc[::10], arrow_length[::10], mag=150, color='Grey')
    
    pl.show(cpos="xy")


# mesh vtk file 
mesh_file = output_dir+f"soa_mesh_test{test_no}.vtk"
# cmap = plt.cm.RdYlBu_r

# plot velocities
plot_vel()

# plot strain rate 
cmap = plt.get_cmap('RdYlBu_r', 20)
plot_field(mesh_file, mesh.data, strain_rate_inv2, _cmap=cmap, _cb_min=2e-5, _cb_max=2e-1, 
           _log_scale=True)

# plot viscosity
cmap = plt.get_cmap('RdYlBu')
plot_field(mesh_file, mesh.data, node_viscosity, _cmap=cmap, _cb_min=1, _cb_max=500, 
           _log_scale=False)

# plot stress inv 
cmap = plt.get_cmap('RdYlBu_r', 20)
plot_field(mesh_file, mesh.data, dev_stress_inv2, _cmap=cmap, _cb_min=2e-5, _cb_max=2e-1, 
           _log_scale=False)




