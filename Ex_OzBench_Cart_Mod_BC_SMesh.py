# ### OzBench model with freeslip boundaries in structured quad box

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy
import gmsh
import os
import time
import h5py
import math

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
test_no = 1

# linear or nonlinear version
linear = False ### False for NL version

# number of steps
nsteps = 3

# # Recycle rate of particles
# recycle_rate = 0

# cohesion value 
cohesion = 0.06

# mesh dimensions and resolution
xmin, xmax = 0., 4.0
ymin, ymax = 0., 1.0
resx, resy = 192, 48

# +
# creating output directory
meshes_dir = os.path.join(os.path.join("./"), "meshes/")
output_dir = os.path.join(os.path.join("./output/"), f"OzBench_Cart_Mod_BC_SMesh_Test{test_no}/")

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)
    step = 0
    time = 0.
# -

# #### Create Mesh

mesh = uw.meshing.StructuredQuadBox(elementRes =(int(resx),int(resy)),
                                    minCoords=(xmin,ymin), 
                                    maxCoords=(xmax,ymax))

# ### Create Stokes Object

# +
v = uw.discretisation.MeshVariable('U', mesh,  mesh.dim, degree=2 )
p = uw.discretisation.MeshVariable('P', mesh, 1, degree=1,  continuous=True)

strain_rate_inv2 = uw.discretisation.MeshVariable("Strain Rate", mesh, 1, degree=v.degree)
dev_stress_inv2 = uw.discretisation.MeshVariable("Stress", mesh, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("Viscosity", mesh, 1, degree=1)
# -


stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p )
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(v)

# +
# with mesh.access():
#     print(mesh.data.shape)
#     print(v.data.shape)
# -

# #### Setup Swarm

swarm     = uw.swarm.Swarm(mesh=mesh)
materialVariable  = uw.swarm.IndexSwarmVariable("material", swarm, indices=5)
swarm.populate(fill_param=2)

# +
# initialise the 'materialVariable' data to represent two different materials. 
upperMantleIndex = 0
lowerMantleIndex = 1
upperSlabIndex   = 2
lowerSlabIndex   = 3
coreSlabIndex    = 4

# Initial material layout has a flat lying slab with at 15\degree perturbation
lowerMantleY   = 0.4
slabLowerShape = mpltPath.Path([ (1.2,0.925 ), (3.25,0.925 ), (3.20,0.900), (1.2,0.900), (1.02,0.825), (1.02,0.850) ])
slabCoreShape  = mpltPath.Path([ (1.2,0.975 ), (3.35,0.975 ), (3.25,0.925), (1.2,0.925), (1.02,0.850), (1.02,0.900) ])
slabUpperShape = mpltPath.Path([ (1.2,1.000 ), (3.40,1.000 ), (3.35,0.975), (1.2,0.975), (1.02,0.900), (1.02,0.925) ])


with swarm.access(materialVariable):
    # initialise everying to be upper mantle material
    materialVariable.data[...] = upperMantleIndex
    
    #indexing lower mantle material
    indices = np.argwhere(swarm.data[:,1] <= lowerMantleY)
    materialVariable.data[indices] = lowerMantleIndex
    
    # change matieral index if the particle is not upper mantle
    for index in range( len(swarm.data) ):
        coord = swarm.data[index][:]
        if slabCoreShape.contains_point(tuple(coord)):
                materialVariable.data[index] = coreSlabIndex
        if slabUpperShape.contains_point(tuple(coord)):
                materialVariable.data[index] = upperSlabIndex
        if slabLowerShape.contains_point(tuple(coord)):
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

    mesh.vtk(output_dir+f"OzBench_Cart_Mod_BC_SMesh_Test{test_no}.vtk")
    pvmesh = pv.read(output_dir+f"OzBench_Cart_Mod_BC_SMesh_Test{test_no}.vtk")

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


def updateFields():

    nodal_strain_rate_inv2.solve()

    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
    nodal_visc_calc.solve(_force_setup=True)

    nodal_tau_inv2.uw_function = 2. * stokes.constitutive_model.Parameters.shear_viscosity_0 * stokes._Einv2
    nodal_tau_inv2.solve(_force_setup=True)


# -

# #### Boundary Conditions

# +
sol_vel = sympy.Matrix([0., 0.])

# # No slip left & right
# stokes.add_dirichlet_bc( sol_vel, ["Left", "Right"],  [0,1] )  # left/right: components, function, markers


# Freeslip left & right
stokes.add_dirichlet_bc( sol_vel, ["Left", "Right"],  [0] )  # left/right: components, function, markers
# free slip top and bottom
stokes.add_dirichlet_bc( sol_vel, ["Top", "Bottom"],  [1] )  # top/bottom: components, function, markers 


# periodic bd conditions
# 1. gmsh with peridoic mesh
# 2. in petsc impose periodicity on dm
# -

# #### Setup Density and Viscosity of Materials

# +
mantleDensity = 0.0
slabDensity   = 1.0 

density_fn = materialVariable.createMask([mantleDensity, 
                                          mantleDensity, 
                                          slabDensity, 
                                          slabDensity, 
                                          slabDensity])

stokes.bodyforce = sympy.Matrix([0, -1 * density_fn ])
# -

stokes.bodyforce

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
                                                slabYieldVisc, 
                                                coreViscosity])

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_mat_fn
stokes.saddle_preconditioner = 1.0 / viscosity_mat_fn


# -

# #### Save Mesh to h5/xdmf file

def saveData(step, outputPath):

    ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
    mesh.petsc_save_checkpoint(index=step, 
                               meshVars=[v, p, strain_rate_inv2, node_viscosity], 
                               outputPath=outputPath)
    
    
    #### save the swarm and selected variables
    swarm.petsc_save_checkpoint('swarm', step, outputPath)

# +
# creating profile points to extract data

# profile1 in upper part
coord_x = np.linspace(0.8, 1.5, 60, endpoint='True')
coord_y = (1000 - 15)/1000
coords = np.zeros((len(coord_x), 2))
coords[:,0] = coord_x
coords[:,1] = coord_y
profile1_coords = coords

# profile2 in core
coord_x = np.linspace(0.8, 1.5, 60, endpoint='True')
coord_y = (1000 - 50)/1000
coords = np.zeros((len(coord_x), 2))
coords[:,0] = coord_x
coords[:,1] = coord_y
profile2_coords = coords

# profile2 in lower part
coord_x = np.linspace(0.8, 1.5, 60, endpoint='True')
coord_y = (1000 - 85)/1000
coords = np.zeros((len(coord_x), 2))
coords[:,0] = coord_x
coords[:,1] = coord_y
profile3_coords = coords
# -

# #### Scaling

rho_M             = 1.
g_M               = 1.
Height_M          = 1.
viscosity_M       = 1.

rho_N             = 80.0  # kg/m**3  note delta rho
g_N               = 9.81    # m/s**2
Height_N          = 1000e3   # m
viscosity_N       = 2e19   #Pa.sec or kg/m.sec

#Non-dimensional (scaling)
rho_scaling = rho_N/rho_M
viscosity_scaling = viscosity_N/viscosity_M
g_scaling = g_N/g_M
Height_scaling = Height_N/Height_M
pressure_scaling = rho_scaling * g_scaling * Height_scaling
time_scaling = viscosity_scaling/pressure_scaling
time_scaling_myr = (viscosity_scaling/pressure_scaling)*(1/(365*24*60*60))/1e6 # in million years
strainrate_scaling = 1./time_scaling
mm_yr               = 1e3*365*24*60*60
velocity_scaling    = (Height_scaling/time_scaling)*mm_yr


# \begin{align}
# {\tau}_N = \frac{{\rho}_{0N}{g}_N{l}_N}{{\rho}_{0M}{g}_M{l}_M} {\tau}_M
# \end{align}
#
# \begin{align}
# {V}_N = \frac{{\eta}_{0M}}{{\eta}_{0N}}\frac{{\rho}_{0N}{g}_N{{l}_N}^2}{{\rho}_{0M}{g}_M{{l}_M}^2} {V}_M
# \end{align}

def save_pf_data_h5(_profile_coords, _profile_depth, _step, _time, _vrms):
    """
    save profile data to h5 file.
    """
    # evaluate field data at coords
    velocity_pf            = uw.function.evaluate(v.sym, _profile_coords)
    viscosity_pf           = uw.function.evaluate(node_viscosity.sym, _profile_coords)
    stressInv_pf           = uw.function.evaluate(dev_stress_inv2.sym, _profile_coords)
    strainRateInv_pf       = uw.function.evaluate(strain_rate_inv2.sym, _profile_coords)

    if uw.mpi.rank == 0:
        with h5py.File(output_dir+'profiles_data.h5', 'a') as pf_h5:
            pf_h5_grp = pf_h5.create_group(str(_profile_depth)+'/'+str(_step))
            pf_h5_grp.create_dataset('coords', data=_profile_coords)
            pf_h5_grp.create_dataset('velocity', data=velocity_pf)
            pf_h5_grp.create_dataset('viscosity', data=viscosity_pf)
            pf_h5_grp.create_dataset('stressInv', data=stressInv_pf)
            pf_h5_grp.create_dataset('strainRateInv', data=strainRateInv_pf)
            pf_h5_grp.create_dataset('time', data=_time)
            pf_h5_grp.create_dataset('vrms', data=_vrms)


# #### Solver Loop for Multiple Iterations

# removing h5 file if an old one exist
if uw.mpi.rank == 0:
    if os.path.isfile(output_dir+'profiles_data.h5'):
        os.remove(output_dir+'profiles_data.h5')


def _vrms(_mesh, _field):
    field_integral = uw.maths.Integral(_mesh, _field.fn.dot(_field.fn)).evaluate()
    volume = uw.maths.Integral(_mesh, 1.0).evaluate()
    return math.sqrt(field_integral/volume)


while step < nsteps:

    # save loop
    if step % 1 == 0:
        if uw.mpi.rank==0:
            vrms = _vrms(mesh, v)
            print('step = {0:6d}; time = {1:.3e}; Vrms = {2:.3e}'.format(step, time, vrms))
            print('step = {0:6d}; time (dim) = {1:.3e}; Vrms (dim) = {2:.3e}'.format(step,time*time_scaling_myr,
                                                                                     vrms*velocity_scaling))
        # update fields first
        updateFields()
        # save mesh variables
        saveData(step, output_dir)
        # save pf data
        save_pf_data_h5(profile1_coords, 15, step, time, vrms)
        save_pf_data_h5(profile2_coords, 50, step, time, vrms)
        save_pf_data_h5(profile3_coords, 85, step, time, vrms)

    # solve stokes 
    stokes.solve(zero_init_guess=False)
    
    # estimate dt
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
    pv.global_theme.jupyter_backend = 'trame'
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
    pv.global_theme.jupyter_backend = 'trame'
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
                scalars=u_mag, use_transparency=False, opacity=1.0, clim=[1e-5, 1e-4])
    
    pl.add_arrows(arrow_loc[::20], arrow_length[::20], mag=4000, color='Grey')
    
    pl.show(cpos="xy")


# mesh vtk file 
mesh_file = output_dir+f"OzBench_Cart_Mod_BC_SMesh_Test{test_no}.vtk"
# cmap = plt.cm.RdYlBu_r

# plot velocities
plot_vel()

# plot strain rate 
cmap = plt.get_cmap('RdYlBu_r')
plot_field(mesh_file, mesh.data, strain_rate_inv2, _cmap=cmap, _cb_min=2e-5, _cb_max=2e-2, 
           _log_scale=True)

# plot viscosity
cmap = plt.get_cmap('RdYlBu')
plot_field(mesh_file, mesh.data, node_viscosity, _cmap=cmap, _cb_min=1, _cb_max=500, 
           _log_scale=False)

# plot stress inv 
cmap = plt.get_cmap('RdYlBu_r', 20)
plot_field(mesh_file, mesh.data, dev_stress_inv2, _cmap=cmap, _cb_min=2e-5, _cb_max=2e-1, 
           _log_scale=False)


