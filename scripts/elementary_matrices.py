import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_box, CellType
import os


results_dir = os.path.dirname(os.path.abspath(__file__)) + "/results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Directory '{results_dir}' created.")


# Physical parameters (waveguide 1)
rho = 7800  # density [kg/m^3]
E = 2*1.e11  # Young's modulus [N/m^2]
nu = 0.3  # Poisson's ratio [-]
eta = 0.01  # damping ratio [-]
h_y = 30*1e-2  # height [m]
h_z = 20*1e-2  # width [m]
d = 5*1e-2  # depth [m]

# Lame parameters
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

parametric_cell = create_box(
    MPI.COMM_WORLD,
    [np.array([-1, -1, -1]), np.array([1, 1, 1])],
    [1, 1, 1],
    cell_type=CellType.hexahedron,
)

coordinates_cell = parametric_cell.geometry.x
connectivity_cell = parametric_cell.topology.connectivity(3, 0).array.reshape(-1, 8)

np.save(results_dir + "coordinates_nodes_parametric_cell.npy", coordinates_cell)
np.save(results_dir + "connectivity_parametric_cell.npy", connectivity_cell)

print(f"Coordinates parametric cell {coordinates_cell}")
print(f"Connectivity parametric cell {connectivity_cell}")

dim = parametric_cell.topology.dim
print(f"Mesh topology dimension d={dim}.")
shape = (dim, )

def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)

V_cell = fem.functionspace(parametric_cell, ("Lagrange", 1, shape))
u_cell = ufl.TrialFunction(V_cell)
v_cell = ufl.TestFunction(V_cell)

dx_cell = ufl.Measure("dx", domain=parametric_cell)

k_form_cell = fem.form(ufl.inner(sigma(u_cell), epsilon(v_cell)) * dx_cell)
m_form_cell = fem.form(rho * ufl.inner(u_cell, v_cell) * dx_cell)

k_scipy_cell = fem.assemble_matrix(k_form_cell).to_scipy()
k_cell = k_scipy_cell.todense()

m_scipy_cell = fem.assemble_matrix(m_form_cell).to_scipy()
m_cell = m_scipy_cell.todense()

print(f"Total mass cell fem {m_cell.sum(axis=1).sum()/3}")
print(f"Total mass cell  {rho * 8}")

np.save(results_dir + "k_cell.npy", k_cell)
np.save(results_dir + "m_cell.npy", m_cell)

# Substructure

Ny, Nz = 4, 4
structure = create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([d, h_y, h_z])],
    [1, Ny, Nz],
    cell_type=CellType.hexahedron,
)

def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], d)

coordinates_structure = structure.geometry.x
connectivity_structure = structure.topology.connectivity(3, 0).array.reshape(-1, 8)

np.save(results_dir + "coordinates_nodes_structure.npy", coordinates_structure)
np.save(results_dir + "connectivity_structure.npy", connectivity_structure)

assert connectivity_structure.shape[0] == Ny * Nz

V_structure = fem.functionspace(structure, ("Lagrange", 1, shape))

left_dofs_structure = fem.locate_dofs_geometrical(V_structure, left)
right_dofs_structure = fem.locate_dofs_geometrical(V_structure, right)

np.save(results_dir + "left_dofs_structure.npy", left_dofs_structure)
np.save(results_dir + "right_dofs_structure.npy", right_dofs_structure)

u_structure = ufl.TrialFunction(V_structure)
v_structure = ufl.TestFunction(V_structure)

dx_structure = ufl.Measure("dx", domain=structure)

k_form_structure = fem.form(ufl.inner(sigma(u_structure), epsilon(v_structure)) * dx_structure)
k_scipy_structure = fem.assemble_matrix(k_form_structure).to_scipy()
k_structure = k_scipy_structure.todense()


m_form_structure = fem.form(rho * ufl.inner(u_structure, v_structure) * dx_structure)
m_scipy_structure = fem.assemble_matrix(m_form_structure).to_scipy()
m_structure = m_scipy_structure.todense()

print(f"Total mass structure fem {m_structure.sum(axis=1).sum()/3}")
print(f"Total mass structure  {rho * d*h_y*h_z}")

print(f"Stiffness matrix structure shape: {k_structure.shape}")    
print(f"Mass matrix structure shape: {m_structure.shape}")  
np.save(results_dir + "k_structure.npy", k_structure)
np.save(results_dir + "m_structure.npy", m_structure)


# Get the Jacobian of the mapping
J_structure = ufl.Jacobian(structure)

# Compute the Jacobian determinant
J_det = ufl.det(J_structure)

# Create a function to evaluate the Jacobian (discontinuous piecewise linear function)
dg1_space = fem.functionspace(structure, ("DG", 1))
J_func = fem.Function(dg1_space)

# J_func.interpolate(J_det)

# Integrate the Jacobian determinant over the domain
J_det_form = J_det * dx_structure
J_det_integral = fem.assemble_scalar(fem.form(J_det_form))

print("Jacobian of mapping:")
print("Jacobian matrix:", J_structure)
print("Jacobian determinant:", J_det)
print("Integrated Jacobian determinant:", J_det_integral)


# np.save(results_dir + "determinant_jacobian.npy", J_func.vector[:])
np.save(results_dir + "determinant_integrand.npy", J_det_integral)