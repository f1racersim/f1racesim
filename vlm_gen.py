import aerosandbox as asb
import aerosandbox.numpy as np

# Your specific V-22 Drone Wing
wing = asb.Wing(
    name="V-22 Drone Wing",
    xsecs=[
        asb.WingXSec(xyz_le=[-0.05, -0.5, 0], chord=0.12, airfoil=asb.Airfoil("naca23012")),
        asb.WingXSec(xyz_le=[0, 0, 0], chord=0.20, airfoil=asb.Airfoil("naca23012")),
        asb.WingXSec(xyz_le=[-0.05, 0.5, 0], chord=0.12, airfoil=asb.Airfoil("naca23012")),
    ]
)

airplane = asb.Airplane(wings=[wing])

def solve_vlm(alpha_deg):
    op = asb.OperatingPoint(velocity=15, alpha=alpha_deg)
    return asb.VortexLatticeMethod(airplane, op).run()

# Solve at 0 and 5 degrees to get slopes
res0 = solve_vlm(0)
res5 = solve_vlm(5)

# Calculations for C
alpha_rad = np.radians(5)
cl_alpha = (res5['CL'] - res0['CL']) / alpha_rad
cm_alpha = (res5['Cm'] - res0['Cm']) / alpha_rad
# Induced drag factor k: CDi = k * CL^2
k_induced = res5['CD'] / (res5['CL']**2 + 1e-6) 

print("\n--- PASTE THIS INTO YOUR main.c INITIALIZATION ---")
print(f"wings[num_wings].cl_alpha  = {cl_alpha:.4f};")
print(f"wings[num_wings].cl0       = {res0['CL']:.4f};")
print(f"wings[num_wings].cd0       = 0.012; // Profile drag for NACA 23012")
print(f"wings[num_wings].k_induced = {k_induced:.4f};")
print(f"wings[num_wings].cm_alpha  = {cm_alpha:.4f};")
print(f"wings[num_wings].cm0       = {res0['Cm']:.4f};")
print(f"wings[num_wings].area      = {wing.area():.4f};")
print(f"wings[num_wings].chord     = {wing.mean_aerodynamic_chord():.4f};")