FEN ZI User Manual
Authors: Narayan Ganesan, Sandeep Patel, and Michela Taufer

Last update: July 6, 2011

1 FEN ZI Overview
The GPU-based code FEN ZI (yun dong de FEN ZI in Mandarin or moving MOLECULES in English) is an MD code enabling simulations at constant energy (NVE), constant temperature (NVT), and constant pressure and temperature (NPT). FEN ZI uses a modified version of the CHARMM force field {1} in terms of force field functional forms and measurement units. FEN ZI treats the conditionally convergent electrostatic interaction energy exactly using the Particle Mesh Ewald method (PME) for solution of Poisson's Equation for the electrostatic potential under periodic boundary conditions. The entire MD simulation (i.e., intra-molecular and long range potentials including the Ewald summation method and PME) is performed on one single GPU.

To perform constant temperature molecular dynamics simulations, FEN ZI maintains constant temperature using velocity reassignment. The velocity reassignment is performed every 20,000 time steps to set the system temperature. We acknowledge that the current method based on velocity reassignment does not reproduce the canonical ensemble. To address this issue, we are currently integrating the Nose-Hoover thermostat {2} in the code. FEN ZI performs molecular dynamics simulations at constant pressure by using the Andersen barostat {3}, wherein the Hamiltonian of the system includes terms arising due to fluctuations in the volume.

FEN ZI performs energy minimization of newly built and unminimized structures by using the hybrid combination of conjugated gradient and steepest descent methods. The conjugated gradient coefficient is calculated via the Fletcher-Reeves formula {4}. Although efficient, the conjugate gradient method is unstable with respect to numerical perturbations and precision. Under finite precision assumptions, the conjugate vector may not preserve conjugation and could accumulate error over several steps of minimization {5}. As a result the final structure is not fully minimized by the conjugated gradient method alone. The method of steepest descent, although slower, is applied to the partially minimized structure until the residual is close to minimal energy. The switch from one method to the other is performed based on the total energy of the system that is monitored while progressively decreasing during the minimization.

A detailed description of FEN ZI can be found in:

N. Ganesan, B.A. Bauer, S. Patel, and M. Taufer: Structural, Dynamic, and Electrostatic Properties of Fully Hydrated DMPC Bilayers from Molecular Dynamics Simulations Accelerated with Graphical Processing Units (GPUs). Journal of Computational Chemistry, 2011. (In Press)

The code has been tested on these GPUs:

C2050
C1060
GTX 480
2 Compiling FEN ZI
2.1 Download and compilation command
Download FEN ZI from the FEN ZI webpage.

The compilation of FEN ZI requires you to work with the code in trunk (most updated, stable version of the code) or with a version of the code in branch.

[user@machine trunk]$ ls > install.com: command to compile FEN ZI > manual.txt > license.txt > notes.txt > README.txt > src: FEN ZI source > tests: examples with FEN ZI and CHARMM

To compile the code for your class of nVidia GPU architectures use the command install.com. The compilation command supports these options:

[user@machine trunk]$ ./install.com --help ../src/configure options

options:

configuration:

-h --help display this help and exit
--prefix (/usr/local) install architecture-independent files in prefix
-b --build (RELEASE) build type (DEBUG, PROFILE, RELEASE)
cuda:

--cudarch (sm_20) cuda architecture (sm_13, sm_20) use sm_20 for Kepler and Fermi GPUs
simulation:

--tssize (small) compile fenzi for a typical system size of tssize (vsmall, small, medium, large, cco, custom) --maxnb (580) maximum # of non-bonded neighbors --cblock (custom only) cell blocksize --catoms (custom only) cell atoms --maxdtyp (custom only) max # of dihedral types --maxatyp (custom only) max # of atom types --maxbtyp (custom only) max # of bond types --pconst (0) run FEN ZI in npt ensamble -default nvt/nve --npt (0) run FEN ZI in npt ensamble (shake works correctly) -default nvt/nve --consfix (0) run FEN ZI with consfix -default no consfix
2.2 Compilation for tested systems
FEN ZI has been tested for NVT and NVE simulations. An NPT integrator is under development. Parts of the code may be integrated in the code but not validated with molecular systems yet.

One important aspect of the compilation and testing is the size of the molecular system the code will be used for. This information has to be selected at the compilation time with the option --tssize or by providing install.com with customized values for these parameters: --cblock, --catoms, -maxdtyp, --maxatyp, --maxbtyp.

Examples of compilations (some of the molecular system are provided with the FEN ZI code in the data/examples directory.

A. Simulation of a small membrane (17K atoms)

./install.com --tssize small
B. Simulation of a large membrane (230K atoms)

./install.com --tssize large
C. Simulation of a two tubes in a water solvent with constraints

./install.com --consfix
3 Writing input files for FEN ZI
FEN ZI provide you with several examples you can use for testing and learning purposes. The key file for the setting of a simulation is the input file and ends in .in. The parameters and their values are described in this section.

3.1 Structure files
coordinates: Path to the coordinate file in CHARMM .crd, .pdb or .xyz format.

structure: Path to the CHARMM .psf file.

parameters: Path to the CHARMM .prm file.

topology: Path to the CHARMM .rtf file. This is needed only in the case of decoding the (different set of) atomic symbols used in the structure and parameter files.

3.2 Output files
outputname: energy output, e.g.,

outputname filename.out

outputtrajectory: trajectory output filename with prefix only, '.dcd' extensions and step counts are added automatically, e.g.,

outputtrajectory filename

3.3 MD temperature setting
temperature: Initial temperature and temperature for velocity reassignment in Kelvins. Default setting is 298

thermostat: Default setting is 5.0

tempresetTiming: Timing in number of steps of dynamics to reassign velocities to match initial temperature, e.g., 20000.

3.3 Minimization
minimizesteps: Number of minimization steps to run. In order to skip minimization, the line should either be commented or set to less than or equal to zero. Default setting is 0.

conjgradfac: Multiplier for the direction of motion of conjugate gradient method. The multiplier is small, e.g., 1e-13, minimized and new structures and reasonable valued for partially minimized structures, e.g., 1e-8. Default setting is 1e-10.

steepdesfac: Multiplier for the direction of motion of steepest descent method. The multiplier is small, e.g., 1e-6, minimized and new structures and reasonable valued for partially minimized structures, e.g., 1e-4. Default setting is 1e-5.

conjgradrate: Rate of increase of the conjgradfac parameter per 100 steps of minimization. This increases the rate of minimization of the structure as it is being progressively minimized, e.g., 10. Default setting is 10.

steepdesrate: Rate of increase of the steepdesrate parameter per 100 steps of minimization. This increases the rate of minimization of the structure as it is being progressively minimized, e.g., 2. Default setting is 2.

3.4 Dynamics
timestep: Timesteps defined in pico seconds e.g., 0.001.

startstep: Starting MD step. This is useful for restarting simulation from a different step number than the default value 0.

dynasteps: Total number of MD dynamics steps.

seed: seed value e.g., 2345. Random seed to be used for initial velocity reassignment.

3.5 Printing, output and checkpoint parameters
outputTiming: Frequency of printing in output. Keep this parameter low for high performance on GPUs.

checkpointTiming: Frequency of checkpointing on CPU. Keep this parameter low for high performance on GPUs. The restart filename is automatically generated based on input filename, followed by '.rst' extension. If a simulation is prematurely terminated, upon restarting, the configuration from the previously generated checkpoint file is loaded automatically, if the file is present in the same folder as the input file.

trajfrequencyTiming: Frequency of frames in trajectroy files

checkpointformat: ('Ascii' or 'Binary', 'A' or 'B'). Format to save the checkpoint files in. Checkpoint file contains information regarding the current timestep at which the checkpoint was generate, along with the number of atoms, the random seed, and the list of positions and corresponding velocities.

keepcheckpoints: (1 or 0) Option to keep previously generated checkpoint files. The kept checkpoint filenames will be generated based on the timestep at which it is generated. This option is useful to restart the simulation from an earlier time step in the simulation history.

printtostdout: Print energy outputs to stdout (1 or 0). If printtostdout is set to 0, the energies are printed to the outputfile whose name is constructed based on the input file, and the device number in use with the fileextension '.out'.

printlevel: Print level, i.e., printlevel 0 - no printing, 1 - output only energy, 2 - output energy and trajectory.

3.6 Cutoff parameters
cuton: Switching distance for VDW nonbond force.

cutoff: Cutoff distance for electrostatic and VDW forces.

pairlistdist: Nonbond buffer, e.g., 9.5. This cutoff radius is used to construct and update the pairlist; the region between cutoff and cutoff radius and cutoff serves as a buffer.

3.7 Comments
#: Lines starting with # are comments. All the keywords and syntax are

case insensitive.
3.8 GPU setting
gpublocksize: Number of GPU threads per thread block for dynamics computations.

3.9 Structural parameters
CellBasisVector1, CellBasisVector2, CellBasisVector3: dimensions of the cell, currently only cubic cells are supported.

CellBasisVector1: Set of three numbers for x size of the box length; second and third entries are zero; first entry corresponds to x size.

cellBasisVector2: Set of three numbers for y size of the box length; first and third entries are zero and second entry corresponds to y size.

cellBasisVector3: Set of three numbers for z size of the box length; first and second entries are zero; third entry corresponds to z size.

PMEGridSizeX, PMEGridSizeY, PMEGridSizez: Grid sizes for PME calculation for full electrostatics, e.g., PME uses CUDA FFT library which delivers best performance while maintaining accuracy, if the grid sizes are chosen to be the nearest power of 2. E.g., PMEGridSizeX 128 PMEGridSizeY 128 PMEGridSizeZ 128

3.10 Constraints and Restraints
shake {on|off} {shaketolerance}: Impose constraints on hydrogen bonds, including tip3 water. The 'shaketolerance' parameter determines tolerance within which the constraints are imposed.

hfac: Factor to multiply hydrogen masses(eg. 2). This parameter is used only when shake is turned off.

consharm {SEGID0 SEGID1}: Impose harmonic restraints between two groups of atoms described by segment id's SEGID0 and SEGID1. The segment ids are the last but one column in the input CHARMM '.crd' file.

consharmfc: Force constant for harmonic restraints (default value 1e-2)

consharmdist {x y z}: Distance of restraints between the group of atoms, specified by (x, y, z) as a vector.

consfix {SEGID0 SEGID1 SEGID2 ... SEGIDN}: Impose rigid constraints on the set of all atoms belonging to segments id's given by SEGID0, ... SEGIDN. The coordinates of the atoms are fixed rigidly in space.

Note: FEN ZI was tested with two tubes only - work in progress.

3.11 Constant Pressure
pref: Reference(external) pressure in atmospheres (defult value 1.0)

pmass: Piston mass of Andersen barostat (default value 1e-5)

Note: FEN ZI cosntat pressure is work in progress.

4 Running FEN ZI tests
FEN ZI must be run from the directory with the .in file.

To run FEN ZI: build/fenzi -d <devioce number> -f <input file>

Packaged with fenzi are several test cases (located in tests folder). Located in examples_fenzi are several input files for testing fenzi. Located in examples_charmm are input files for charmm that leads to the same results as the fenzi examples. Used for accuracy comparison.

Packaged Examples:

dmpc_small is a small membrane with explicit water moleucles. The water model is SPCFW. Fenzi must be compiled with "--tssize small"

../../../build/fenzi -d 0 -f dmpc_small_runtime.in

dmpc_medium_tip3 is a medium membrane in explicit water molecules. The water model is TIP3. Fenzi must be compiled with "--tssize medium"

../../../build/fenzi -d 0 -f dmpc_medium_tip3_runtime.in

5 Contacts
The best contact path for any question on FEN ZI is by e-mail to fenzi@gcl.cis.udel.edu or send correspondence to:

FEN ZI Team Att.: Michela Taufer Global Computing Laboratory (GCL) 406 Smith Hall Newark DE 19711

6 References
{1} B. R. Brooks, R. E. Bruccoleri, B. D. Olafson, D. J. States, S. Swaminathan, M. Karplus, CHARMM: A Program for Macromolecular Energy, Minimization, and Dynamics Calculations., J. Comp. Chem. 4 (1983) 187-217.

{2} S. Bond, B. Leimkuhler, B. Laird, The Nos_e-Poincar_e Method for Constant Temperature Molecular Dynamics, J. Comput. Phys. 151 (1999) 114-134.

{3} H. C. Andersen, Molecular Dynamics Simulations at Constant Pressure and/or Temperature, J. Chem. Phys. 72 (1980) 2384-2393.

{4} M. C. Payne, M. P. Teter, D. C. Allan, T. A. Arias, J. D. Joannopoulos, Iterative minimization techniques for ab initio total-energy calculations: molecular dynamics and conjugate gradients, Rev. Mod. Phys. 64 (1992) 1045-1097.

{5} I. Stich, R. Car, M. Parrinello, S. Baroni, Conjugate gradient minimization of the energy functional: A new method for electronic structure calculation, Phys. Rev. B 39 (1989) 4997-5004.