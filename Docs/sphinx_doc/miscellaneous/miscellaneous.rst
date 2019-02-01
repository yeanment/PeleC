.. highlight:: rst

Miscellaneous Information
=========================

Here we list supplementary information applicable to PeleC.

.. include:: ./kgen.rst

Grow cells.
-----------

The number of grow cells necessary to support the calculations varies throughout the program and is controlled by multiple variables. In the routines do_mol_advance or do_sdc_iteration, the current state is supplied to the transport routines via Sborder which is defined and filled from State_Type.

nGrowTr is the number of grow cells necessary to support the diffusion transport operations and is hard coded in PeleC.cpp (currently nGrowTr=4). NUM_GROW is the number of grow cells necessary to support the hydro operations and is fetched from the fortran meth_params_module where it is hardcoded as NHYP = 4. 


There is some inconsistency where in the MOL variants of the time advance routines Sborder is filled using nGrowTr grow cells, whereas in the SDC routines the NUM_GROW value. In PeleC.cpp, Sborder is defined to have NUM_GROW cells when do_hydro is set and nGrowTr if do_hydro is false but do_diffuse is true.

A potential pitfall is that the EB geometry sparse data structures *must* be built with a consistent number of grow cells. These are built from the MultiFab vfrac, defined in PeleC.cpp with NUM_GROW grow cells. The sparse structures are built in PeleC::initialize_eb2_structs; with a box that covers the grown fabs in the first pass to fill the per cut-cell ebg structures and a box grown by nGrowTr in the second pass to fill flux interpolation stencils.


A potentially bigger pitfall is that the EB MOL routines (see pc_hyp_mol_flux) in Hyp_pele_MOL_3d.F90 expect to operate on a box given by the validregion +/- nextra grow cells. The size of the faced based data structure is assumed to be big enough, and comes from amrex::suroundingNodes(cbox,d). So if cbox isn't big enough, bad things will happen.

To summarize, changing  the number of grow cells needs to be done three places:

NYP in meth_params_module
nGrowTr in PeleC.cpp
nextra in Hyp_pele_MOL_3d.F90

Missing one of these or inconsistencies may or may not be picked up by the compiler... YMMV.



