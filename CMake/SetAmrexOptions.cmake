set(USE_XSDK_DEFAULTS OFF)
set(AMReX_SPACEDIM "${PELE_DIM}" CACHE STRING "Number of physical dimensions" FORCE)
set(AMReX_MPI ${PELE_ENABLE_MPI})
set(AMReX_OMP ${PELE_ENABLE_OPENMP})
set(AMReX_EB ON)
set(AMReX_PARTICLES ${PELE_ENABLE_PARTICLES})
set(AMReX_PLOTFILE_TOOLS ${PELE_ENABLE_FCOMPARE})
set(AMReX_SUNDIALS OFF)
set(AMReX_FORTRAN OFF)
set(AMReX_FORTRAN_INTERFACES OFF)
set(AMReX_PIC OFF)
set(AMReX_PRECISION "${PELE_PRECISION}" CACHE STRING "Floating point precision" FORCE)
set(AMReX_LINEAR_SOLVERS OFF)
set(AMReX_AMRDATA OFF)
set(AMReX_ASCENT ${PELE_ENABLE_ASCENT})
set(AMReX_SENSEI OFF)
set(AMReX_CONDUIT ${PELE_ENABLE_ASCENT})
set(AMReX_HYPRE OFF)
set(AMReX_FPE OFF)
set(AMReX_ASSERTIONS OFF)
set(AMReX_BASE_PROFILE OFF)
set(AMReX_TINY_PROFILE ${PELE_ENABLE_TINY_PROFILE})
set(AMReX_TRACE_PROFILE OFF)
set(AMReX_MEM_PROFILE OFF)
set(AMReX_COMM_PROFILE OFF)
set(AMReX_BACKTRACE OFF)
set(AMReX_PROFPARSER OFF)
set(AMReX_ACC OFF)
set(AMReX_INSTALL ON)
set(AMReX_HDF5 ${PELE_ENABLE_HDF5})
set(AMReX_HDF5_ZFP ${PELE_ENABLE_HDF5_ZFP})

if(PELE_ENABLE_CUDA)
  set(AMReX_GPU_BACKEND CUDA CACHE STRING "AMReX GPU type" FORCE)
elseif(PELE_ENABLE_HIP)
  set(AMReX_GPU_BACKEND HIP CACHE STRING "AMReX GPU type" FORCE)
elseif(PELE_ENABLE_SYCL)
  set(AMReX_GPU_BACKEND SYCL CACHE STRING "AMReX GPU type" FORCE)
  set(AMReX_SYCL_ONEDPL ${PELE_ENABLE_SYCL})
else()
  set(AMReX_GPU_BACKEND NONE CACHE STRING "AMReX GPU type" FORCE)
endif()
