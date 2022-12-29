#include "kinsolma.H"

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

namespace pele {
namespace kinsolma {

/* -----------------------------------------------------------------------------
 * Main program
 * ---------------------------------------------------------------------------*/
int kinsol_getma(realtype& Ma1, realtype& Ma2, const int& func_sol,
                 const realtype gamma, const realtype pre_ratio)
{
  /* User data */
  UserData data = NULL;
  data = (UserData)malloc(sizeof *data);
  data->gamma = gamma; // 1.4;
  data->pre_ratio = pre_ratio; // 10.;

  /* Problem data */
  N_Vector  u       = NULL;
  N_Vector  scale   = NULL;
  SUNMatrix J       = NULL;
  SUNLinearSolver LS= NULL;
  // int       func_sol= 1;           /* default get_Ma  */ 
  booleantype debug = SUNFALSE;     /* no debug output */ 
  realtype* udata   = NULL;        /* access init val */
  // realtype  Ma1     = TEN;         /* Ma_1 > 1        */
  // realtype  Ma2     = PTONE;       /* Ma_2 < 1        */
  Ma1               = TEN;         /* Ma_1 > 1        */
  Ma2               = PTONE;       /* Ma_2 < 1        */
  

  /* Solver options */
  /* glstr = KIN_NONE KIN_LINESEARCH KIN_FP KIN_PICARD*/
  int       glstr   = KIN_NONE;    /* default solver  */
  long int  maa     = 0;           /* no acceleration */
  long int  mxiter  = 1000;        /* max iters       */
  realtype  damping = RCONST(1.0); /* no damping      */
  realtype fnormtol = FTOL;        /* func norm tol   */ // SQRT(UNIT_ROUNDOFF);
  realtype scsteptol= STOL;        /* scaled step tol */

  int       retval  = 0;           /* retval of funcs */
  void*     kmem;

  /* --------------------------------------
   * Create vectors for solution and scales
   * -------------------------------------- */

  /* Create serial vectors of length NEQ */
  u = N_VNew_Serial(NEQ, *amrex::sundials::The_Sundials_Context());
  if (check_retval((void *)u, "N_VNew_Serial", 0)) return(1);

  scale = N_VNew_Serial(NEQ, *amrex::sundials::The_Sundials_Context());
  if (check_retval((void *)scale, "N_VNew_Serial", 0)) return(1);
  N_VConst(ONE, scale); /* no scaling */

  /* Create dense SUNMatrix */
  J = SUNDenseMatrix(NEQ, NEQ, *amrex::sundials::The_Sundials_Context());
  if(check_retval((void *)J, "SUNDenseMatrix", 0)) return(1);

  /* Create dense SUNLinearSolver object */
  LS = SUNLinSol_Dense(u, J, *amrex::sundials::The_Sundials_Context());
  if(check_retval((void *)LS, "SUNLinSol_Dense", 0)) return(1);
  
  /* -----------------------------------------
   * Initialize and allocate memory for KINSOL
   * ----------------------------------------- */

  kmem = KINCreate(*amrex::sundials::The_Sundials_Context());
  if (check_retval((void *)kmem, "KINCreate", 0)) return(1);

  if (glstr == KIN_FP) {
    if (func_sol == 0) {
      retval = KINInit(kmem, func_critical_Ma_fp, u);
    } else if (func_sol == 1) {
      retval = KINInit(kmem, func_get_Ma_fp, u);
    } else {
      fprintf(stderr, "\nERROR: KINInit() failed, not supported func_sol\n\n");
      return (1);
    }
    if (check_retval(&retval, "KINInit", 1)) return(1);
  } else if (glstr == KIN_NONE || glstr == KIN_LINESEARCH || glstr == KIN_PICARD) {
    if (func_sol == 0) {
      retval = KINInit(kmem, func_critical_Ma, u);
      if (check_retval(&retval, "KINInit", 1)) return(1);

      /* Attach the matrix and linear solver to KINSOL */
      retval = KINSetLinearSolver(kmem, LS, J);
      if(check_retval(&retval, "KINSetLinearSolver", 1)) return(1);

      /* Set the Jacobian function */
      retval = KINSetJacFn(kmem, func_critical_Ma_jac);
      if (check_retval(&retval, "KINSetJacFn", 1)) return(1);

      // /* Set the Jacobian vector product function */
      // retval = KINSetJacTimesVecFn(kmem, func_critical_Ma_jactimes);
      // if (check_retval(&retval, "KINSetJacTimesVecFn", 1)) return(1);
    } else if (func_sol == 1) {
      retval = KINInit(kmem, func_get_Ma, u);
      if (check_retval(&retval, "KINInit", 1)) return(1);

      /* Attach the matrix and linear solver to KINSOL */
      retval = KINSetLinearSolver(kmem, LS, J);
      if(check_retval(&retval, "KINSetLinearSolver", 1)) return(1);

      /* Set the Jacobian function */
      retval = KINSetJacFn(kmem, func_get_Ma_jac);
      if (check_retval(&retval, "KINSetJacFn", 1)) return(1);

      // /* Set the Jacobian vector product function */
      // retval = KINSetJacTimesVecFn(kmem, func_get_Ma_jactimes);
      // if (check_retval(&retval, "KINSetJacTimesVecFn", 1)) return(1);
    } else {
      fprintf(stderr, "\nERROR: KINInit() failed, not supported func_sol\n\n");
      return (1);
    }
  } else {
    fprintf(stderr, "\nERROR: KINInit() failed, not supported glstr\n\n");
    return (1);
  }

  retval = KINSetUserData(kmem, data);
  if (check_retval(&retval, "KINSetUserData", 1)) return(1);

  /* -------------------
   * Set optional inputs
   * ------------------- */

  /* Set number of prior residuals used in Anderson acceleration */
  retval = KINSetMAA(kmem, maa);
  if (check_retval(&retval, "KINSetMAA", 1)) return(1);

  /* Set maximum number of iterations */
  retval = KINSetNumMaxIters(kmem, mxiter);
  if (check_retval(&retval, "KINSetNumMaxItersFuncNormTol", 1)) return(1);

  /* Set Anderson acceleration damping parameter */
  retval = KINSetDampingAA(kmem, damping);
  if (check_retval(&retval, "KINSetDampingAA", 1)) return(1);

  /* Specify stopping tolerance based on residual */
  retval = KINSetFuncNormTol(kmem, fnormtol);
  if (check_retval(&retval, "KINSetFuncNormTol", 1)) return(1);
  retval = KINSetScaledStepTol(kmem, scsteptol);
  if (check_retval(&retval, "KINSetScaledStepTol", 1)) return(1);

  /* --------------------------------------------
   * Initial guess & call KINSol to solve problem
   * -------------------------------------------- */

  /* Get vector data array */
  udata = N_VGetArrayPointer(u);
  if (check_retval((void *)data, "N_VGetArrayPointer", 0)) return(1);

  udata[0] =  Ma1;

  /* Call main solver */
  retval = KINSol(kmem,         /* KINSol memory block */
                  u,            /* initial guess on input; solution vector */
                  glstr,        /* global strategy choice */
                  scale,        /* scaling vector, for the variable cc */
                  scale);       /* scaling vector for function values fval */
  if (check_retval(&retval, "KINSol", 1)) return(1);

  Ma1 = udata[0];

  /* Print solver statistics */
  if (debug) {
    printf("Solver statistics of Ma1:\n");
    print_solver_stats(kmem);
  }

  if (func_sol == 1) {
    Ma2 = Ma1;
  } else {
    udata[0] =  Ma2;

    /* Call main solver */
    retval = KINSol(kmem,         /* KINSol memory block */
                    u,            /* initial guess on input; solution vector */
                    glstr,        /* global strategy choice */
                    scale,        /* scaling vector, for the variable cc */
                    scale);       /* scaling vector for function values fval */

    Ma2 = udata[0];

    /* Print solver statistics */
    if (debug) {
      printf("Solver statistics of Ma2:\n");
      print_solver_stats(kmem);
    }
  }

  /* Print the solution */
  if (debug) {
    printf("Computed solution:\n");
    printf("    Ma = %" GSYM "/%" GSYM "\n", Ma1, Ma2);
  }
  
  /* -----------
   * Free memory
   * ----------- */

  N_VDestroy(u);
  N_VDestroy(scale);
  KINFree(&kmem);
  SUNLinSolFree(LS);
  SUNMatDestroy(J);

  return(retval);
}

/* -----------------------------------------------------------------------------
 * Nonlinear system for the critical Mach number
 *
 * pre_ratio = area_ratio
 * pre_ratio * Ma = (2/(gamma+1)*(1+(gamma-1)*Ma*Ma/2))^((gamma+1)/(2*(gamma-1)))
 *
 * Nonlinear fixed point function
 *
 * g1(Ma) = (2/(gamma+1)*(1+(gamma-1)*Ma*Ma/2))^((gamma+1)/(2*(gamma-1)))/pre_ratio
 *
 * Nonlinear function
 *
 * f1(Ma) = (2/(gamma+1)*(1+(gamma-1)*Ma*Ma/2))^((gamma+1)/(2*(gamma-1)))
 *        - pre_ratio*Ma
 *
 * f'(Ma) = (2/(gamma+1)*(1+(gamma-1)*Ma*Ma/2))^((3-gamma)/(2*(gamma-1)))*Ma
 *        - pre_ratio
 * ---------------------------------------------------------------------------*/
static int func_critical_Ma_fp(N_Vector u, N_Vector g, void* user_data)
{
  realtype* udata = NULL;
  realtype* gdata = NULL;
  realtype  Ma;

  UserData data;
  realtype pre_ratio, gamma;
  data = (UserData)user_data;
  pre_ratio = data->pre_ratio;
  gamma = data->gamma;

  /* Get vector data arrays */
  udata = N_VGetArrayPointer(u);
  if (check_retval((void*)udata, "N_VGetArrayPointer", 0)) return(-1);

  gdata = N_VGetArrayPointer(g);
  if (check_retval((void*)gdata, "N_VGetArrayPointer", 0)) return(-1);

  Ma = udata[0];

  gdata[0] = POW((2./(gamma+1.)*(1.+(gamma-1.)*Ma*Ma/2.)),
                 ((gamma+1.)/(2.*(gamma-1.)))
                )/pre_ratio;

  return(0);
}

static int func_critical_Ma(N_Vector u, N_Vector f, void* user_data)
{
  realtype* udata = NULL;
  realtype* fdata = NULL;
  realtype  Ma;

  UserData data;
  data = (UserData)user_data;
  realtype pre_ratio = data->pre_ratio;
  realtype gamma = data->gamma;

  /* Get vector data arrays */
  udata = N_VGetArrayPointer(u);
  if (check_retval((void*)udata, "N_VGetArrayPointer", 0)) return(-1);

  fdata = N_VGetArrayPointer(f);
  if (check_retval((void*)fdata, "N_VGetArrayPointer", 0)) return(-1);

  Ma = udata[0];

  fdata[0] = POW((2./(gamma+1.)*(1.+(gamma-1.)*Ma*Ma/2.)),
                 ((gamma+1.)/(2.*(gamma-1.))))
           - pre_ratio * Ma;

  return(0);
}

static int func_critical_Ma_jac(N_Vector u, N_Vector f, SUNMatrix J,
                      void *user_data, N_Vector tmp1, N_Vector tmp2)
{

  realtype* udata = NULL;
  realtype* fdata = NULL;
  realtype  Ma;

  UserData data;
  data = (UserData)user_data;
  realtype pre_ratio = data->pre_ratio;
  realtype gamma = data->gamma;

  udata = N_VGetArrayPointer(u);
  if (check_retval((void*)udata, "N_VGetArrayPointer", 0)) return(-1);

  Ma = udata[0];

  SM_ELEMENT_D(J,0,0) = POW((2./(gamma+1.)*(1.+(gamma-1.)*Ma*Ma/2.)),
                           ((3.-gamma)/(2.*(gamma-1.))))*Ma
                      - pre_ratio;

  return(0);
}

static int func_critical_Ma_jactimes(N_Vector v, N_Vector Jv, N_Vector u,
                                      booleantype *new_u, void *user_data)
{
  realtype* vdata = NULL;
  realtype* Jvdata = NULL;
  realtype* udata = NULL;
  realtype* fdata = NULL;
  realtype  Ma, dMa;

  UserData data;
  data = (UserData)user_data;
  realtype pre_ratio = data->pre_ratio;
  realtype gamma = data->gamma;

  udata = N_VGetArrayPointer(u);
  if (check_retval((void*)udata, "N_VGetArrayPointer", 0)) return(-1);

  Ma = udata[0];

  dMa = POW((2./(gamma+1.)*(1.+(gamma-1.)*Ma*Ma/2.)),
            ((3.-gamma)/(2.*(gamma-1.))))*Ma
      - pre_ratio;

  vdata  = N_VGetArrayPointer(v);
  if (check_retval((void*)vdata, "N_VGetArrayPointer", 0)) return(-1);

  Jvdata = N_VGetArrayPointer(Jv);
  if (check_retval((void*)vdata, "N_VGetArrayPointer", 0)) return(-1);

  Jvdata[0] = vdata[0] * dMa;

  return(0);
}

/* -----------------------------------------------------------------------------
 * Nonlinear system for the current Mach number
 *
 * pre_ratio = area_ratio * p_w/ p_0
 * pre_ratio * Ma = (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))
 *                 /sqrt(1+(gamma-1)*Ma*Ma/2))
 *
 * Nonlinear fixed point function
 *
 * g1(Ma) = (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))
 *        / sqrt(1+(gamma-1)*Ma*Ma/2)) / pre_ratio
 *
 * Nonlinear function
 *
 * f1(Ma) = (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))/sqrt(1+(gamma-1)*Ma*Ma/2))
 *        - pre_ratio*Ma
 *
 * f'(Ma) = (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))
 *        * (1-gamma) * pow(1+(gamma-1)*Ma*Ma/2), -1.5) *Ma / 2
 *        - pre_ratio
* ---------------------------------------------------------------------------*/
static int func_get_Ma_fp(N_Vector u, N_Vector g, void* user_data)
{
  realtype* udata = NULL;
  realtype* gdata = NULL;
  realtype  Ma;

  UserData data;
  realtype pre_ratio, gamma;
  data = (UserData)user_data;
  pre_ratio = data->pre_ratio;
  gamma = data->gamma;

  /* Get vector data arrays */
  udata = N_VGetArrayPointer(u);
  if (check_retval((void*)udata, "N_VGetArrayPointer", 0)) return(-1);

  gdata = N_VGetArrayPointer(g);
  if (check_retval((void*)gdata, "N_VGetArrayPointer", 0)) return(-1);

  Ma = udata[0];

  gdata[0] = POW((2./(gamma+1)), ((gamma+1.)/(2.*(gamma-1.))))
           / SQRT((1.+(gamma-1.)*Ma*Ma/2.)) / pre_ratio;

  return(0);
}

static int func_get_Ma(N_Vector u, N_Vector f, void* user_data)
{
  realtype* udata = NULL;
  realtype* fdata = NULL;
  realtype  Ma;

  UserData data;
  data = (UserData)user_data;
  realtype pre_ratio = data->pre_ratio;
  realtype gamma = data->gamma;

  /* Get vector data arrays */
  udata = N_VGetArrayPointer(u);
  if (check_retval((void*)udata, "N_VGetArrayPointer", 0)) return(-1);

  fdata = N_VGetArrayPointer(f);
  if (check_retval((void*)fdata, "N_VGetArrayPointer", 0)) return(-1);

  Ma = udata[0];

  fdata[0] = POW((2./(gamma+1)), ((gamma+1.)/(2.*(gamma-1.))))
           / SQRT((1.+(gamma-1.)*Ma*Ma/2.))
           - pre_ratio * Ma;

  return(0);
}

static int func_get_Ma_jac(N_Vector u, N_Vector f, SUNMatrix J,
                      void *user_data, N_Vector tmp1, N_Vector tmp2)
{

  realtype* udata = NULL;
  realtype* fdata = NULL;
  realtype  Ma;

  UserData data;
  data = (UserData)user_data;
  realtype pre_ratio = data->pre_ratio;
  realtype gamma = data->gamma;

  udata = N_VGetArrayPointer(u);
  if (check_retval((void*)udata, "N_VGetArrayPointer", 0)) return(-1);

  Ma = udata[0];

  SM_ELEMENT_D(J,0,0) = POW((2./(gamma+1)), ((gamma+1.)/(2.*(gamma-1.))))
                      * (1.-gamma)/2 * POW((1.+(gamma-1.)*Ma*Ma/2.), 1.5) * Ma
                      - pre_ratio;
  return(0);
}

static int func_get_Ma_jactimes(N_Vector v, N_Vector Jv, N_Vector u,
                                      booleantype *new_u, void *user_data)
{
  realtype* vdata = NULL;
  realtype* Jvdata = NULL;
  realtype* udata = NULL;
  realtype* fdata = NULL;
  realtype  Ma, dMa;

  UserData data;
  data = (UserData)user_data;
  realtype pre_ratio = data->pre_ratio;
  realtype gamma = data->gamma;

  udata = N_VGetArrayPointer(u);
  if (check_retval((void*)udata, "N_VGetArrayPointer", 0)) return(-1);

  Ma = udata[0];

  dMa = POW((2./(gamma+1)), ((gamma+1.)/(2.*(gamma-1.))))
      * (1.-gamma)/2 * POW((1.+(gamma-1.)*Ma*Ma/2.), 1.5) * Ma
      - pre_ratio;

  vdata  = N_VGetArrayPointer(v);
  if (check_retval((void*)vdata, "N_VGetArrayPointer", 0)) return(-1);

  Jvdata = N_VGetArrayPointer(Jv);
  if (check_retval((void*)vdata, "N_VGetArrayPointer", 0)) return(-1);

  Jvdata[0] = vdata[0] * dMa;

  return(0);
}

/* -----------------------------------------------------------------------------
 * Check the solution of the nonlinear system and return PASS or FAIL
 * ---------------------------------------------------------------------------*/
static int check_ans(N_Vector u, realtype tol)
{
  realtype* data = NULL;
  realtype  ex, ey, ez;

  /* Get vector data array */
  data = N_VGetArrayPointer(u);
  if (check_retval((void *)data, "N_VGetArrayPointer", 0)) return(1);

  /* print the solution */
  printf("Computed solution:\n");
  printf("    Ma = %" GSYM "\n", data[0]);

  /* solution error */
  ex = ABS(data[0] - MA1TRUE);
  ey = ABS(data[0] - MA2TRUE);

  /* print the solution error */
  printf("Solution error:\n");
  printf("    ex = %" GSYM "\n", ex);
  printf("    ey = %" GSYM "\n", ey);

  tol *= TEN;
  if (ex > tol && ey > tol) {
    printf("FAIL\n");
    return(1);
  }

  printf("PASS\n");
  return(0);
}

/* -----------------------------------------------------------------------------
 * Check function return value
 *   opt == 0 check if returned NULL pointer
 *   opt == 1 check if returned a non-zero value
 * ---------------------------------------------------------------------------*/
static int check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *errflag;

  /* Check if the function returned a NULL pointer -- no memory allocated */
  if (opt == 0) {
    if (returnvalue == NULL) {
      fprintf(stderr, "\nERROR: %s() failed -- returned NULL\n\n", funcname);
      return(1);
    } else {
      return(0);
    }
  }

  /* Check if the function returned an non-zero value -- internal failure */
  if (opt == 1) {
    errflag = (int *) returnvalue;
    if (*errflag != 0) {
      fprintf(stderr, "\nERROR: %s() failed -- returned %d\n\n", funcname, *errflag);
      return(1);
    } else {
      return(0);
    }
  }

  /* if we make it here then opt was not 0 or 1 */
  fprintf(stderr, "\nERROR: check_retval failed -- Invalid opt value\n\n");
  return(1);
}

/* -----------------------------------------------------------------------------
 * Print solver final statistics
 * ---------------------------------------------------------------------------*/
static void print_solver_stats(void *kmem)
{
  long int nni, nfe, nli, npe, nps, ncfl, nfeLS, njvevals;
  long int lenrw, leniw, lenrwLS, leniwLS;
  int retval;

  /* Main solver statistics */

  retval = KINGetNumNonlinSolvIters(kmem, &nni);
  check_retval(&retval, "KINGetNumNonlinSolvIters", 1);
  retval = KINGetNumFuncEvals(kmem, &nfe);
  check_retval(&retval, "KINGetNumFuncEvals", 1);

  /* Linear solver statistics */

  retval = KINGetNumLinIters(kmem, &nli);
  check_retval(&retval, "KINGetNumLinIters", 1);
  retval = KINGetNumLinFuncEvals(kmem, &nfeLS);
  check_retval(&retval, "KINGetNumLinFuncEvals", 1);
  retval = KINGetNumLinConvFails(kmem, &ncfl);
  check_retval(&retval, "KINGetNumLinConvFails", 1);
  retval = KINGetNumJtimesEvals(kmem, &njvevals);
  check_retval(&retval, "KINGetNumJtimesEvals", 1);
  retval = KINGetNumPrecEvals(kmem, &npe);
  check_retval(&retval, "KINGetNumPrecEvals", 1);
  retval = KINGetNumPrecSolves(kmem, &nps);
  check_retval(&retval, "KINGetNumPrecSolves", 1);

  /* Main solver workspace size */

  retval = KINGetWorkSpace(kmem, &lenrw, &leniw);
  check_retval(&retval, "KINGetWorkSpace", 1);

  /* Linear solver workspace size */

  retval = KINGetLinWorkSpace(kmem, &lenrwLS, &leniwLS);
  check_retval(&retval, "KINGetLinWorkSpace", 1);

  printf("Final Statistics.. \n");
  printf("nni = %6ld  nli   = %6ld  ncfl = %6ld\n", nni, nli, ncfl);
  printf("nfe = %6ld  nfeLS = %6ld  njt  = %6ld\n", nfe, nfeLS, njvevals);
  printf("npe = %6ld  nps   = %6ld\n", npe, nps);
  printf("\n");
  printf("lenrw   = %6ld  leniw   = %6ld\n", lenrw, leniw);
  printf("lenrwLS = %6ld  leniwLS = %6ld\n\n", lenrwLS, leniwLS);
}

} // namespace kinsol
} // namespace pele

#ifdef __cplusplus
}
#endif
