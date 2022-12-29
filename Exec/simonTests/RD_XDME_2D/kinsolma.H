#ifndef KINSOLMA_H
#define KINSOLMA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "AMReX_Sundials.H"

#include "kinsol/kinsol.h"             /* access to KINSOL func., consts. */
#include "nvector/nvector_serial.h"    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype */

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

namespace pele {
namespace kinsolma {

/* precision specific formatting macros */
#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

/* precision specific math function macros */
#if defined(SUNDIALS_DOUBLE_PRECISION)
#define ABS(x)    (fabs((x)))
#define SQRT(x)   (sqrt((x)))
#define EXP(x)    (exp((x)))
#define SIN(x)    (sin((x)))
#define COS(x)    (cos((x)))
#define POW(x, y) (pow((x), (y)))
#elif defined(SUNDIALS_SINGLE_PRECISION)
#define ABS(x)    (fabsf((x)))
#define SQRT(x)   (sqrtf((x)))
#define EXP(x)    (expf((x)))
#define SIN(x)    (sinf((x)))
#define COS(x)    (cosf((x)))
#define POW(x, y) (powf((x), (y)))
#elif defined(SUNDIALS_EXTENDED_PRECISION)
#define ABS(x)    (fabsl((x)))
#define SQRT(x)   (sqrtl((x)))
#define EXP(x)    (expl((x)))
#define SIN(x)    (sinl((x)))
#define COS(x)    (cosl((x)))
#define POW(x, y) (powl((x), (y)))
#endif

/* problem constants */
#define NEQ 1 /* number of equations */

#define FTOL   RCONST(1.e-10) /* function tolerance */
#define STOL   RCONST(1.e-10) /* step tolerance */

#define ZERO         RCONST(0.0)             /* real 0.0  */
#define PTONE        RCONST(0.1)             /* real 0.1  */
#define HALF         RCONST(0.5)             /* real 0.5  */
#define PTNINE       RCONST(0.9)             /* real 0.9  */
#define ONE          RCONST(1.0)             /* real 1.0  */
#define ONEPTZEROSIX RCONST(1.06)            /* real 1.06 */
#define ONEPTONE     RCONST(1.1)             /* real 1.1  */
#define THREE        RCONST(3.0)             /* real 3.0  */
#define FOUR         RCONST(4.0)             /* real 4.0  */
#define SIX          RCONST(6.0)             /* real 6.0  */
#define NINE         RCONST(9.0)             /* real 9.0  */
#define TEN          RCONST(10.0)            /* real 10.0 */
#define TWENTY       RCONST(20.0)            /* real 20.0 */
#define SIXTY        RCONST(60.0)            /* real 60.0 */
#define EIGHTYONE    RCONST(81.0)            /* real 81.0 */
#define PI           RCONST(3.1415926535898) /* real pi   */

/* analytic solution */
#define MA1TRUE 3.922551820933724
#define MA2TRUE 0.057987202922874

typedef struct {
  realtype gamma;
  realtype pre_ratio;
} *UserData;


/* Nonlinear fixed point function */
static int func_critical_Ma_fp(N_Vector u, N_Vector g, void* user_data);

static int func_critical_Ma(N_Vector u, N_Vector f, void *user_data);

static int func_critical_Ma_jac(N_Vector u, N_Vector f, SUNMatrix J,
                      void *user_data, N_Vector tmp1, N_Vector tmp2);

static int func_critical_Ma_jactimes(N_Vector v, N_Vector Jv, N_Vector u,
                                     booleantype *new_u, void *user_data);

static int func_get_Ma_fp(N_Vector u, N_Vector g, void* user_data);

static int func_get_Ma(N_Vector u, N_Vector f, void *user_data);

static int func_get_Ma_jac(N_Vector u, N_Vector f, SUNMatrix J,
                           void *user_data, N_Vector tmp1, N_Vector tmp2);

static int func_get_Ma_jactimes(N_Vector v, N_Vector Jv, N_Vector u,
                                booleantype *new_u, void *user_data);


/* Check function return values */
static int check_retval(void *returnvalue, const char *funcname, int opt);

/* Check the system solution */
static int check_ans(N_Vector u, realtype tol);

/* Print solver statistics */
static void print_solver_stats(void *kmem);


/* -----------------------------------------------------------------------------
 * Main solver
 * ---------------------------------------------------------------------------*/
int kinsol_getma(realtype& Ma1, realtype& Ma2, const int& func_sol=0,
                 const realtype gamma=1.4, const realtype pre_ratio=10.0);

} // namespace kinsol
} // namespace pele

#ifdef __cplusplus
}
#endif

#endif
