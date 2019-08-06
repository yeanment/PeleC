#include "PeleC_NSCBC_2D.H"
#include <PeleC.H> 

void impose_NSCBC(Box box, const amrex::Array4<amrex::Real> &u, 
                  const amrex::Array4<amrex::Real> &q, 
                  const amrex::Array4<amrex::Real> &qaux, 
                  const amrex::Array4<const int> x_bcMask, 
                  const amrex::Array4<const int> y_bcMask, 
                  const int nscbc_isAnyPerio, const int nscbc_perio, 
                  const amrex::Real time, const amrex::Real *delta, 
                  const amrex::Real dt) 
{
        const amrex::Real *problo = geom.ProbLo(); 
        const amrex::Real *probhi = geom.ProbHi(); 
        Box dom = geom.Domain(); 
        const int* domlo = dom.loVect(); 
        const int* domhi = dom.hiVect(); 
        const int* qlo = box.loVect(); 
        const int* qhi = box.hiVect(); 
        if(domlo[0] < qlo[0] && domlo[1] < qlo[1] && domhi[0] > qhi[0] && domhi[1] > qhi[1]) 
            return; 

        const int domlox = domlo[0]; 
        const int domloy = domlo[1]; 
        const int domhix = domhi[0]; 
        const int domhiy = domhi[1]; 
        const amrex::Real dx = delta[0]; 
        const amrex::Real dy = delta[1]; 

        if(nscbc_isAnyPerio==0){ 
                amrex::ParallelFor(box, 
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        if((i == domhix || i == domlox) && (j == domhiy || j == domloy))
                        {
                            int x_isgn, x_id_Mask, test_keyword_x, y_isgn, y_id_Mask, 
                                test_keyword_y; 
                            if(i == domhix){
                                x_isgn = -1; 
                                x_idx_Mask = i+1;   
                                test_keyword_x = physbc_hix; 
                            }else{
                                x_isgn = 1; 
                                x_idx_Mask = i;   
                                test_keyword_x = physbc_lox; 
                            }
                             if(j == domhiy){
                                y_isgn = -1; 
                                y_idx_Mask = j+1;   
                                test_keyword_y = physbc_hiy; 
                            }else{
                                y_isgn = 1; 
                                y_idx_Mask = j;   
                                test_keyword_y = physbc_loy; 
                            }
                           amrex::Real x = (i + 0.5e0)*dx; 
                           amrex::Real y = (j + 0.5e0)*dy; 
                            
                           PeleC_normal_deriv(i, j, k, 0, x_isgn, dx, dpdx, dudx, dvdx, drhodx, q); 
                           PeleC_normal_deriv(i, j, k, 1, y_isgn, dy, dpdy, dudy, dvdy, drhody, q); 
                           PeleC_compute_trans_terms(i,j,k,0,Tx,dpdy,dudy,dvdy,drhody,q, qaux); 
                           PeleC_compute_trans_terms(i,j,k,1,Ty,dpdx,dudx,dvdx,drhodx,q, qaux); 

/* TODO create interface for this
                           if(test_keyword_x==UserBC) 
                                bcnormal(args....)
*/                        
//                            else
                                x_bc_type = test_keyword_x; 
                            x_bcMask(i,j,k) = x_bc_type; 
/* TODO create interface for this
                           if(test_keyword_y==UserBC) 
                                bcnormal(args....)
*/                        
//                            else
                                y_bc_type = test_keyword_y; 
                            y_bcMask(i,j,k) = y_bc_type; 
                            PeleC_compute_waves(i,j,k,0,x_isgn, x_bc_type, x_bc_params, x_bc_target, 
                                                Tx, Lx, dpdx, dudx, dvdx, drhodx, q, qaux); 
                            PeleC_compute_waves(i,j,k,1,y_isgn, y_bc_type, y_bc_params, y_bc_target, 
                                                Ty, Ly, dpdy, dudy, dvdy, drhody, q, qaux); 
                            PeleC_update_gc(i, j, k, x_bc_type, 0, x_isgn, dx, domlox, domhix, 
                                            Lx, u, q, qaux); 
                            PeleC_update_gc(i, j, k, y_bc_type, 1, y_isgn, dy, domloy, domhiy, 
                                            Ly, u, q, qaux); 
                           

                        }
                    });//End Launch 
        }//End isAnyPerio 
        amrex::ParallelFor(box, 
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                //Low X 
                if(i == domlox){
                    amrex::Real  x = (i + 0.5e0)*dx; 
                    amrex::Real  y = (j + 0.5e0)*dy; 

                    //normal deriv along x
                    PeleC_normal_deriv(i, j, k, 0, 1, dx, dpdx, dudx, dvdx, drhodx, q); 
                    //Tangential to x, derivative along y 
                    PeleC_tang_deriv(i, j, k, 0, dy, dpdy, dudy, dvdy, drhody, q);
                    //Compute Transverse Terms 
                    PeleC_compute_trans_terms(i,j,k,0,Tx, dpdy, dudy, dvdy, drhody, q, qaux); 
                    // Call BC normal; 
                    // bcnormal(args....); 
                    x_bcMask(i,j,k) = x_bc_type; 
                    //Compute the LODI System 
                    PeleC_compute_waves(i,j,k,0,1, x_bc_type, bc_params, bc_target, Tx, Lx,
                                        dpdx, dudx, dvdx, drhodx, q, qaux); 
                    //Compute Ghost Cells using LODI system 
                    PeleC_upgdate_gc(i, j, k, x_bc_type, 0, 1, dx, domlox, domhix, Lx, u, q, qaux); 
                }
                //high X 
                if(i == domhix){
                    amrex::Real  x = (i + 0.5e0)*dx; 
                    amrex::Real  y = (j + 0.5e0)*dy; 

                    //normal deriv along x
                    PeleC_normal_deriv(i, j, k, 0, -1, dx, dpdx, dudx, dvdx, drhodx, q); 
                    //Tangential to x, derivative along y 
                    PeleC_tang_deriv(i, j, k, 0, dy, dpdy, dudy, dvdy, drhody, q);
                    //Compute Transverse Terms 
                    PeleC_compute_trans_terms(i,j,k,0,Tx, dpdy, dudy, dvdy, drhody, q, qaux); 
                    // Call BC normal; 
                    // bcnormal(args....); 
                    x_bcMask(i,j,k) = x_bc_type; 
                    //Compute the LODI System 
                    PeleC_compute_waves(i,j,k,0,-1, x_bc_type, bc_params, bc_target, Tx, Lx,
                                        dpdx, dudx, dvdx, drhodx, q, qaux); 
                    //Compute Ghost Cells using LODI system 
                    PeleC_upgdate_gc(i, j, k, x_bc_type, 0, -1, dx, domlox, domhix, Lx, u, q, qaux); 
                }
                //low Y 
                if(j == domloy){ 
                    amrex::Real  x = (i + 0.5e0)*dx; 
                    amrex::Real  y = (j + 0.5e0)*dy; 

                    //normal deriv along y
                    PeleC_normal_deriv(i, j, k, 1, 1, dy, dpdy, dudy, dvdy, drhody, q); 
                    //Tangential to y, derivative along x 
                    PeleC_tang_deriv(i, j, k, 1, dx, dpdx, dudx, dvdx, drhodx, q);
                    //Compute Transverse Terms 
                    PeleC_compute_trans_terms(i,j,k,1,Ty, dpdx, dudx, dvdx, drhodx, q, qaux); 
                    // Call BC normal; 
                    // bcnormal(args....); 
                    y_bcMask(i,j,k) = y_bc_type; 
                    //Compute the LODI System 
                    PeleC_compute_waves(i,j,k,1,1, y_bc_type, bc_params, bc_target, Ty, Ly,
                                        dpdy, dudy, dvdy, drhody, q, qaux); 
                    //Compute Ghost Cells using LODI system 
                    PeleC_upgdate_gc(i, j, k, y_bc_type, 1, 1, dy, domloy, domhiy, Ly, u, q, qaux); 

                }
                //high Y 
                if(j == domhiy){
                    amrex::Real  x = (i + 0.5e0)*dx; 
                    amrex::Real  y = (j + 0.5e0)*dy; 

                    //normal deriv along y
                    PeleC_normal_deriv(i, j, k, 1,-1, dy, dpdy, dudy, dvdy, drhody, q); 
                    //Tangential to y, derivative along x 
                    PeleC_tang_deriv(i, j, k, 1, dx, dpdx, dudx, dvdx, drhodx, q);
                    //Compute Transverse Terms 
                    PeleC_compute_trans_terms(i,j,k,1,Ty, dpdx, dudx, dvdx, drhodx, q, qaux); 
                    // Call BC normal; 
                    // bcnormal(args....); 
                    y_bcMask(i,j,k) = y_bc_type; 
                    //Compute the LODI System 
                    PeleC_compute_waves(i,j,k,1,-1, y_bc_type, bc_params, bc_target, Ty, Ly,
                                        dpdy, dudy, dvdy, drhody, q, qaux); 
                    //Compute Ghost Cells using LODI system 
                    PeleC_upgdate_gc(i, j, k, y_bc_type, 1, -1, dy, domloy, domhiy, Ly, u, q, qaux); 
                }                
            }); 
}
