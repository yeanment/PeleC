#include "base_getrates.h"
#include <cmath>
#include <cassert>
#include <cstdlib>

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

static AMREX_GPU_DEVICE_MANAGED double molecular_masses[35] = {170.34102, 1.00797, 15.9994, 17.00737, 
  33.00677, 2.01594, 18.01534, 34.01474, 31.9988, 15.03506, 16.04303, 30.02649, 
  28.01055, 44.00995, 26.03824, 28.05418, 30.07012, 43.04561, 41.0733, 42.08127, 
  56.06473, 55.10039, 56.10836, 69.12748000000001, 70.13545000000001, 
  84.16254000000001, 98.18963000000001, 112.21672, 126.24381, 127.25178, 
  140.2709, 168.32508, 201.33185, 216.32328, 28.0134}; 

static AMREX_GPU_DEVICE_MANAGED double recip_molecular_masses[35] = {5.870576564587907e-03, 
  0.9920930186414277, 0.06250234383789392, 0.05879803873262004, 
  0.03029681486555637, 0.4960465093207139, 0.05550825019122593, 
  0.02939901936631002, 0.03125117191894696, 0.06651120780362699, 
  0.06233236489615739, 0.03330392596670473, 0.03570083414998991, 
  0.02272213442641948, 0.0384050534905585, 0.03564531203549703, 
  0.03325560390181349, 0.02323117270262868, 0.02434671672351625, 
  0.02376354135699802, 0.01783652574443861, 0.01814869186951308, 
  0.01782265601774851, 0.01446602711396394, 0.01425812481419881, 
  0.01188177067849901, 0.01018437486728486, 8.911328008874257e-03, 
  7.921180452332673e-03, 7.858436243485159e-03, 7.129062407099405e-03, 
  5.940885339249504e-03, 4.966924011277897e-03, 4.622710972207892e-03, 
  0.03569720205330306}; 


AMREX_GPU_HOST_DEVICE
void base_getrates(const double pressure, const double temperature, const double 
  avmolwt, const double *mass_frac, const double *diffusion, const double dt, 
  double *wdot) 
{
  
  const double PA = 1.013250e+06;
  const double R0 = 8.314510e+07;
  const double R0c = 1.9872155832;
  const double DLn10 = 2.3025850929940459e0;
  
  const double otc     = 1.0 / temperature;
  const double ortc    = 1.0 / (temperature * R0c);
  const double vlntemp = log(temperature);
  const double prt     = PA / (R0 * temperature);
  const double oprt    = 1.0 / prt;
  
  double cgspl[53];
  // Gibbs computation
  {
    const double &tk1 = temperature;
    double tklog = log(tk1);
    double tk2 = tk1 * tk1;
    double tk3 = tk1 * tk2;
    double tk4 = tk1 * tk3;
    double tk5 = tk1 * tk4;
    
    // Species NC12H26
    {
      if (tk1 > 1.391e+03)
      {
        cgspl[0] = 38.5095037*tk1*(1-tklog) + -0.0281775024*tk2 + 
          3.191553333333334e-06*tk3 + -2.466873849999999e-10*tk4 + 
          8.562207499999999e-15*tk5 + (-5.48843465e+04 - tk1*-172.670922); 
      }
      else
      {
        cgspl[0] = -2.62181594*tk1*(1-tklog) + -0.0736188555*tk2 + 
          1.573283785e-05*tk3 + -2.562010566666666e-09*tk4 + 2.01801115e-13*tk5 
          + (-4.00654253e+04 - tk1*50.0994626); 
      }
    }
    // Species H
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[1] = 2.50000001*tk1*(1-tklog) + 1.154214865e-11*tk2 + 
          -2.692699133333334e-15*tk3 + 3.945960291666666e-19*tk4 + 
          -2.490986785e-23*tk5 + (2.54736599e+04 - tk1*-0.446682914); 
      }
      else
      {
        cgspl[1] = 2.5*tk1*(1-tklog) + -3.526664095e-13*tk2 + 
          3.326532733333334e-16*tk3 + -1.917346933333333e-19*tk4 + 
          4.638661659999999e-23*tk5 + (2.54736599e+04 - tk1*-0.446682853); 
      }
    }
    // Species O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[2] = 2.56942078*tk1*(1-tklog) + 4.298705685e-05*tk2 + 
          -6.991409816666668e-09*tk3 + 8.348149916666664e-13*tk4 + 
          -6.141684549999998e-17*tk5 + (2.92175791e+04 - tk1*4.78433864); 
      }
      else
      {
        cgspl[2] = 3.1682671*tk1*(1-tklog) + 1.63965942e-03*tk2 + 
          -1.107177326666667e-06*tk3 + 5.106721866666665e-10*tk4 + 
          -1.056329855e-13*tk5 + (2.91222592e+04 - tk1*2.05193346); 
      }
    }
    // Species OH
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[3] = 2.86472886*tk1*(1-tklog) + -5.2825224e-04*tk2 + 
          4.318045966666668e-08*tk3 + -2.543488949999999e-12*tk4 + 
          6.659793799999999e-17*tk5 + (3.71885774e+03 - tk1*5.70164073); 
      }
      else
      {
        cgspl[3] = 4.12530561*tk1*(1-tklog) + 1.612724695e-03*tk2 + 
          -1.087941151666667e-06*tk3 + 4.832113691666665e-10*tk4 + 
          -1.031186895e-13*tk5 + (3.38153812e+03 - tk1*-0.69043296); 
      }
    }
    // Species HO2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[4] = 4.0172109*tk1*(1-tklog) + -1.119910065e-03*tk2 + 
          1.056096916666667e-07*tk3 + -9.520530833333331e-12*tk4 + 
          5.395426749999999e-16*tk5 + (111.856713 - tk1*3.78510215); 
      }
      else
      {
        cgspl[4] = 4.30179801*tk1*(1-tklog) + 2.374560255e-03*tk2 + 
          -3.526381516666667e-06*tk3 + 2.02303245e-09*tk4 + -4.64612562e-13*tk5 
          + (294.80804 - tk1*3.71666245); 
      }
    }
    // Species H2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[5] = 3.3372792*tk1*(1-tklog) + 2.470123655e-05*tk2 + 
          -8.324279633333334e-08*tk3 + 1.496386616666666e-11*tk4 + 
          -1.00127688e-15*tk5 + (-950.158922 - tk1*-3.20502331); 
      }
      else
      {
        cgspl[5] = 2.34433112*tk1*(1-tklog) + -3.990260375e-03*tk2 + 
          3.2463585e-06*tk3 + -1.67976745e-09*tk4 + 3.688058804999999e-13*tk5 + 
          (-917.935173 - tk1*0.683010238); 
      }
    }
    // Species H2O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[6] = 3.03399249*tk1*(1-tklog) + -1.08845902e-03*tk2 + 
          2.734541966666667e-08*tk3 + 8.086832249999998e-12*tk4 + 
          -8.410049599999998e-16*tk5 + (-3.00042971e+04 - tk1*4.9667701); 
      }
      else
      {
        cgspl[6] = 4.19864056*tk1*(1-tklog) + 1.01821705e-03*tk2 + 
          -1.086733685e-06*tk3 + 4.573308849999999e-10*tk4 + 
          -8.859890849999997e-14*tk5 + (-3.02937267e+04 - tk1*-0.849032208); 
      }
    }
    // Species H2O2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[7] = 4.16500285*tk1*(1-tklog) + -2.45415847e-03*tk2 + 
          3.168987083333334e-07*tk3 + -3.093216549999999e-11*tk4 + 
          1.439541525e-15*tk5 + (-1.78617877e+04 - tk1*2.91615662); 
      }
      else
      {
        cgspl[7] = 4.27611269*tk1*(1-tklog) + 2.714112085e-04*tk2 + 
          -2.788928350000001e-06*tk3 + 1.798090108333333e-09*tk4 + 
          -4.312271814999999e-13*tk5 + (-1.77025821e+04 - tk1*3.43505074); 
      }
    }
    // Species O2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[8] = 3.28253784*tk1*(1-tklog) + -7.4154377e-04*tk2 + 
          1.263277781666667e-07*tk3 + -1.745587958333333e-11*tk4 + 
          1.08358897e-15*tk5 + (-1.08845772e+03 - tk1*5.45323129); 
      }
      else
      {
        cgspl[8] = 3.78245636*tk1*(1-tklog) + 1.49836708e-03*tk2 + 
          -1.641217001666667e-06*tk3 + 8.067745908333333e-10*tk4 + 
          -1.621864185e-13*tk5 + (-1.06394356e+03 - tk1*3.65767573); 
      }
    }
    // Species CH2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[9] = 2.87410113*tk1*(1-tklog) + -1.82819646e-03*tk2 + 
          2.348243283333334e-07*tk3 + -2.168162908333333e-11*tk4 + 
          9.386378349999998e-16*tk5 + (4.6263604e+04 - tk1*6.17119324); 
      }
      else
      {
        cgspl[9] = 3.76267867*tk1*(1-tklog) + -4.844360715e-04*tk2 + 
          -4.658164016666667e-07*tk3 + 3.209092941666666e-10*tk4 + 
          -8.437085949999999e-14*tk5 + (4.60040401e+04 - tk1*1.56253185); 
      }
    }
    // Species CH2*
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[10] = 2.29203842*tk1*(1-tklog) + -2.327943185e-03*tk2 + 
          3.353199116666667e-07*tk3 + -3.482549999999999e-11*tk4 + 
          1.698581825e-15*tk5 + (5.09259997e+04 - tk1*8.62650169); 
      }
      else
      {
        cgspl[10] = 4.19860411*tk1*(1-tklog) + 1.183307095e-03*tk2 + 
          -1.372160366666667e-06*tk3 + 5.573466508333332e-10*tk4 + 
          -9.715736849999998e-14*tk5 + (5.04968163e+04 - tk1*-0.769118967); 
      }
    }
    // Species CH3
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[11] = 2.28571772*tk1*(1-tklog) + -3.619950185e-03*tk2 + 
          4.978572466666668e-07*tk3 + -4.964038699999999e-11*tk4 + 
          2.33577197e-15*tk5 + (1.67755843e+04 - tk1*8.48007179); 
      }
      else
      {
        cgspl[11] = 3.6735904*tk1*(1-tklog) + -1.005475875e-03*tk2 + 
          -9.550364266666668e-07*tk3 + 5.725978541666665e-10*tk4 + 
          -1.27192867e-13*tk5 + (1.64449988e+04 - tk1*1.60456433); 
      }
    }
    // Species CH4
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[12] = 0.074851495*tk1*(1-tklog) + -6.69547335e-03*tk2 + 
          9.554763483333335e-07*tk3 + -1.019104458333333e-10*tk4 + 
          5.090761499999999e-15*tk5 + (-9.468344590000001e+03 - tk1*18.437318); 
      }
      else
      {
        cgspl[12] = 5.14987613*tk1*(1-tklog) + 6.8354894e-03*tk2 + 
          -8.19667665e-06*tk3 + 4.039525216666666e-09*tk4 + 
          -8.334697799999998e-13*tk5 + (-1.02466476e+04 - tk1*-4.64130376); 
      }
    }
    // Species HCO
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[13] = 2.77217438*tk1*(1-tklog) + -2.47847763e-03*tk2 + 
          4.140760216666667e-07*tk3 + -4.909681483333332e-11*tk4 + 
          2.667543554999999e-15*tk5 + (4.01191815e+03 - tk1*9.79834492); 
      }
      else
      {
        cgspl[13] = 4.22118584*tk1*(1-tklog) + 1.62196266e-03*tk2 + 
          -2.296657433333334e-06*tk3 + 1.109534108333333e-09*tk4 + 
          -2.168844324999999e-13*tk5 + (3.83956496e+03 - tk1*3.39437243); 
      }
    }
    // Species CH2O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[14] = 1.76069008*tk1*(1-tklog) + -4.60000041e-03*tk2 + 
          7.370980216666667e-07*tk3 + -8.386767666666665e-11*tk4 + 
          4.419278199999999e-15*tk5 + (-1.39958323e+04 - tk1*13.656323); 
      }
      else
      {
        cgspl[14] = 4.79372315*tk1*(1-tklog) + 4.954166845e-03*tk2 + 
          -6.220333466666667e-06*tk3 + 3.160710508333332e-09*tk4 + 
          -6.588632599999998e-13*tk5 + (-1.43089567e+04 - tk1*0.6028129); 
      }
    }
    // Species CH3O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[15] = 4.75779238*tk1*(1-tklog) + -3.72071237e-03*tk2 + 
          4.495086266666667e-07*tk3 + -3.650754199999999e-11*tk4 + 
          1.31768549e-15*tk5 + (378.11194 - tk1*-1.96680028); 
      }
      else
      {
        cgspl[15] = 3.71180502*tk1*(1-tklog) + 1.40231653e-03*tk2 + 
          -6.275849516666667e-06*tk3 + 3.942267408333333e-09*tk4 + 
          -9.329420999999999e-13*tk5 + (1.2956976e+03 - tk1*6.57240864); 
      }
    }
    // Species CO
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[16] = 2.71518561*tk1*(1-tklog) + -1.031263715e-03*tk2 + 
          1.664709618333334e-07*tk3 + -1.9171084e-11*tk4 + 1.01823858e-15*tk5 + 
          (-1.41518724e+04 - tk1*7.81868772); 
      }
      else
      {
        cgspl[16] = 3.57953347*tk1*(1-tklog) + 3.0517684e-04*tk2 + 
          -1.69469055e-07*tk3 + -7.558382366666664e-11*tk4 + 
          4.522122494999999e-14*tk5 + (-1.4344086e+04 - tk1*3.50840928); 
      }
    }
    // Species CO2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[17] = 3.85746029*tk1*(1-tklog) + -2.20718513e-03*tk2 + 
          3.691356733333334e-07*tk3 + -4.362418233333333e-11*tk4 + 
          2.36042082e-15*tk5 + (-4.8759166e+04 - tk1*2.27163806); 
      }
      else
      {
        cgspl[17] = 2.35677352*tk1*(1-tklog) + -4.492298385e-03*tk2 + 
          1.187260448333333e-06*tk3 + -2.049325183333333e-10*tk4 + 
          7.184977399999998e-15*tk5 + (-4.83719697e+04 - tk1*9.90105222); 
      }
    }
    // Species C2H2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[18] = 4.14756964*tk1*(1-tklog) + -2.98083332e-03*tk2 + 
          3.9549142e-07*tk3 + -3.895101424999999e-11*tk4 + 1.806176065e-15*tk5 + 
          (2.59359992e+04 - tk1*-1.23028121); 
      }
      else
      {
        cgspl[18] = 0.808681094*tk1*(1-tklog) + -0.01168078145*tk2 + 
          5.91953025e-06*tk3 + -2.334603641666666e-09*tk4 + 
          4.250364869999999e-13*tk5 + (2.64289807e+04 - tk1*13.9397051); 
      }
    }
    // Species C2H3
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[19] = 3.016724*tk1*(1-tklog) + -5.1651146e-03*tk2 + 
          7.801372483333334e-07*tk3 + -8.480273999999997e-11*tk4 + 
          4.313035204999999e-15*tk5 + (3.46128739e+04 - tk1*7.78732378); 
      }
      else
      {
        cgspl[19] = 3.21246645*tk1*(1-tklog) + -7.5739581e-04*tk2 + 
          -4.320156866666667e-06*tk3 + 2.980482058333333e-09*tk4 + 
          -7.357543649999998e-13*tk5 + (3.48598468e+04 - tk1*8.51054025); 
      }
    }
    // Species C2H4
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[20] = 2.03611116*tk1*(1-tklog) + -7.32270755e-03*tk2 + 
          1.118463191666667e-06*tk3 + -1.226857691666667e-10*tk4 + 
          6.285303049999998e-15*tk5 + (4.93988614e+03 - tk1*10.3053693); 
      }
      else
      {
        cgspl[20] = 3.95920148*tk1*(1-tklog) + 3.785261235e-03*tk2 + 
          -9.516504866666668e-06*tk3 + 5.763239608333332e-09*tk4 + 
          -1.349421865e-12*tk5 + (5.08977593e+03 - tk1*4.09733096); 
      }
    }
    // Species C2H5
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[21] = 1.95465642*tk1*(1-tklog) + -8.698636100000001e-03*tk2 + 
          1.330344446666667e-06*tk3 + -1.460147408333333e-10*tk4 + 
          7.482078799999998e-15*tk5 + (1.285752e+04 - tk1*13.4624343); 
      }
      else
      {
        cgspl[21] = 4.30646568*tk1*(1-tklog) + 2.09329446e-03*tk2 + 
          -8.28571345e-06*tk3 + 4.992721716666666e-09*tk4 + -1.15254502e-12*tk5 
          + (1.28416265e+04 - tk1*4.70720924); 
      }
    }
    // Species C2H6
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[22] = 1.0718815*tk1*(1-tklog) + -0.01084263385*tk2 + 
          1.67093445e-06*tk3 + -1.845100008333333e-10*tk4 + 
          9.500144499999998e-15*tk5 + (-1.14263932e+04 - tk1*15.1156107); 
      }
      else
      {
        cgspl[22] = 4.29142492*tk1*(1-tklog) + 2.75077135e-03*tk2 + 
          -9.990638133333334e-06*tk3 + 5.903885708333332e-09*tk4 + 
          -1.343428855e-12*tk5 + (-1.15222055e+04 - tk1*2.66682316); 
      }
    }
    // Species CH2CHO
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[23] = 5.9756699*tk1*(1-tklog) + -4.0652957e-03*tk2 + 
          4.5727075e-07*tk3 + -3.391920083333333e-11*tk4 + 1.08800855e-15*tk5 + 
          (-969.5 - tk1*-5.0320879); 
      }
      else
      {
        cgspl[23] = 3.4090624*tk1*(1-tklog) + -5.369287e-03*tk2 + 
          -3.1524875e-07*tk3 + 5.965485916666666e-10*tk4 + -1.43369255e-13*tk5 + 
          (62.0 - tk1*9.571453500000001); 
      }
    }
    // Species AC3H5
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[24] = 6.5007877*tk1*(1-tklog) + -7.1623655e-03*tk2 + 
          9.463605333333333e-07*tk3 + -9.234000833333331e-11*tk4 + 
          4.518194349999999e-15*tk5 + (1.7482449e+04 - tk1*-11.24305); 
      }
      else
      {
        cgspl[24] = 1.3631835*tk1*(1-tklog) + -9.906910499999999e-03*tk2 + 
          -2.082843333333333e-06*tk3 + 2.779629583333333e-09*tk4 + 
          -7.923285499999998e-13*tk5 + (1.9245629e+04 - tk1*17.173214); 
      }
    }
    // Species C3H6
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[25] = 6.732257*tk1*(1-tklog) + -7.45417e-03*tk2 + 
          8.249831666666667e-07*tk3 + -6.010018333333333e-11*tk4 + 
          1.883101999999999e-15*tk5 + (-923.5703 - tk1*-13.31335); 
      }
      else
      {
        cgspl[25] = 1.493307*tk1*(1-tklog) + -0.01046259*tk2 + 
          -7.477990000000001e-07*tk3 + 1.39076e-09*tk4 + 
          -3.579072999999999e-13*tk5 + (1.074826e+03 - tk1*16.14534); 
      }
    }
    // Species NC3H7
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[26] = 7.7097479*tk1*(1-tklog) + -8.015742500000001e-03*tk2 + 
          8.786706333333333e-07*tk3 + -6.324029333333333e-11*tk4 + 
          1.94313595e-15*tk5 + (7.9762236e+03 - tk1*-15.515297); 
      }
      else
      {
        cgspl[26] = 1.0491173*tk1*(1-tklog) + -0.0130044865*tk2 + 
          -3.923752666666667e-07*tk3 + 1.632927666666666e-09*tk4 + 
          -4.686010349999999e-13*tk5 + (1.0312346e+04 - tk1*21.136034); 
      }
    }
    // Species C2H3CHO
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[27] = 5.8111868*tk1*(1-tklog) + -8.557128000000001e-03*tk2 + 
          1.247236016666667e-06*tk3 + -1.187687416666666e-10*tk4 + 
          4.587342049999999e-15*tk5 + (-1.0784054e+04 - tk1*-4.8588004); 
      }
      else
      {
        cgspl[27] = 1.2713498*tk1*(1-tklog) + -0.013115527*tk2 + 
          1.548538416666667e-06*tk3 + 3.986439333333333e-10*tk4 + 
          -1.67402715e-13*tk5 + (-9.335734399999999e+03 - tk1*19.498077); 
      }
    }
    // Species C4H7
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[28] = 7.0134835*tk1*(1-tklog) + -0.011317279*tk2 + 
          1.5424245e-06*tk3 + -1.400660583333333e-10*tk4 + 
          5.204308499999999e-15*tk5 + (2.0955008e+04 - tk1*-8.889308); 
      }
      else
      {
        cgspl[28] = 0.74449432*tk1*(1-tklog) + -0.0198394285*tk2 + 
          3.816347666666667e-06*tk3 + -1.779414416666666e-10*tk4 + 
          -1.15481875e-13*tk5 + (2.2653328e+04 - tk1*23.437878); 
      }
    }
    // Species C4H81
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[29] = 2.0535841*tk1*(1-tklog) + -0.0171752535*tk2 + 
          2.6471995e-06*tk3 + -2.757471833333332e-10*tk4 + 1.26805225e-14*tk5 + 
          (-2.1397231e+03 - tk1*15.543201); 
      }
      else
      {
        cgspl[29] = 1.181138*tk1*(1-tklog) + -0.01542669*tk2 + 
          -8.477541166666668e-07*tk3 + 2.054573999999999e-09*tk4 + 
          -5.555096499999998e-13*tk5 + (-1.7904004e+03 - tk1*21.062469); 
      }
    }
    // Species PC4H9
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[30] = 8.6822395*tk1*(1-tklog) + -0.0118455355*tk2 + 
          1.265814416666667e-06*tk3 + -5.535594666666665e-11*tk4 + 
          -2.7422568e-15*tk5 + (4.9644058e+03 - tk1*-17.891747); 
      }
      else
      {
        cgspl[30] = 1.2087042*tk1*(1-tklog) + -0.0191487485*tk2 + 
          1.211008483333333e-06*tk3 + 1.28571225e-09*tk4 + 
          -4.342971749999999e-13*tk5 + (7.322104e+03 - tk1*22.169268); 
      }
    }
    // Species C5H9
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[31] = 10.13864*tk1*(1-tklog) + -0.011357069*tk2 + 
          1.298507716666667e-06*tk3 + -9.897101666666664e-11*tk4 + 
          3.296622399999999e-15*tk5 + (-1.7218359e+03 - tk1*-33.125885); 
      }
      else
      {
        cgspl[31] = -2.4190111*tk1*(1-tklog) + -0.0202151945*tk2 + 
          -1.130038983333333e-06*tk3 + 2.810395166666666e-09*tk4 + 
          -7.558356499999999e-13*tk5 + (2.8121887e+03 - tk1*36.459244); 
      }
    }
    // Species C5H10
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[32] = 14.5851539*tk1*(1-tklog) + -0.01120362355*tk2 + 
          1.272246708333333e-06*tk3 + -9.849080499999998e-11*tk4 + 
          3.421925694999999e-15*tk5 + (-1.00898205e+04 - tk1*-52.3683936); 
      }
      else
      {
        cgspl[32] = -1.06223481*tk1*(1-tklog) + -0.0287109147*tk2 + 
          6.241448166666668e-06*tk3 + -1.061374908333333e-09*tk4 + 
          8.980489449999998e-14*tk5 + (-4.46546666e+03 - tk1*32.273979); 
      }
    }
    // Species PXC5H11
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[33] = 15.2977446*tk1*(1-tklog) + -0.0119867655*tk2 + 
          1.363988246666667e-06*tk3 + -1.057358966666666e-10*tk4 + 
          3.677045274999999e-15*tk5 + (-980.712307 - tk1*-54.4829293); 
      }
      else
      {
        cgspl[33] = 0.0524384081*tk1*(1-tklog) + -0.0280398479*tk2 + 
          5.525763383333334e-06*tk3 + -8.146114841666664e-10*tk4 + 
          5.700482999999999e-14*tk5 + (4.7161146e+03 - tk1*28.7238666); 
      }
    }
    // Species C6H12
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[34] = 17.8337529*tk1*(1-tklog) + -0.0133688829*tk2 + 
          1.516727955e-06*tk3 + -1.173498066666666e-10*tk4 + 4.07562122e-15*tk5 
          + (-1.4206286e+04 - tk1*-68.38188510000001); 
      }
      else
      {
        cgspl[34] = -1.35275205*tk1*(1-tklog) + -0.0349327713*tk2 + 
          7.656800366666667e-06*tk3 + -1.308061191666666e-09*tk4 + 
          1.106480875e-13*tk5 + (-7.34368617e+03 - tk1*35.3120691); 
      }
    }
    // Species PXC6H13
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[35] = 18.538547*tk1*(1-tklog) + -0.0141553981*tk2 + 
          1.60884541e-06*tk3 + -1.246229875e-10*tk4 + 4.331680319999999e-15*tk5 
          + (-5.09299041e+03 - tk1*-70.4490943); 
      }
      else
      {
        cgspl[35] = -0.204871465*tk1*(1-tklog) + -0.0341900636*tk2 + 
          6.9074652e-06*tk3 + -1.05129835e-09*tk4 + 7.656002899999998e-14*tk5 + 
          (1.83280393e+03 - tk1*31.6075093); 
      }
    }
    // Species C7H14
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[36] = 21.0898039*tk1*(1-tklog) + -0.0155303939*tk2 + 
          1.76074655e-06*tk3 + -1.361714833333333e-10*tk4 + 
          4.727991094999999e-15*tk5 + (-1.83260065e+04 - 
          tk1*-84.43911079999999); 
      }
      else
      {
        cgspl[36] = -1.67720549*tk1*(1-tklog) + -0.04123058005*tk2 + 
          9.108401800000001e-06*tk3 + -1.565519191666666e-09*tk4 + 
          1.328689915e-13*tk5 + (-1.02168601e+04 - tk1*38.5068032); 
      }
    }
    // Species PXC7H15
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[37] = 21.7940709*tk1*(1-tklog) + -0.01631401215*tk2 + 
          1.852304066666667e-06*tk3 + -1.4338929e-10*tk4 + 
          4.981834994999999e-15*tk5 + (-9.20938221e+03 - tk1*-86.4954311); 
      }
      else
      {
        cgspl[37] = -0.499570406*tk1*(1-tklog) + -0.04044132335*tk2 + 
          8.342212566666669e-06*tk3 + -1.304577566666666e-09*tk4 + 
          9.830811349999998e-14*tk5 + (-1.04590223e+03 - tk1*34.6564011); 
      }
    }
    // Species C8H16
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[38] = 24.3540125*tk1*(1-tklog) + -0.0176833231*tk2 + 
          2.003473133333334e-06*tk3 + -1.548792108333333e-10*tk4 + 
          5.376113099999999e-15*tk5 + (-2.24485674e+04 - tk1*-100.537716); 
      }
      else
      {
        cgspl[38] = -1.89226915*tk1*(1-tklog) + -0.04730331785*tk2 + 
          1.045642535e-05*tk3 + -1.792985908333333e-09*tk4 + 1.513593415e-13*tk5 
          + (-1.31074559e+04 - tk1*41.1878981); 
      }
    }
    // Species PXC8H17
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[39] = 25.0510356*tk1*(1-tklog) + -0.0184740081*tk2 + 
          2.096087733333334e-06*tk3 + -1.621903408333333e-10*tk4 + 
          5.633444899999999e-15*tk5 + (-1.33300535e+04 - tk1*-102.557384); 
      }
      else
      {
        cgspl[39] = -0.772759438*tk1*(1-tklog) + -0.04662748525*tk2 + 
          9.740787416666667e-06*tk3 + -1.54641845e-09*tk4 + 1.185637415e-13*tk5 
          + (-3.92689511e+03 - tk1*37.6130631); 
      }
    }
    // Species C9H18
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[40] = 27.6142176*tk1*(1-tklog) + -0.01984126435*tk2 + 
          2.246990766666667e-06*tk3 + -1.7365871e-10*tk4 + 
          6.026964699999999e-15*tk5 + (-2.65709061e+04 - tk1*-116.618623); 
      }
      else
      {
        cgspl[40] = -2.16108263*tk1*(1-tklog) + -0.0534791485*tk2 + 
          1.184955406666667e-05*tk3 + -2.033092308333333e-09*tk4 + 
          1.713857734999999e-13*tk5 + (-1.59890847e+04 - tk1*44.1245128); 
      }
    }
    // Species PXC9H19
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[41] = 28.3097514*tk1*(1-tklog) + -0.0206328672*tk2 + 
          2.339721483333334e-06*tk3 + -1.809790591666666e-10*tk4 + 
          6.284615349999998e-15*tk5 + (-1.7451603e+04 - tk1*-116.837897); 
      }
      else
      {
        cgspl[41] = -1.04387292*tk1*(1-tklog) + -0.0528086415*tk2 + 
          1.113666618333334e-05*tk3 + -1.787384716666666e-09*tk4 + 
          1.387021375e-13*tk5 + (-6.80818512e+03 - tk1*42.3518992); 
      }
    }
    // Species C10H20
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[42] = 30.8753903*tk1*(1-tklog) + -0.0219985763*tk2 + 
          2.4904255e-06*tk3 + -1.924313983333333e-10*tk4 + 
          6.677573849999999e-15*tk5 + (-3.06937307e+04 - tk1*-132.705172); 
      }
      else
      {
        cgspl[42] = -2.42901688*tk1*(1-tklog) + -0.059652799*tk2 + 
          1.324148375e-05*tk3 + -2.272804966666666e-09*tk4 + 
          1.913591864999999e-13*tk5 + (-1.88708365e+04 - tk1*47.0571383); 
      }
    }
    // Species PXC10H21
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[43] = 31.569716*tk1*(1-tklog) + -0.02279092015*tk2 + 
          2.583249416666667e-06*tk3 + -1.997591108333333e-10*tk4 + 
          6.935477949999998e-15*tk5 + (-2.15737832e+04 - tk1*-134.708986); 
      }
      else
      {
        cgspl[43] = -1.31358348*tk1*(1-tklog) + -0.0589864065*tk2 + 
          1.253071798333334e-05*tk3 + -2.027759216666666e-09*tk4 + 
          1.58761426e-13*tk5 + (-9.689675499999999e+03 - tk1*43.5010452); 
      }
    }
    // Species PXC12H25
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[44] = 38.0921885*tk1*(1-tklog) + -0.0271053924*tk2 + 
          3.070091950000001e-06*tk3 + -2.373018108333333e-10*tk4 + 
          8.236587399999999e-15*tk5 + (-2.98194375e+04 - tk1*-166.882734); 
      }
      else
      {
        cgspl[44] = -1.85028741*tk1*(1-tklog) + -0.071335354*tk2 + 
          1.531527591666667e-05*tk3 + -2.5073616e-09*tk4 + 
          1.987271499999999e-13*tk5 + (-1.54530435e+04 - tk1*49.3702421); 
      }
    }
    // Species SXC12H25
    {
      if (tk1 > 1.385e+03)
      {
        cgspl[45] = 37.9688268*tk1*(1-tklog) + -0.0269359732*tk2 + 
          3.036187716666667e-06*tk3 + -2.339787524999999e-10*tk4 + 
          8.105420999999999e-15*tk5 + (-3.12144988e+04 - tk1*-165.805933); 
      }
      else
      {
        cgspl[45] = -1.36787089*tk1*(1-tklog) + -0.06867767399999999*tk2 + 
          1.373460263333333e-05*tk3 + -1.970179683333333e-09*tk4 + 
          1.23717966e-13*tk5 + (-1.67660539e+04 - tk1*48.3521895); 
      }
    }
    // Species S3XC12H25
    {
      if (tk1 > 1.385e+03)
      {
        cgspl[46] = 37.9688268*tk1*(1-tklog) + -0.0269359732*tk2 + 
          3.036187716666667e-06*tk3 + -2.339787524999999e-10*tk4 + 
          8.105420999999999e-15*tk5 + (-3.12144988e+04 - tk1*-165.805933); 
      }
      else
      {
        cgspl[46] = -1.36787089*tk1*(1-tklog) + -0.06867767399999999*tk2 + 
          1.373460263333333e-05*tk3 + -1.970179683333333e-09*tk4 + 
          1.23717966e-13*tk5 + (-1.67660539e+04 - tk1*48.3521895); 
      }
    }
    // Species C12H24
    {
      if (tk1 > 1.391e+03)
      {
        cgspl[47] = 37.4002111*tk1*(1-tklog) + -0.02631153765*tk2 + 
          2.977071983333334e-06*tk3 + -2.299582191666666e-10*tk4 + 
          7.978124949999998e-15*tk5 + (-3.89405962e+04 - tk1*-164.892663); 
      }
      else
      {
        cgspl[47] = -2.96342681*tk1*(1-tklog) + -0.07199618000000001*tk2 + 
          1.602306691666667e-05*tk3 + -2.751453941666666e-09*tk4 + 
          2.31199095e-13*tk5 + (-2.46345299e+04 - tk1*52.915887); 
      }
    }
    // Species C12H25O2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[48] = 28.4782*tk1*(1-tklog) + -0.02687695*tk2 + 2.8031e-06*tk3 + 
          -2.094725e-10*tk4 + 7.360399999999999e-15*tk5 + (-3.74118e+04 - 
          tk1*-109.121); 
      }
      else
      {
        cgspl[48] = 5.31404*tk1*(1-tklog) + -0.04469365*tk2 + 
          -2.422516666666667e-06*tk3 + 6.243749999999999e-09*tk4 + 
          -1.676625e-12*tk5 + (-2.98918e+04 - tk1*16.9741); 
      }
    }
    // Species C12OOH
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[49] = 29.2019*tk1*(1-tklog) + -0.02579585*tk2 + 
          2.622116666666667e-06*tk3 + -1.919216666666666e-10*tk4 + 
          6.631999999999999e-15*tk5 + (-3.11192e+04 - tk1*-108.855); 
      }
      else
      {
        cgspl[49] = 5.15231*tk1*(1-tklog) + -0.04989565*tk2 + 
          3.010583333333334e-06*tk3 + 3.486958333333333e-09*tk4 + 
          -1.11393e-12*tk5 + (-2.3838e+04 - tk1*19.3526); 
      }
    }
    // Species O2C12H24OOH
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[50] = 35.0907*tk1*(1-tklog) + -0.0255295*tk2 + 
          2.572416666666667e-06*tk3 + -1.871891666666666e-10*tk4 + 
          6.445049999999999e-15*tk5 + (-5.12675e+04 - tk1*-137.75); 
      }
      else
      {
        cgspl[50] = 0.481972*tk1*(1-tklog) + -0.07251000000000001*tk2 + 
          1.665513333333334e-05*tk3 + -2.170183333333333e-09*tk4 + 
          -5.967899999999999e-14*tk5 + (-4.16875e+04 - tk1*41.3429); 
      }
    }
    // Species OC12H23OOH
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[51] = 23.6731*tk1*(1-tklog) + -0.0308196*tk2 + 
          3.497266666666667e-06*tk3 + -2.776383333333333e-10*tk4 + 
          1.01795e-14*tk5 + (-7.18258e+04 - tk1*-77.7662); 
      }
      else
      {
        cgspl[51] = 8.80733*tk1*(1-tklog) + -0.03253115*tk2 + -1.15843e-05*tk3 + 
          1.057541666666667e-08*tk4 + -2.554954999999999e-12*tk5 + 
          (-6.653610000000001e+04 - tk1*6.84155); 
      }
    }
    // Species N2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[52] = 2.92664*tk1*(1-tklog) + -7.439884e-04*tk2 + 
          9.474600000000001e-08*tk3 + -8.414198333333332e-12*tk4 + 
          3.376675499999999e-16*tk5 + (-922.7977 - tk1*5.980528); 
      }
      else
      {
        cgspl[52] = 3.298677*tk1*(1-tklog) + -7.041202e-04*tk2 + 6.60537e-07*tk3 
          + -4.7012625e-10*tk4 + 1.222427e-13*tk5 + (-1.0208999e+03 - 
          tk1*3.950372); 
      }
    }
  }
  
  double mole_frac[35];
  // Compute mole fractions
  {
    double sumyow = temperature * avmolwt * R0;
    sumyow = pressure/sumyow;
    mole_frac[0] = mass_frac[0] * recip_molecular_masses[0];
    mole_frac[0] = (mole_frac[0] > 1e-200) ? mole_frac[0] : 1e-200;
    mole_frac[0] *= sumyow;
    mole_frac[1] = mass_frac[1] * recip_molecular_masses[1];
    mole_frac[1] = (mole_frac[1] > 1e-200) ? mole_frac[1] : 1e-200;
    mole_frac[1] *= sumyow;
    mole_frac[2] = mass_frac[2] * recip_molecular_masses[2];
    mole_frac[2] = (mole_frac[2] > 1e-200) ? mole_frac[2] : 1e-200;
    mole_frac[2] *= sumyow;
    mole_frac[3] = mass_frac[3] * recip_molecular_masses[3];
    mole_frac[3] = (mole_frac[3] > 1e-200) ? mole_frac[3] : 1e-200;
    mole_frac[3] *= sumyow;
    mole_frac[4] = mass_frac[4] * recip_molecular_masses[4];
    mole_frac[4] = (mole_frac[4] > 1e-200) ? mole_frac[4] : 1e-200;
    mole_frac[4] *= sumyow;
    mole_frac[5] = mass_frac[5] * recip_molecular_masses[5];
    mole_frac[5] = (mole_frac[5] > 1e-200) ? mole_frac[5] : 1e-200;
    mole_frac[5] *= sumyow;
    mole_frac[6] = mass_frac[6] * recip_molecular_masses[6];
    mole_frac[6] = (mole_frac[6] > 1e-200) ? mole_frac[6] : 1e-200;
    mole_frac[6] *= sumyow;
    mole_frac[7] = mass_frac[7] * recip_molecular_masses[7];
    mole_frac[7] = (mole_frac[7] > 1e-200) ? mole_frac[7] : 1e-200;
    mole_frac[7] *= sumyow;
    mole_frac[8] = mass_frac[8] * recip_molecular_masses[8];
    mole_frac[8] = (mole_frac[8] > 1e-200) ? mole_frac[8] : 1e-200;
    mole_frac[8] *= sumyow;
    mole_frac[9] = mass_frac[9] * recip_molecular_masses[9];
    mole_frac[9] = (mole_frac[9] > 1e-200) ? mole_frac[9] : 1e-200;
    mole_frac[9] *= sumyow;
    mole_frac[10] = mass_frac[10] * recip_molecular_masses[10];
    mole_frac[10] = (mole_frac[10] > 1e-200) ? mole_frac[10] : 1e-200;
    mole_frac[10] *= sumyow;
    mole_frac[11] = mass_frac[11] * recip_molecular_masses[11];
    mole_frac[11] = (mole_frac[11] > 1e-200) ? mole_frac[11] : 1e-200;
    mole_frac[11] *= sumyow;
    mole_frac[12] = mass_frac[12] * recip_molecular_masses[12];
    mole_frac[12] = (mole_frac[12] > 1e-200) ? mole_frac[12] : 1e-200;
    mole_frac[12] *= sumyow;
    mole_frac[13] = mass_frac[13] * recip_molecular_masses[13];
    mole_frac[13] = (mole_frac[13] > 1e-200) ? mole_frac[13] : 1e-200;
    mole_frac[13] *= sumyow;
    mole_frac[14] = mass_frac[14] * recip_molecular_masses[14];
    mole_frac[14] = (mole_frac[14] > 1e-200) ? mole_frac[14] : 1e-200;
    mole_frac[14] *= sumyow;
    mole_frac[15] = mass_frac[15] * recip_molecular_masses[15];
    mole_frac[15] = (mole_frac[15] > 1e-200) ? mole_frac[15] : 1e-200;
    mole_frac[15] *= sumyow;
    mole_frac[16] = mass_frac[16] * recip_molecular_masses[16];
    mole_frac[16] = (mole_frac[16] > 1e-200) ? mole_frac[16] : 1e-200;
    mole_frac[16] *= sumyow;
    mole_frac[17] = mass_frac[17] * recip_molecular_masses[17];
    mole_frac[17] = (mole_frac[17] > 1e-200) ? mole_frac[17] : 1e-200;
    mole_frac[17] *= sumyow;
    mole_frac[18] = mass_frac[18] * recip_molecular_masses[18];
    mole_frac[18] = (mole_frac[18] > 1e-200) ? mole_frac[18] : 1e-200;
    mole_frac[18] *= sumyow;
    mole_frac[19] = mass_frac[19] * recip_molecular_masses[19];
    mole_frac[19] = (mole_frac[19] > 1e-200) ? mole_frac[19] : 1e-200;
    mole_frac[19] *= sumyow;
    mole_frac[20] = mass_frac[20] * recip_molecular_masses[20];
    mole_frac[20] = (mole_frac[20] > 1e-200) ? mole_frac[20] : 1e-200;
    mole_frac[20] *= sumyow;
    mole_frac[21] = mass_frac[21] * recip_molecular_masses[21];
    mole_frac[21] = (mole_frac[21] > 1e-200) ? mole_frac[21] : 1e-200;
    mole_frac[21] *= sumyow;
    mole_frac[22] = mass_frac[22] * recip_molecular_masses[22];
    mole_frac[22] = (mole_frac[22] > 1e-200) ? mole_frac[22] : 1e-200;
    mole_frac[22] *= sumyow;
    mole_frac[23] = mass_frac[23] * recip_molecular_masses[23];
    mole_frac[23] = (mole_frac[23] > 1e-200) ? mole_frac[23] : 1e-200;
    mole_frac[23] *= sumyow;
    mole_frac[24] = mass_frac[24] * recip_molecular_masses[24];
    mole_frac[24] = (mole_frac[24] > 1e-200) ? mole_frac[24] : 1e-200;
    mole_frac[24] *= sumyow;
    mole_frac[25] = mass_frac[25] * recip_molecular_masses[25];
    mole_frac[25] = (mole_frac[25] > 1e-200) ? mole_frac[25] : 1e-200;
    mole_frac[25] *= sumyow;
    mole_frac[26] = mass_frac[26] * recip_molecular_masses[26];
    mole_frac[26] = (mole_frac[26] > 1e-200) ? mole_frac[26] : 1e-200;
    mole_frac[26] *= sumyow;
    mole_frac[27] = mass_frac[27] * recip_molecular_masses[27];
    mole_frac[27] = (mole_frac[27] > 1e-200) ? mole_frac[27] : 1e-200;
    mole_frac[27] *= sumyow;
    mole_frac[28] = mass_frac[28] * recip_molecular_masses[28];
    mole_frac[28] = (mole_frac[28] > 1e-200) ? mole_frac[28] : 1e-200;
    mole_frac[28] *= sumyow;
    mole_frac[29] = mass_frac[29] * recip_molecular_masses[29];
    mole_frac[29] = (mole_frac[29] > 1e-200) ? mole_frac[29] : 1e-200;
    mole_frac[29] *= sumyow;
    mole_frac[30] = mass_frac[30] * recip_molecular_masses[30];
    mole_frac[30] = (mole_frac[30] > 1e-200) ? mole_frac[30] : 1e-200;
    mole_frac[30] *= sumyow;
    mole_frac[31] = mass_frac[31] * recip_molecular_masses[31];
    mole_frac[31] = (mole_frac[31] > 1e-200) ? mole_frac[31] : 1e-200;
    mole_frac[31] *= sumyow;
    mole_frac[32] = mass_frac[32] * recip_molecular_masses[32];
    mole_frac[32] = (mole_frac[32] > 1e-200) ? mole_frac[32] : 1e-200;
    mole_frac[32] *= sumyow;
    mole_frac[33] = mass_frac[33] * recip_molecular_masses[33];
    mole_frac[33] = (mole_frac[33] > 1e-200) ? mole_frac[33] : 1e-200;
    mole_frac[33] *= sumyow;
    mole_frac[34] = mass_frac[34] * recip_molecular_masses[34];
    mole_frac[34] = (mole_frac[34] > 1e-200) ? mole_frac[34] : 1e-200;
    mole_frac[34] *= sumyow;
  }
  
  double thbctemp[9];
  // Computing third body values
  {
    double ctot = 0.0;
    ctot += mole_frac[0];
    ctot += mole_frac[1];
    ctot += mole_frac[2];
    ctot += mole_frac[3];
    ctot += mole_frac[4];
    ctot += mole_frac[5];
    ctot += mole_frac[6];
    ctot += mole_frac[7];
    ctot += mole_frac[8];
    ctot += mole_frac[9];
    ctot += mole_frac[10];
    ctot += mole_frac[11];
    ctot += mole_frac[12];
    ctot += mole_frac[13];
    ctot += mole_frac[14];
    ctot += mole_frac[15];
    ctot += mole_frac[16];
    ctot += mole_frac[17];
    ctot += mole_frac[18];
    ctot += mole_frac[19];
    ctot += mole_frac[20];
    ctot += mole_frac[21];
    ctot += mole_frac[22];
    ctot += mole_frac[23];
    ctot += mole_frac[24];
    ctot += mole_frac[25];
    ctot += mole_frac[26];
    ctot += mole_frac[27];
    ctot += mole_frac[28];
    ctot += mole_frac[29];
    ctot += mole_frac[30];
    ctot += mole_frac[31];
    ctot += mole_frac[32];
    ctot += mole_frac[33];
    ctot += mole_frac[34];
    thbctemp[0] = ctot + 10.89*mole_frac[6] - 0.15*mole_frac[8] + 
      0.09000000000000008*mole_frac[12] + 1.18*mole_frac[13]; 
    thbctemp[1] = ctot + mole_frac[5] + 5.0*mole_frac[6] + 0.75*mole_frac[12] + 
      2.6*mole_frac[13]; 
    thbctemp[2] = ctot - mole_frac[5] - mole_frac[6] - mole_frac[13];
    thbctemp[3] = ctot + mole_frac[5] + 5.3*mole_frac[6] + 0.75*mole_frac[12] + 
      2.6*mole_frac[13]; 
    thbctemp[4] = ctot + 1.4*mole_frac[5] + 14.4*mole_frac[6] + 
      0.75*mole_frac[12] + 2.6*mole_frac[13]; 
    thbctemp[5] = ctot + mole_frac[5] + 11.0*mole_frac[6] + 0.75*mole_frac[12] + 
      2.6*mole_frac[13]; 
    thbctemp[6] = ctot + mole_frac[5] - mole_frac[6] + 0.75*mole_frac[12] + 
      2.6*mole_frac[13]; 
    thbctemp[7] = ctot + mole_frac[5] + 5.0*mole_frac[6] + mole_frac[10] + 
      0.5*mole_frac[12] + mole_frac[13] + 2.0*mole_frac[16]; 
    thbctemp[8] = ctot + mole_frac[5] + 5.0*mole_frac[6] + mole_frac[10] + 
      0.5*mole_frac[12] + mole_frac[13] + 2.0*mole_frac[14] + 2.0*mole_frac[15] 
      + 2.0*mole_frac[16]; 
  }
  
  double rr_f[268];
  double rr_r[268];
  //   0)  H + O2 <=> O + OH
  {
    double forward = 9.756e+13 * exp(-1.484226e+04*ortc);
    double xik = -cgspl[1] + cgspl[2] + cgspl[3] - cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[0] = forward * mole_frac[1] * mole_frac[8];
    rr_r[0] = reverse * mole_frac[2] * mole_frac[3];
  }
  //   1)  O + H2 <=> H + OH
  {
    double forward = 4.589e+04 * exp(2.7*vlntemp - 6.26e+03*ortc);
    double xik = cgspl[1] - cgspl[2] + cgspl[3] - cgspl[5];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[1] = forward * mole_frac[2] * mole_frac[5];
    rr_r[1] = reverse * mole_frac[1] * mole_frac[3];
  }
  //   2)  OH + H2 <=> H + H2O
  {
    double forward = 1.024e+08 * exp(1.6*vlntemp - 3.29828e+03*ortc);
    double xik = cgspl[1] - cgspl[3] - cgspl[5] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[2] = forward * mole_frac[3] * mole_frac[5];
    rr_r[2] = reverse * mole_frac[1] * mole_frac[6];
  }
  //   3)  2 OH <=> O + H2O
  {
    double forward = 3.973e+04 * exp(2.4*vlntemp + 2.11e+03*ortc);
    double xik = cgspl[2] - 2.0 * cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[3] = forward * mole_frac[3] * mole_frac[3];
    rr_r[3] = reverse * mole_frac[2] * mole_frac[6];
  }
  //   4)  H + O2 (+M) <=> HO2 (+M)
  {
    double rr_k0 = 6.328e+19 * exp(-1.4 * vlntemp);
    double rr_kinf = 5.116e+12 * exp(0.44 * vlntemp);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.5 * exp(-9.999999999999999e+29 * temperature) + 
      0.5 * exp(-9.999999999999999e-31 * temperature),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] + cgspl[4] - cgspl[8];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[4] = forward * mole_frac[1] * mole_frac[8];
    rr_r[4] = reverse * mole_frac[4];
  }
  //   5)  H + HO2 <=> 2 OH
  {
    double forward = 7.485e+13 * exp(-295.0*ortc);
    double xik = -cgspl[1] + 2.0 * cgspl[3] - cgspl[4];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[5] = forward * mole_frac[1] * mole_frac[4];
    rr_r[5] = reverse * mole_frac[3] * mole_frac[3];
  }
  //   6)  H2 + O2 <=> H + HO2
  {
    double forward = 5.916e+05 * exp(2.433*vlntemp - 5.3502e+04*ortc);
    double xik = cgspl[1] + cgspl[4] - cgspl[5] - cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[6] = forward * mole_frac[5] * mole_frac[8];
    rr_r[6] = reverse * mole_frac[1] * mole_frac[4];
  }
  //   7)  OH + HO2 <=> H2O + O2
  {
    double forward = 2.891e+13 * exp(501.91*ortc);
    double xik = -cgspl[3] - cgspl[4] + cgspl[6] + cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[7] = forward * mole_frac[3] * mole_frac[4];
    rr_r[7] = reverse * mole_frac[6] * mole_frac[8];
  }
  //   8)  H + HO2 <=> O + H2O
  {
    double forward = 3.97e+12 * exp(-671.0*ortc);
    double xik = -cgspl[1] + cgspl[2] - cgspl[4] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[8] = forward * mole_frac[1] * mole_frac[4];
    rr_r[8] = reverse * mole_frac[2] * mole_frac[6];
  }
  //   9)  O + HO2 <=> OH + O2
  {
    double forward = 4.0e+13;
    double xik = -cgspl[2] + cgspl[3] - cgspl[4] + cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[9] = forward * mole_frac[2] * mole_frac[4];
    rr_r[9] = reverse * mole_frac[3] * mole_frac[8];
  }
  //  10)  2 HO2 <=> H2O2 + O2
  {
    double forward = 1.3e+11 * exp(1.63e+03*ortc);
    double xik = -2.0 * cgspl[4] + cgspl[7] + cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[10] = forward * mole_frac[4] * mole_frac[4];
    rr_r[10] = reverse * mole_frac[7] * mole_frac[8];
  }
  //  11)  2 HO2 <=> H2O2 + O2
  {
    double forward = 3.658e+14 * exp(-1.2e+04*ortc);
    double xik = -2.0 * cgspl[4] + cgspl[7] + cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[11] = forward * mole_frac[4] * mole_frac[4];
    rr_r[11] = reverse * mole_frac[7] * mole_frac[8];
  }
  //  12)  H + H2O2 <=> OH + H2O
  {
    double forward = 2.41e+13 * exp(-3.97e+03*ortc);
    double xik = -cgspl[1] + cgspl[3] + cgspl[6] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[12] = forward * mole_frac[1] * mole_frac[7];
    rr_r[12] = reverse * mole_frac[3] * mole_frac[6];
  }
  //  13)  H + H2O2 <=> HO2 + H2
  {
    double forward = 6.05e+06 * temperature * temperature * exp(-5.2e+03*ortc);
    double xik = -cgspl[1] + cgspl[4] + cgspl[5] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[13] = forward * mole_frac[1] * mole_frac[7];
    rr_r[13] = reverse * mole_frac[4] * mole_frac[5];
  }
  //  14)  O + H2O2 <=> OH + HO2
  {
    double forward = 9.63e+06 * temperature * temperature * exp(-3.97e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[4] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[14] = forward * mole_frac[2] * mole_frac[7];
    rr_r[14] = reverse * mole_frac[3] * mole_frac[4];
  }
  //  15)  OH + H2O2 <=> HO2 + H2O
  {
    double forward = 2.0e+12 * exp(-427.0*ortc);
    double xik = -cgspl[3] + cgspl[4] + cgspl[6] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[15] = forward * mole_frac[3] * mole_frac[7];
    rr_r[15] = reverse * mole_frac[4] * mole_frac[6];
  }
  //  16)  OH + H2O2 <=> HO2 + H2O
  {
    double forward = 2.67e+41 * exp(-7.0*vlntemp - 3.76e+04*ortc);
    double xik = -cgspl[3] + cgspl[4] + cgspl[6] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[16] = forward * mole_frac[3] * mole_frac[7];
    rr_r[16] = reverse * mole_frac[4] * mole_frac[6];
  }
  //  17)  2 OH (+M) <=> H2O2 (+M)
  {
    double rr_k0 = 2.01e+17 * exp(-0.584*vlntemp + 2.293e+03*ortc);
    double rr_kinf = 1.11e+14 * exp(-0.37 * vlntemp);
    double pr = rr_k0 / rr_kinf * thbctemp[1];
    double fcent = log10(MAX(0.2654 * exp(-0.01063829787234043 * temperature) + 
      0.7346 * exp(-5.694760820045558e-04 * temperature) + exp(-5.182e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -2.0 * cgspl[3] + cgspl[7];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[17] = forward * mole_frac[3] * mole_frac[3];
    rr_r[17] = reverse * mole_frac[7];
  }
  //  18)  2 H + M <=> H2 + M
  {
    double forward = 1.78e+18 * otc;
    double xik = -2.0 * cgspl[1] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[18] = forward * mole_frac[1] * mole_frac[1];
    rr_r[18] = reverse * mole_frac[5];
    rr_f[18] *= thbctemp[2];
    rr_r[18] *= thbctemp[2];
  }
  //  19)  H + OH + M <=> H2O + M
  {
    double forward = 4.4e+22 * otc * otc;
    double xik = -cgspl[1] - cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[19] = forward * mole_frac[1] * mole_frac[3];
    rr_r[19] = reverse * mole_frac[6];
    rr_f[19] *= thbctemp[3];
    rr_r[19] *= thbctemp[3];
  }
  //  20)  2 O + M <=> O2 + M
  {
    double forward = 1.2e+17 * otc;
    double xik = -2.0 * cgspl[2] + cgspl[8];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[20] = forward * mole_frac[2] * mole_frac[2];
    rr_r[20] = reverse * mole_frac[8];
    rr_f[20] *= thbctemp[4];
    rr_r[20] *= thbctemp[4];
  }
  //  21)  2 H + H2 <=> 2 H2
  {
    double forward = 9.0e+16 * exp(-0.6 * vlntemp);
    double xik = -2.0 * cgspl[1] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[21] = forward * mole_frac[1] * mole_frac[1] * mole_frac[5];
    rr_r[21] = reverse * mole_frac[5] * mole_frac[5];
  }
  //  22)  2 H + H2O <=> H2 + H2O
  {
    double forward = 5.624e+19 * exp(-1.25 * vlntemp);
    double xik = -2.0 * cgspl[1] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[22] = forward * mole_frac[1] * mole_frac[1] * mole_frac[6];
    rr_r[22] = reverse * mole_frac[5] * mole_frac[6];
  }
  //  23)  2 H + CO2 <=> H2 + CO2
  {
    double forward = 5.5e+20 * otc * otc;
    double xik = -2.0 * cgspl[1] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[23] = forward * mole_frac[1] * mole_frac[1] * mole_frac[13];
    rr_r[23] = reverse * mole_frac[5] * mole_frac[13];
  }
  //  24)  H + O + M <=> OH + M
  {
    double forward = 9.428e+18 * otc;
    double xik = -cgspl[1] - cgspl[2] + cgspl[3];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[24] = forward * mole_frac[1] * mole_frac[2];
    rr_r[24] = reverse * mole_frac[3];
    rr_f[24] *= thbctemp[5];
    rr_r[24] *= thbctemp[5];
  }
  //  25)  OH + CO <=> H + CO2
  {
    double forward = 7.046e+04 * exp(2.053*vlntemp + 355.67*ortc);
    double xik = cgspl[1] - cgspl[3] - cgspl[16] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[25] = forward * mole_frac[3] * mole_frac[12];
    rr_r[25] = reverse * mole_frac[1] * mole_frac[13];
  }
  //  26)  OH + CO <=> H + CO2
  {
    double forward = 5.757e+12 * exp(-0.664*vlntemp - 331.83*ortc);
    double xik = cgspl[1] - cgspl[3] - cgspl[16] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[26] = forward * mole_frac[3] * mole_frac[12];
    rr_r[26] = reverse * mole_frac[1] * mole_frac[13];
  }
  //  27)  HO2 + CO <=> OH + CO2
  {
    double forward = 1.57e+05 * exp(2.18*vlntemp - 1.794261e+04*ortc);
    double xik = cgspl[3] - cgspl[4] - cgspl[16] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[27] = forward * mole_frac[4] * mole_frac[12];
    rr_r[27] = reverse * mole_frac[3] * mole_frac[13];
  }
  //  28)  O + CO (+M) <=> CO2 (+M)
  {
    double rr_k0 = 1.173e+24 * exp(-2.79*vlntemp - 4.191e+03*ortc);
    double rr_kinf = 1.362e+10 * exp(-2.384e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[5];
    double forward = rr_kinf * pr/(1.0 + pr);
    double xik = -cgspl[2] - cgspl[16] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[28] = forward * mole_frac[2] * mole_frac[12];
    rr_r[28] = reverse * mole_frac[13];
  }
  //  29)  O2 + CO <=> O + CO2
  {
    double forward = 1.119e+12 * exp(-4.77e+04*ortc);
    double xik = cgspl[2] - cgspl[8] - cgspl[16] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[29] = forward * mole_frac[8] * mole_frac[12];
    rr_r[29] = reverse * mole_frac[2] * mole_frac[13];
  }
  //  30)  HCO + M <=> H + CO + M
  {
    double forward = 1.87e+17 * otc * exp(-1.7e+04*ortc);
    double xik = cgspl[1] - cgspl[13] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[30] = forward;
    rr_r[30] = reverse * mole_frac[1] * mole_frac[12];
    rr_f[30] *= thbctemp[6];
    rr_r[30] *= thbctemp[6];
  }
  //  31)  H + HCO <=> H2 + CO
  {
    double forward = 1.2e+14;
    double xik = -cgspl[1] + cgspl[5] - cgspl[13] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[31] = forward * mole_frac[1];
    rr_r[31] = reverse * mole_frac[5] * mole_frac[12];
  }
  //  32)  O + HCO <=> OH + CO
  {
    double forward = 3.0e+13;
    double xik = -cgspl[2] + cgspl[3] - cgspl[13] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[32] = forward * mole_frac[2];
    rr_r[32] = reverse * mole_frac[3] * mole_frac[12];
  }
  //  33)  O + HCO <=> H + CO2
  {
    double forward = 3.0e+13;
    double xik = cgspl[1] - cgspl[2] - cgspl[13] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[33] = forward * mole_frac[2];
    rr_r[33] = reverse * mole_frac[1] * mole_frac[13];
  }
  //  34)  OH + HCO <=> H2O + CO
  {
    double forward = 3.02e+13;
    double xik = -cgspl[3] + cgspl[6] - cgspl[13] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[34] = forward * mole_frac[3];
    rr_r[34] = reverse * mole_frac[6] * mole_frac[12];
  }
  //  35)  O2 + HCO <=> HO2 + CO
  {
    double forward = 1.204e+10 * exp(0.8070000000000001*vlntemp + 727.0*ortc);
    double xik = cgspl[4] - cgspl[8] - cgspl[13] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[35] = forward * mole_frac[8];
    rr_r[35] = reverse * mole_frac[4] * mole_frac[12];
  }
  //  36)  H2O + HCO <=> H + H2O + CO
  {
    double forward = 2.244e+18 * otc * exp(-1.7e+04*ortc);
    double xik = cgspl[1] - cgspl[13] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[36] = forward * mole_frac[6];
    rr_r[36] = reverse * mole_frac[1] * mole_frac[6] * mole_frac[12];
  }
  //  37)  H2 + CO (+M) <=> CH2O (+M)
  {
    double rr_k0 = 5.07e+27 * exp(-3.42*vlntemp - 8.435e+04*ortc);
    double rr_kinf = 4.3e+07 * exp(1.5*vlntemp - 7.96e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.06799999999999995 * exp(-5.076142131979695e-03 * 
      temperature) + 0.9320000000000001 * exp(-6.493506493506494e-04 * 
      temperature) + exp(-1.03e+04 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[5] + cgspl[14] - cgspl[16];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[37] = forward * mole_frac[5] * mole_frac[12];
    rr_r[37] = reverse * mole_frac[11];
  }
  //  38)  H + HCO (+M) <=> CH2O (+M)
  {
    double rr_k0 = 1.35e+24 * exp(-2.57*vlntemp - 1.425e+03*ortc);
    double rr_kinf = 1.09e+12 * exp(0.48*vlntemp + 260.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.2176 * exp(-3.690036900369004e-03 * temperature) 
      + 0.7824 * exp(-3.629764065335753e-04 * temperature) + exp(-6.57e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[13] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[38] = forward * mole_frac[1];
    rr_r[38] = reverse * mole_frac[11];
  }
  //  39)  H + CH2 (+M) <=> CH3 (+M)
  {
    double rr_k0 = 3.2e+27 * exp(-3.14*vlntemp - 1.23e+03*ortc);
    double rr_kinf = 2.5e+16 * exp(-0.8 * vlntemp);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.32 * exp(-0.01282051282051282 * temperature) + 
      0.68 * exp(-5.012531328320802e-04 * temperature) + exp(-5.59e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[9] + cgspl[11];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[39] = forward * mole_frac[1];
    rr_r[39] = reverse * mole_frac[9];
  }
  //  40)  O + CH2 <=> H + HCO
  {
    double forward = 8.0e+13;
    double xik = cgspl[1] - cgspl[2] - cgspl[9] + cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[40] = forward * mole_frac[2];
    rr_r[40] = reverse * mole_frac[1];
  }
  //  41)  OH + CH2 <=> H + CH2O
  {
    double forward = 2.0e+13;
    double xik = cgspl[1] - cgspl[3] - cgspl[9] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[41] = forward * mole_frac[3];
    rr_r[41] = reverse * mole_frac[1] * mole_frac[11];
  }
  //  42)  H2 + CH2 <=> H + CH3
  {
    double forward = 5.0e+05 * temperature * temperature * exp(-7.23e+03*ortc);
    double xik = cgspl[1] - cgspl[5] - cgspl[9] + cgspl[11];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[42] = forward * mole_frac[5];
    rr_r[42] = reverse * mole_frac[1] * mole_frac[9];
  }
  //  43)  O2 + CH2 <=> OH + HCO
  {
    double forward = 1.06e+13 * exp(-1.5e+03*ortc);
    double xik = cgspl[3] - cgspl[8] - cgspl[9] + cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[43] = forward * mole_frac[8];
    rr_r[43] = reverse * mole_frac[3];
  }
  //  44)  O2 + CH2 <=> 2 H + CO2
  {
    double forward = 2.64e+12 * exp(-1.5e+03*ortc);
    double xik = 2.0 * cgspl[1] - cgspl[8] - cgspl[9] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[44] = forward * mole_frac[8];
    rr_r[44] = reverse * mole_frac[1] * mole_frac[1] * mole_frac[13];
  }
  //  45)  HO2 + CH2 <=> OH + CH2O
  {
    double forward = 2.0e+13;
    double xik = cgspl[3] - cgspl[4] - cgspl[9] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[45] = forward * mole_frac[4];
    rr_r[45] = reverse * mole_frac[3] * mole_frac[11];
  }
  //  46)  2 CH2 <=> H2 + C2H2
  {
    double forward = 3.2e+13;
    double xik = cgspl[5] - 2.0 * cgspl[9] + cgspl[18];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[46] = forward;
    rr_r[46] = reverse * mole_frac[5] * mole_frac[14];
  }
  //  47)  CH2* + N2 <=> CH2 + N2
  {
    double forward = 1.5e+13 * exp(-600.0*ortc);
    double xik = cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[47] = forward * mole_frac[34];
    rr_r[47] = reverse * mole_frac[34];
  }
  //  48)  O + CH2* <=> H2 + CO
  {
    double forward = 1.5e+13;
    double xik = -cgspl[2] + cgspl[5] - cgspl[10] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[48] = forward * mole_frac[2];
    rr_r[48] = reverse * mole_frac[5] * mole_frac[12];
  }
  //  49)  O + CH2* <=> H + HCO
  {
    double forward = 1.5e+13;
    double xik = cgspl[1] - cgspl[2] - cgspl[10] + cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[49] = forward * mole_frac[2];
    rr_r[49] = reverse * mole_frac[1];
  }
  //  50)  OH + CH2* <=> H + CH2O
  {
    double forward = 3.0e+13;
    double xik = cgspl[1] - cgspl[3] - cgspl[10] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[50] = forward * mole_frac[3];
    rr_r[50] = reverse * mole_frac[1] * mole_frac[11];
  }
  //  51)  H2 + CH2* <=> H + CH3
  {
    double forward = 7.0e+13;
    double xik = cgspl[1] - cgspl[5] - cgspl[10] + cgspl[11];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[51] = forward * mole_frac[5];
    rr_r[51] = reverse * mole_frac[1] * mole_frac[9];
  }
  //  52)  O2 + CH2* <=> H + OH + CO
  {
    double forward = 2.8e+13;
    double xik = cgspl[1] + cgspl[3] - cgspl[8] - cgspl[10] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[52] = forward * mole_frac[8];
    rr_r[52] = reverse * mole_frac[1] * mole_frac[3] * mole_frac[12];
  }
  //  53)  O2 + CH2* <=> H2O + CO
  {
    double forward = 1.2e+13;
    double xik = cgspl[6] - cgspl[8] - cgspl[10] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[53] = forward * mole_frac[8];
    rr_r[53] = reverse * mole_frac[6] * mole_frac[12];
  }
  //  54)  H2O + CH2* <=> H2O + CH2
  {
    double forward = 3.0e+13;
    double xik = cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[54] = forward * mole_frac[6];
    rr_r[54] = reverse * mole_frac[6];
  }
  //  55)  CH2* + CO <=> CH2 + CO
  {
    double forward = 9.0e+12;
    double xik = cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[55] = forward * mole_frac[12];
    rr_r[55] = reverse * mole_frac[12];
  }
  //  56)  CH2* + CO2 <=> CH2 + CO2
  {
    double forward = 7.0e+12;
    double xik = cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[56] = forward * mole_frac[13];
    rr_r[56] = reverse * mole_frac[13];
  }
  //  57)  CH2* + CO2 <=> CH2O + CO
  {
    double forward = 1.4e+13;
    double xik = -cgspl[10] + cgspl[14] + cgspl[16] - cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[57] = forward * mole_frac[13];
    rr_r[57] = reverse * mole_frac[11] * mole_frac[12];
  }
  //  58)  H + CH2O (+M) <=> CH3O (+M)
  {
    double rr_k0 = 2.2e+30 * exp(-4.8*vlntemp - 5.56e+03*ortc);
    double rr_kinf = 5.4e+11 * exp(0.454*vlntemp - 2.6e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.242 * exp(-0.01063829787234043 * temperature) + 
      0.758 * exp(-6.430868167202572e-04 * temperature) + exp(-4.2e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[14] + cgspl[15];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[58] = forward * mole_frac[1] * mole_frac[11];
    rr_r[58] = reverse;
  }
  //  59)  H + CH2O <=> H2 + HCO
  {
    double forward = 2.3e+10 * exp(1.05*vlntemp - 3.275e+03*ortc);
    double xik = -cgspl[1] + cgspl[5] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[59] = forward * mole_frac[1] * mole_frac[11];
    rr_r[59] = reverse * mole_frac[5];
  }
  //  60)  O + CH2O <=> OH + HCO
  {
    double forward = 3.9e+13 * exp(-3.54e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[60] = forward * mole_frac[2] * mole_frac[11];
    rr_r[60] = reverse * mole_frac[3];
  }
  //  61)  OH + CH2O <=> H2O + HCO
  {
    double forward = 3.43e+09 * exp(1.18*vlntemp + 447.0*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[61] = forward * mole_frac[3] * mole_frac[11];
    rr_r[61] = reverse * mole_frac[6];
  }
  //  62)  O2 + CH2O <=> HO2 + HCO
  {
    double forward = 1.0e+14 * exp(-4.0e+04*ortc);
    double xik = cgspl[4] - cgspl[8] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[62] = forward * mole_frac[8] * mole_frac[11];
    rr_r[62] = reverse * mole_frac[4];
  }
  //  63)  HO2 + CH2O <=> H2O2 + HCO
  {
    double forward = 1.0e+12 * exp(-8.0e+03*ortc);
    double xik = -cgspl[4] + cgspl[7] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[63] = forward * mole_frac[4] * mole_frac[11];
    rr_r[63] = reverse * mole_frac[7];
  }
  //  64)  H + CH3 (+M) <=> CH4 (+M)
  {
    double rr_k0 = 2.477e+33 * exp(-4.76*vlntemp - 2.44e+03*ortc);
    double rr_kinf = 1.27e+16 * exp(-0.63*vlntemp - 383.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.217 * exp(-0.01351351351351351 * temperature) + 
      0.783 * exp(-3.400204012240735e-04 * temperature) + exp(-6.964e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[11] + cgspl[12];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[64] = forward * mole_frac[1] * mole_frac[9];
    rr_r[64] = reverse * mole_frac[10];
  }
  //  65)  O + CH3 <=> H + CH2O
  {
    double forward = 8.43e+13;
    double xik = cgspl[1] - cgspl[2] - cgspl[11] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[65] = forward * mole_frac[2] * mole_frac[9];
    rr_r[65] = reverse * mole_frac[1] * mole_frac[11];
  }
  //  66)  OH + CH3 <=> H2O + CH2
  {
    double forward = 5.6e+07 * exp(1.6*vlntemp - 5.42e+03*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[9] - cgspl[11];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[66] = forward * mole_frac[3] * mole_frac[9];
    rr_r[66] = reverse * mole_frac[6];
  }
  //  67)  OH + CH3 <=> H2O + CH2*
  {
    double forward = 2.501e+13;
    double xik = -cgspl[3] + cgspl[6] + cgspl[10] - cgspl[11];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[67] = forward * mole_frac[3] * mole_frac[9];
    rr_r[67] = reverse * mole_frac[6];
  }
  //  68)  O2 + CH3 <=> O + CH3O
  {
    double forward = 3.083e+13 * exp(-2.88e+04*ortc);
    double xik = cgspl[2] - cgspl[8] - cgspl[11] + cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[68] = forward * mole_frac[8] * mole_frac[9];
    rr_r[68] = reverse * mole_frac[2];
  }
  //  69)  O2 + CH3 <=> OH + CH2O
  {
    double forward = 3.6e+10 * exp(-8.94e+03*ortc);
    double xik = cgspl[3] - cgspl[8] - cgspl[11] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[69] = forward * mole_frac[8] * mole_frac[9];
    rr_r[69] = reverse * mole_frac[3] * mole_frac[11];
  }
  //  70)  HO2 + CH3 <=> O2 + CH4
  {
    double forward = 1.0e+12;
    double xik = -cgspl[4] + cgspl[8] - cgspl[11] + cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[70] = forward * mole_frac[4] * mole_frac[9];
    rr_r[70] = reverse * mole_frac[8] * mole_frac[10];
  }
  //  71)  HO2 + CH3 <=> OH + CH3O
  {
    double forward = 1.34e+13;
    double xik = cgspl[3] - cgspl[4] - cgspl[11] + cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[71] = forward * mole_frac[4] * mole_frac[9];
    rr_r[71] = reverse * mole_frac[3];
  }
  //  72)  H2O2 + CH3 <=> HO2 + CH4
  {
    double forward = 2.45e+04 * exp(2.47*vlntemp - 5.18e+03*ortc);
    double xik = cgspl[4] - cgspl[7] - cgspl[11] + cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[72] = forward * mole_frac[7] * mole_frac[9];
    rr_r[72] = reverse * mole_frac[4] * mole_frac[10];
  }
  //  73)  CH3 + HCO <=> CH4 + CO
  {
    double forward = 8.48e+12;
    double xik = -cgspl[11] + cgspl[12] - cgspl[13] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[73] = forward * mole_frac[9];
    rr_r[73] = reverse * mole_frac[10] * mole_frac[12];
  }
  //  74)  CH3 + CH2O <=> CH4 + HCO
  {
    double forward = 3.32e+03 * exp(2.81*vlntemp - 5.86e+03*ortc);
    double xik = -cgspl[11] + cgspl[12] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[74] = forward * mole_frac[9] * mole_frac[11];
    rr_r[74] = reverse * mole_frac[10];
  }
  //  75)  CH2 + CH3 <=> H + C2H4
  {
    double forward = 4.0e+13;
    double xik = cgspl[1] - cgspl[9] - cgspl[11] + cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[75] = forward * mole_frac[9];
    rr_r[75] = reverse * mole_frac[1] * mole_frac[15];
  }
  //  76)  CH2* + CH3 <=> H + C2H4
  {
    double forward = 1.2e+13 * exp(570.0*ortc);
    double xik = cgspl[1] - cgspl[10] - cgspl[11] + cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[76] = forward * mole_frac[9];
    rr_r[76] = reverse * mole_frac[1] * mole_frac[15];
  }
  //  77)  2 CH3 (+M) <=> C2H6 (+M)
  {
    double rr_k0 = 1.77e+50 * exp(-9.67*vlntemp - 6.22e+03*ortc);
    double rr_kinf = 2.12e+16 * exp(-0.97*vlntemp - 620.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.4675 * exp(-6.622516556291391e-03 * temperature) 
      + 0.5325 * exp(-9.633911368015414e-04 * temperature) + exp(-4.97e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -2.0 * cgspl[11] + cgspl[22];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[77] = forward * mole_frac[9] * mole_frac[9];
    rr_r[77] = reverse * mole_frac[16];
  }
  //  78)  2 CH3 <=> H + C2H5
  {
    double forward = 4.99e+12 * exp(0.1*vlntemp - 1.06e+04*ortc);
    double xik = cgspl[1] - 2.0 * cgspl[11] + cgspl[21];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[78] = forward * mole_frac[9] * mole_frac[9];
    rr_r[78] = reverse * mole_frac[1];
  }
  //  79)  H + CH3O <=> H2 + CH2O
  {
    double forward = 2.0e+13;
    double xik = -cgspl[1] + cgspl[5] + cgspl[14] - cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[79] = forward * mole_frac[1];
    rr_r[79] = reverse * mole_frac[5] * mole_frac[11];
  }
  //  80)  H + CH3O <=> OH + CH3
  {
    double forward = 3.2e+13;
    double xik = -cgspl[1] + cgspl[3] + cgspl[11] - cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[80] = forward * mole_frac[1];
    rr_r[80] = reverse * mole_frac[3] * mole_frac[9];
  }
  //  81)  H + CH3O <=> H2O + CH2*
  {
    double forward = 1.6e+13;
    double xik = -cgspl[1] + cgspl[6] + cgspl[10] - cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[81] = forward * mole_frac[1];
    rr_r[81] = reverse * mole_frac[6];
  }
  //  82)  O + CH3O <=> OH + CH2O
  {
    double forward = 1.0e+13;
    double xik = -cgspl[2] + cgspl[3] + cgspl[14] - cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[82] = forward * mole_frac[2];
    rr_r[82] = reverse * mole_frac[3] * mole_frac[11];
  }
  //  83)  OH + CH3O <=> H2O + CH2O
  {
    double forward = 5.0e+12;
    double xik = -cgspl[3] + cgspl[6] + cgspl[14] - cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[83] = forward * mole_frac[3];
    rr_r[83] = reverse * mole_frac[6] * mole_frac[11];
  }
  //  84)  O2 + CH3O <=> HO2 + CH2O
  {
    double forward = 4.28e-13 * exp(7.6*vlntemp + 3.53e+03*ortc);
    double xik = cgspl[4] - cgspl[8] + cgspl[14] - cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[84] = forward * mole_frac[8];
    rr_r[84] = reverse * mole_frac[4] * mole_frac[11];
  }
  //  85)  H + CH4 <=> H2 + CH3
  {
    double forward = 6.6e+08 * exp(1.62*vlntemp - 1.084e+04*ortc);
    double xik = -cgspl[1] + cgspl[5] + cgspl[11] - cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[85] = forward * mole_frac[1] * mole_frac[10];
    rr_r[85] = reverse * mole_frac[5] * mole_frac[9];
  }
  //  86)  O + CH4 <=> OH + CH3
  {
    double forward = 1.02e+09 * exp(1.5*vlntemp - 8.6e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[11] - cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[86] = forward * mole_frac[2] * mole_frac[10];
    rr_r[86] = reverse * mole_frac[3] * mole_frac[9];
  }
  //  87)  OH + CH4 <=> H2O + CH3
  {
    double forward = 1.0e+08 * exp(1.6*vlntemp - 3.12e+03*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[11] - cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[87] = forward * mole_frac[3] * mole_frac[10];
    rr_r[87] = reverse * mole_frac[6] * mole_frac[9];
  }
  //  88)  CH2 + CH4 <=> 2 CH3
  {
    double forward = 2.46e+06 * temperature * temperature * exp(-8.27e+03*ortc);
    double xik = -cgspl[9] + 2.0 * cgspl[11] - cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[88] = forward * mole_frac[10];
    rr_r[88] = reverse * mole_frac[9] * mole_frac[9];
  }
  //  89)  CH2* + CH4 <=> 2 CH3
  {
    double forward = 1.6e+13 * exp(570.0*ortc);
    double xik = -cgspl[10] + 2.0 * cgspl[11] - cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[89] = forward * mole_frac[10];
    rr_r[89] = reverse * mole_frac[9] * mole_frac[9];
  }
  //  90)  C2H3 (+M) <=> H + C2H2 (+M)
  {
    double rr_k0 = 2.565e+27 * exp(-3.4*vlntemp - 3.579872e+04*ortc);
    double rr_kinf = 3.86e+07 * exp(1.62*vlntemp - 3.70482e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(-0.9816 * exp(-1.857458625109126e-04 * temperature) 
      + 1.9816 * exp(-0.2329264884002609 * temperature) + exp(0.0795 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = cgspl[1] + cgspl[18] - cgspl[19];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[90] = forward;
    rr_r[90] = reverse * mole_frac[1] * mole_frac[14];
  }
  //  91)  O + C2H2 <=> CH2 + CO
  {
    double forward = 4.08e+06 * temperature * temperature * exp(-1.9e+03*ortc);
    double xik = -cgspl[2] + cgspl[9] + cgspl[16] - cgspl[18];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[91] = forward * mole_frac[2] * mole_frac[14];
    rr_r[91] = reverse * mole_frac[12];
  }
  //  92)  OH + C2H2 <=> CH3 + CO
  {
    double forward = 4.83e-04 * exp(4.0*vlntemp + 2.0e+03*ortc);
    double xik = -cgspl[3] + cgspl[11] + cgspl[16] - cgspl[18];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[92] = forward * mole_frac[3] * mole_frac[14];
    rr_r[92] = reverse * mole_frac[9] * mole_frac[12];
  }
  //  93)  HCO + C2H2 <=> CO + C2H3
  {
    double forward = 1.0e+07 * temperature * temperature * exp(-6.0e+03*ortc);
    double xik = -cgspl[13] + cgspl[16] - cgspl[18] + cgspl[19];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[93] = forward * mole_frac[14];
    rr_r[93] = reverse * mole_frac[12];
  }
  //  94)  CH3 + C2H2 <=> AC3H5
  {
    double forward = 2.68e+53 * exp(-12.82*vlntemp - 3.573e+04*ortc);
    double xik = -cgspl[11] - cgspl[18] + cgspl[24];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[94] = forward * mole_frac[9] * mole_frac[14];
    rr_r[94] = reverse * mole_frac[18];
  }
  //  95)  H + C2H3 (+M) <=> C2H4 (+M)
  {
    double rr_k0 = 1.4e+30 * exp(-3.86*vlntemp - 3.32e+03*ortc);
    double rr_kinf = 6.08e+12 * exp(0.27*vlntemp - 280.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.218 * exp(-4.819277108433735e-03 * temperature) + 
      0.782 * exp(-3.755163349605708e-04 * temperature) + exp(-6.095e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[19] + cgspl[20];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[95] = forward * mole_frac[1];
    rr_r[95] = reverse * mole_frac[15];
  }
  //  96)  H + C2H3 <=> H2 + C2H2
  {
    double forward = 9.0e+13;
    double xik = -cgspl[1] + cgspl[5] + cgspl[18] - cgspl[19];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[96] = forward * mole_frac[1];
    rr_r[96] = reverse * mole_frac[5] * mole_frac[14];
  }
  //  97)  O + C2H3 <=> CH3 + CO
  {
    double forward = 4.8e+13;
    double xik = -cgspl[2] + cgspl[11] + cgspl[16] - cgspl[19];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[97] = forward * mole_frac[2];
    rr_r[97] = reverse * mole_frac[9] * mole_frac[12];
  }
  //  98)  OH + C2H3 <=> H2O + C2H2
  {
    double forward = 3.011e+13;
    double xik = -cgspl[3] + cgspl[6] + cgspl[18] - cgspl[19];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[98] = forward * mole_frac[3];
    rr_r[98] = reverse * mole_frac[6] * mole_frac[14];
  }
  //  99)  O2 + C2H3 <=> HO2 + C2H2
  {
    double forward = 1.34e+06 * exp(1.61*vlntemp + 383.4*ortc);
    double xik = cgspl[4] - cgspl[8] + cgspl[18] - cgspl[19];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[99] = forward * mole_frac[8];
    rr_r[99] = reverse * mole_frac[4] * mole_frac[14];
  }
  // 100)  O2 + C2H3 <=> O + CH2CHO
  {
    double forward = 3.0e+11 * exp(0.29*vlntemp - 11.0*ortc);
    double xik = cgspl[2] - cgspl[8] - cgspl[19] + cgspl[23];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[100] = forward * mole_frac[8];
    rr_r[100] = reverse * mole_frac[2] * mole_frac[17];
  }
  // 101)  O2 + C2H3 <=> HCO + CH2O
  {
    double forward = 4.6e+16 * exp(-1.39*vlntemp - 1.01e+03*ortc);
    double xik = -cgspl[8] + cgspl[13] + cgspl[14] - cgspl[19];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[101] = forward * mole_frac[8];
    rr_r[101] = reverse * mole_frac[11];
  }
  // 102)  HO2 + C2H3 <=> OH + CH2CHO
  {
    double forward = 1.0e+13;
    double xik = cgspl[3] - cgspl[4] - cgspl[19] + cgspl[23];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[102] = forward * mole_frac[4];
    rr_r[102] = reverse * mole_frac[3] * mole_frac[17];
  }
  // 103)  H2O2 + C2H3 <=> HO2 + C2H4
  {
    double forward = 1.21e+10 * exp(596.0*ortc);
    double xik = cgspl[4] - cgspl[7] - cgspl[19] + cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[103] = forward * mole_frac[7];
    rr_r[103] = reverse * mole_frac[4] * mole_frac[15];
  }
  // 104)  HCO + C2H3 <=> CO + C2H4
  {
    double forward = 9.033e+13;
    double xik = -cgspl[13] + cgspl[16] - cgspl[19] + cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[104] = forward;
    rr_r[104] = reverse * mole_frac[12] * mole_frac[15];
  }
  // 105)  HCO + C2H3 <=> C2H3CHO
  {
    double forward = 1.8e+13;
    double xik = -cgspl[13] - cgspl[19] + cgspl[27];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[105] = forward;
    rr_r[105] = reverse * mole_frac[20];
  }
  // 106)  CH3 + C2H3 <=> CH4 + C2H2
  {
    double forward = 3.92e+11;
    double xik = -cgspl[11] + cgspl[12] + cgspl[18] - cgspl[19];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[106] = forward * mole_frac[9];
    rr_r[106] = reverse * mole_frac[10] * mole_frac[14];
  }
  // 107)  CH3 + C2H3 (+M) <=> C3H6 (+M)
  {
    double rr_k0 = 4.27e+58 * exp(-11.94*vlntemp - 9.769799999999999e+03*ortc);
    double rr_kinf = 2.5e+13;
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.825 * exp(-7.459346561241235e-04 * temperature) + 
      0.175 * exp(-1.666666666666667e-05 * temperature) + exp(-1.01398e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[11] - cgspl[19] + cgspl[25];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[107] = forward * mole_frac[9];
    rr_r[107] = reverse * mole_frac[19];
  }
  // 108)  CH3 + C2H3 <=> H + AC3H5
  {
    double forward = 1.5e+24 * exp(-2.83*vlntemp - 1.8618e+04*ortc);
    double xik = cgspl[1] - cgspl[11] - cgspl[19] + cgspl[24];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[108] = forward * mole_frac[9];
    rr_r[108] = reverse * mole_frac[1] * mole_frac[18];
  }
  // 109)  2 C2H3 <=> C2H2 + C2H4
  {
    double forward = 9.6e+11;
    double xik = cgspl[18] - 2.0 * cgspl[19] + cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[109] = forward;
    rr_r[109] = reverse * mole_frac[14] * mole_frac[15];
  }
  // 110)  CH2CHO <=> CH3 + CO
  {
    double forward = 7.799999999999999e+41 * exp(-9.147*vlntemp - 
      4.69e+04*ortc); 
    double xik = cgspl[11] + cgspl[16] - cgspl[23];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[110] = forward * mole_frac[17];
    rr_r[110] = reverse * mole_frac[9] * mole_frac[12];
  }
  // 111)  H + CH2CHO <=> CH3 + HCO
  {
    double forward = 9.0e+13;
    double xik = -cgspl[1] + cgspl[11] + cgspl[13] - cgspl[23];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[111] = forward * mole_frac[1] * mole_frac[17];
    rr_r[111] = reverse * mole_frac[9];
  }
  // 112)  O2 + CH2CHO <=> OH + CH2O + CO
  {
    double forward = 1.8e+10;
    double xik = cgspl[3] - cgspl[8] + cgspl[14] + cgspl[16] - cgspl[23];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[112] = forward * mole_frac[8] * mole_frac[17];
    rr_r[112] = reverse * mole_frac[3] * mole_frac[11] * mole_frac[12];
  }
  // 113)  H + C2H4 (+M) <=> C2H5 (+M)
  {
    double rr_k0 = 4.715e+18 * exp(-755.26*ortc);
    double rr_kinf = 3.975e+09 * exp(1.28*vlntemp - 1.29063e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.24 * exp(-0.025 * temperature) + 0.76 * 
      exp(-9.75609756097561e-04 * temperature),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[20] + cgspl[21];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[113] = forward * mole_frac[1] * mole_frac[15];
    rr_r[113] = reverse;
  }
  // 114)  H + C2H4 <=> H2 + C2H3
  {
    double forward = 5.07e+07 * exp(1.9*vlntemp - 1.295e+04*ortc);
    double xik = -cgspl[1] + cgspl[5] + cgspl[19] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[114] = forward * mole_frac[1] * mole_frac[15];
    rr_r[114] = reverse * mole_frac[5];
  }
  // 115)  O + C2H4 <=> OH + C2H3
  {
    double forward = 1.51e+07 * exp(1.9*vlntemp - 3.74e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[19] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[115] = forward * mole_frac[2] * mole_frac[15];
    rr_r[115] = reverse * mole_frac[3];
  }
  // 116)  O + C2H4 <=> CH3 + HCO
  {
    double forward = 1.92e+07 * exp(1.83*vlntemp - 220.0*ortc);
    double xik = -cgspl[2] + cgspl[11] + cgspl[13] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[116] = forward * mole_frac[2] * mole_frac[15];
    rr_r[116] = reverse * mole_frac[9];
  }
  // 117)  O + C2H4 <=> CH2 + CH2O
  {
    double forward = 3.84e+05 * exp(1.83*vlntemp - 220.0*ortc);
    double xik = -cgspl[2] + cgspl[9] + cgspl[14] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[117] = forward * mole_frac[2] * mole_frac[15];
    rr_r[117] = reverse * mole_frac[11];
  }
  // 118)  OH + C2H4 <=> H2O + C2H3
  {
    double forward = 3.6e+06 * temperature * temperature * exp(-2.5e+03*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[19] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[118] = forward * mole_frac[3] * mole_frac[15];
    rr_r[118] = reverse * mole_frac[6];
  }
  // 119)  HCO + C2H4 <=> CO + C2H5
  {
    double forward = 1.0e+07 * temperature * temperature * exp(-8.0e+03*ortc);
    double xik = -cgspl[13] + cgspl[16] - cgspl[20] + cgspl[21];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[119] = forward * mole_frac[15];
    rr_r[119] = reverse * mole_frac[12];
  }
  // 120)  CH2 + C2H4 <=> H + AC3H5
  {
    double forward = 2.0e+13 * exp(-6.0e+03*ortc);
    double xik = cgspl[1] - cgspl[9] - cgspl[20] + cgspl[24];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[120] = forward * mole_frac[15];
    rr_r[120] = reverse * mole_frac[1] * mole_frac[18];
  }
  // 121)  CH2* + C2H4 <=> H + AC3H5
  {
    double forward = 5.0e+13;
    double xik = cgspl[1] - cgspl[10] - cgspl[20] + cgspl[24];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[121] = forward * mole_frac[15];
    rr_r[121] = reverse * mole_frac[1] * mole_frac[18];
  }
  // 122)  CH3 + C2H4 <=> CH4 + C2H3
  {
    double forward = 2.27e+05 * temperature * temperature * exp(-9.2e+03*ortc);
    double xik = -cgspl[11] + cgspl[12] + cgspl[19] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[122] = forward * mole_frac[9] * mole_frac[15];
    rr_r[122] = reverse * mole_frac[10];
  }
  // 123)  NC3H7 <=> CH3 + C2H4
  {
    double forward = 9.6e+13 * exp(-3.102294e+04*ortc);
    double xik = cgspl[11] + cgspl[20] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[123] = forward;
    rr_r[123] = reverse * mole_frac[9] * mole_frac[15];
  }
  // 124)  O2 + C2H4 <=> HO2 + C2H3
  {
    double forward = 4.22e+13 * exp(-6.08e+04*ortc);
    double xik = cgspl[4] - cgspl[8] + cgspl[19] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[124] = forward * mole_frac[8] * mole_frac[15];
    rr_r[124] = reverse * mole_frac[4];
  }
  // 125)  C2H3 + C2H4 <=> C4H7
  {
    double forward = 7.930000000000001e+38 * exp(-8.470000000000001*vlntemp - 
      1.422e+04*ortc); 
    double xik = -cgspl[19] - cgspl[20] + cgspl[28];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[125] = forward * mole_frac[15];
    rr_r[125] = reverse * mole_frac[21];
  }
  // 126)  H + C2H5 (+M) <=> C2H6 (+M)
  {
    double rr_k0 = 1.99e+41 * exp(-7.08*vlntemp - 6.685e+03*ortc);
    double rr_kinf = 5.21e+17 * exp(-0.99*vlntemp - 1.58e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.1578000000000001 * exp(-8.0e-03 * temperature) + 
      0.8421999999999999 * exp(-4.506534474988734e-04 * temperature) + 
      exp(-6.882e+03 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[21] + cgspl[22];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[126] = forward * mole_frac[1];
    rr_r[126] = reverse * mole_frac[16];
  }
  // 127)  H + C2H5 <=> H2 + C2H4
  {
    double forward = 2.0e+12;
    double xik = -cgspl[1] + cgspl[5] + cgspl[20] - cgspl[21];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[127] = forward * mole_frac[1];
    rr_r[127] = reverse * mole_frac[5] * mole_frac[15];
  }
  // 128)  O + C2H5 <=> CH3 + CH2O
  {
    double forward = 1.604e+13;
    double xik = -cgspl[2] + cgspl[11] + cgspl[14] - cgspl[21];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[128] = forward * mole_frac[2];
    rr_r[128] = reverse * mole_frac[9] * mole_frac[11];
  }
  // 129)  O2 + C2H5 <=> HO2 + C2H4
  {
    double forward = 2.0e+10;
    double xik = cgspl[4] - cgspl[8] + cgspl[20] - cgspl[21];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[129] = forward * mole_frac[8];
    rr_r[129] = reverse * mole_frac[4] * mole_frac[15];
  }
  // 130)  HO2 + C2H5 <=> O2 + C2H6
  {
    double forward = 3.0e+11;
    double xik = -cgspl[4] + cgspl[8] - cgspl[21] + cgspl[22];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[130] = forward * mole_frac[4];
    rr_r[130] = reverse * mole_frac[8] * mole_frac[16];
  }
  // 131)  HO2 + C2H5 <=> H2O2 + C2H4
  {
    double forward = 3.0e+11;
    double xik = -cgspl[4] + cgspl[7] + cgspl[20] - cgspl[21];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[131] = forward * mole_frac[4];
    rr_r[131] = reverse * mole_frac[7] * mole_frac[15];
  }
  // 132)  HO2 + C2H5 <=> OH + CH3 + CH2O
  {
    double forward = 2.4e+13;
    double xik = cgspl[3] - cgspl[4] + cgspl[11] + cgspl[14] - cgspl[21];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[132] = forward * mole_frac[4];
    rr_r[132] = reverse * mole_frac[3] * mole_frac[9] * mole_frac[11];
  }
  // 133)  H2O2 + C2H5 <=> HO2 + C2H6
  {
    double forward = 8.7e+09 * exp(-974.0*ortc);
    double xik = cgspl[4] - cgspl[7] - cgspl[21] + cgspl[22];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[133] = forward * mole_frac[7];
    rr_r[133] = reverse * mole_frac[4] * mole_frac[16];
  }
  // 134)  C2H3 + C2H5 (+M) <=> C4H81 (+M)
  {
    double rr_k0 = 1.55e+56 * exp(-11.79*vlntemp - 8.9845e+03*ortc);
    double rr_kinf = 1.5e+13;
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.802 * exp(-4.390008341015848e-04 * temperature) + 
      0.198 * exp(-1.666666666666667e-05 * temperature) + exp(-5.7232e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[19] - cgspl[21] + cgspl[29];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[134] = forward;
    rr_r[134] = reverse * mole_frac[22];
  }
  // 135)  C2H3 + C2H5 <=> CH3 + AC3H5
  {
    double forward = 3.9e+32 * exp(-5.22*vlntemp - 1.9747e+04*ortc);
    double xik = cgspl[11] - cgspl[19] - cgspl[21] + cgspl[24];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[135] = forward;
    rr_r[135] = reverse * mole_frac[9] * mole_frac[18];
  }
  // 136)  H + C2H6 <=> H2 + C2H5
  {
    double forward = 1.15e+08 * exp(1.9*vlntemp - 7.53e+03*ortc);
    double xik = -cgspl[1] + cgspl[5] + cgspl[21] - cgspl[22];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[136] = forward * mole_frac[1] * mole_frac[16];
    rr_r[136] = reverse * mole_frac[5];
  }
  // 137)  O + C2H6 <=> OH + C2H5
  {
    double forward = 8.98e+07 * exp(1.92*vlntemp - 5.69e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[21] - cgspl[22];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[137] = forward * mole_frac[2] * mole_frac[16];
    rr_r[137] = reverse * mole_frac[3];
  }
  // 138)  OH + C2H6 <=> H2O + C2H5
  {
    double forward = 3.54e+06 * exp(2.12*vlntemp - 870.0*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[21] - cgspl[22];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[138] = forward * mole_frac[3] * mole_frac[16];
    rr_r[138] = reverse * mole_frac[6];
  }
  // 139)  CH2* + C2H6 <=> CH3 + C2H5
  {
    double forward = 4.0e+13 * exp(550.0*ortc);
    double xik = -cgspl[10] + cgspl[11] + cgspl[21] - cgspl[22];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[139] = forward * mole_frac[16];
    rr_r[139] = reverse * mole_frac[9];
  }
  // 140)  CH3 + C2H6 <=> CH4 + C2H5
  {
    double forward = 6.14e+06 * exp(1.74*vlntemp - 1.045e+04*ortc);
    double xik = -cgspl[11] + cgspl[12] + cgspl[21] - cgspl[22];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[140] = forward * mole_frac[9] * mole_frac[16];
    rr_r[140] = reverse * mole_frac[10];
  }
  // 141)  H + AC3H5 (+M) <=> C3H6 (+M)
  {
    double rr_k0 = 1.33e+60 * exp(-12.0*vlntemp - 5.9678e+03*ortc);
    double rr_kinf = 2.0e+14;
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.98 * exp(-9.119095385737736e-04 * temperature) + 
      0.02 * exp(-9.119095385737736e-04 * temperature) + exp(-6.8595e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[24] + cgspl[25];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[141] = forward * mole_frac[1] * mole_frac[18];
    rr_r[141] = reverse * mole_frac[19];
  }
  // 142)  O + AC3H5 <=> H + C2H3CHO
  {
    double forward = 6.0e+13;
    double xik = cgspl[1] - cgspl[2] - cgspl[24] + cgspl[27];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[142] = forward * mole_frac[2] * mole_frac[18];
    rr_r[142] = reverse * mole_frac[1] * mole_frac[20];
  }
  // 143)  OH + AC3H5 <=> 2 H + C2H3CHO
  {
    double forward = 4.2e+32 * exp(-5.16*vlntemp - 3.0126e+04*ortc);
    double xik = 2.0 * cgspl[1] - cgspl[3] - cgspl[24] + cgspl[27];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[143] = forward * mole_frac[3] * mole_frac[18];
    rr_r[143] = reverse * mole_frac[1] * mole_frac[1] * mole_frac[20];
  }
  // 144)  O2 + AC3H5 <=> OH + C2H3CHO
  {
    double forward = 1.82e+13 * exp(-0.41*vlntemp - 2.2859e+04*ortc);
    double xik = cgspl[3] - cgspl[8] - cgspl[24] + cgspl[27];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[144] = forward * mole_frac[8] * mole_frac[18];
    rr_r[144] = reverse * mole_frac[3] * mole_frac[20];
  }
  // 145)  HO2 + AC3H5 <=> O2 + C3H6
  {
    double forward = 2.66e+12;
    double xik = -cgspl[4] + cgspl[8] - cgspl[24] + cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[145] = forward * mole_frac[4] * mole_frac[18];
    rr_r[145] = reverse * mole_frac[8] * mole_frac[19];
  }
  // 146)  HO2 + AC3H5 <=> OH + CH2O + C2H3
  {
    double forward = 6.6e+12;
    double xik = cgspl[3] - cgspl[4] + cgspl[14] + cgspl[19] - cgspl[24];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[146] = forward * mole_frac[4] * mole_frac[18];
    rr_r[146] = reverse * mole_frac[3] * mole_frac[11];
  }
  // 147)  HCO + AC3H5 <=> CO + C3H6
  {
    double forward = 6.0e+13;
    double xik = -cgspl[13] + cgspl[16] - cgspl[24] + cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[147] = forward * mole_frac[18];
    rr_r[147] = reverse * mole_frac[12] * mole_frac[19];
  }
  // 148)  CH3 + AC3H5 (+M) <=> C4H81 (+M)
  {
    double rr_k0 = 3.91e+60 * exp(-12.81*vlntemp - 6.25e+03*ortc);
    double rr_kinf = 1.0e+14 * exp(-0.32*vlntemp + 262.3*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.896 * exp(-6.226650062266501e-04 * temperature) + 
      0.104 * exp(-1.666666666666667e-05 * temperature) + exp(-6.1184e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[11] - cgspl[24] + cgspl[29];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[148] = forward * mole_frac[9] * mole_frac[18];
    rr_r[148] = reverse * mole_frac[22];
  }
  // 149)  H + C3H6 (+M) <=> NC3H7 (+M)
  {
    double rr_k0 = 6.26e+38 * exp(-6.66*vlntemp - 7.0e+03*ortc);
    double rr_kinf = 1.33e+13 * exp(-3.2607e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.0 * exp(-1.0e-03 * temperature) + 1.0 * 
      exp(-7.633587786259542e-04 * temperature) + exp(-4.8097e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[25] + cgspl[26];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[149] = forward * mole_frac[1] * mole_frac[19];
    rr_r[149] = reverse;
  }
  // 150)  H + C3H6 <=> CH3 + C2H4
  {
    double forward = 8.0e+21 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[11] + cgspl[20] - cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[150] = forward * mole_frac[1] * mole_frac[19];
    rr_r[150] = reverse * mole_frac[9] * mole_frac[15];
  }
  // 151)  H + C3H6 <=> H2 + AC3H5
  {
    double forward = 1.73e+05 * exp(2.5*vlntemp - 2.49e+03*ortc);
    double xik = -cgspl[1] + cgspl[5] + cgspl[24] - cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[151] = forward * mole_frac[1] * mole_frac[19];
    rr_r[151] = reverse * mole_frac[5] * mole_frac[18];
  }
  // 152)  O + C3H6 <=> 2 H + C2H3CHO
  {
    double forward = 4.0e+07 * exp(1.65*vlntemp - 327.0*ortc);
    double xik = 2.0 * cgspl[1] - cgspl[2] - cgspl[25] + cgspl[27];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[152] = forward * mole_frac[2] * mole_frac[19];
    rr_r[152] = reverse * mole_frac[1] * mole_frac[1] * mole_frac[20];
  }
  // 153)  O + C3H6 <=> HCO + C2H5
  {
    double forward = 3.5e+07 * exp(1.65*vlntemp + 972.0*ortc);
    double xik = -cgspl[2] + cgspl[13] + cgspl[21] - cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[153] = forward * mole_frac[2] * mole_frac[19];
    rr_r[153] = reverse;
  }
  // 154)  O + C3H6 <=> OH + AC3H5
  {
    double forward = 1.8e+11 * exp(0.7*vlntemp - 5.88e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[24] - cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[154] = forward * mole_frac[2] * mole_frac[19];
    rr_r[154] = reverse * mole_frac[3] * mole_frac[18];
  }
  // 155)  OH + C3H6 <=> H2O + AC3H5
  {
    double forward = 3.1e+06 * temperature * temperature * exp(298.0*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[24] - cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[155] = forward * mole_frac[3] * mole_frac[19];
    rr_r[155] = reverse * mole_frac[6] * mole_frac[18];
  }
  // 156)  HO2 + C3H6 <=> H2O2 + AC3H5
  {
    double forward = 9.6e+03 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double xik = -cgspl[4] + cgspl[7] + cgspl[24] - cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[156] = forward * mole_frac[4] * mole_frac[19];
    rr_r[156] = reverse * mole_frac[7] * mole_frac[18];
  }
  // 157)  CH3 + C3H6 <=> CH4 + AC3H5
  {
    double forward = 2.2 * exp(3.5*vlntemp - 5.675e+03*ortc);
    double xik = -cgspl[11] + cgspl[12] + cgspl[24] - cgspl[25];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[157] = forward * mole_frac[9] * mole_frac[19];
    rr_r[157] = reverse * mole_frac[10] * mole_frac[18];
  }
  // 158)  H + C2H3CHO <=> HCO + C2H4
  {
    double forward = 1.08e+11 * exp(0.454*vlntemp - 5.82e+03*ortc);
    double xik = -cgspl[1] + cgspl[13] + cgspl[20] - cgspl[27];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[158] = forward * mole_frac[1] * mole_frac[20];
    rr_r[158] = reverse * mole_frac[15];
  }
  // 159)  O + C2H3CHO <=> OH + CO + C2H3
  {
    double forward = 3.0e+13 * exp(-3.54e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[16] + cgspl[19] - cgspl[27];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[159] = forward * mole_frac[2] * mole_frac[20];
    rr_r[159] = reverse * mole_frac[3] * mole_frac[12];
  }
  // 160)  OH + C2H3CHO <=> H2O + CO + C2H3
  {
    double forward = 3.43e+09 * exp(1.18*vlntemp + 447.0*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[16] + cgspl[19] - cgspl[27];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[160] = forward * mole_frac[3] * mole_frac[20];
    rr_r[160] = reverse * mole_frac[6] * mole_frac[12];
  }
  // 161)  H + NC3H7 <=> CH3 + C2H5
  {
    double forward = 3.7e+24 * exp(-2.92*vlntemp - 1.2505e+04*ortc);
    double xik = -cgspl[1] + cgspl[11] + cgspl[21] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[161] = forward * mole_frac[1];
    rr_r[161] = reverse * mole_frac[9];
  }
  // 162)  H + NC3H7 <=> H2 + C3H6
  {
    double forward = 1.8e+12;
    double xik = -cgspl[1] + cgspl[5] + cgspl[25] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[162] = forward * mole_frac[1];
    rr_r[162] = reverse * mole_frac[5] * mole_frac[19];
  }
  // 163)  O + NC3H7 <=> CH2O + C2H5
  {
    double forward = 9.6e+13;
    double xik = -cgspl[2] + cgspl[14] + cgspl[21] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[163] = forward * mole_frac[2];
    rr_r[163] = reverse * mole_frac[11];
  }
  // 164)  OH + NC3H7 <=> H2O + C3H6
  {
    double forward = 2.4e+13;
    double xik = -cgspl[3] + cgspl[6] + cgspl[25] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[164] = forward * mole_frac[3];
    rr_r[164] = reverse * mole_frac[6] * mole_frac[19];
  }
  // 165)  O2 + NC3H7 <=> HO2 + C3H6
  {
    double forward = 9.0e+10;
    double xik = cgspl[4] - cgspl[8] + cgspl[25] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[165] = forward * mole_frac[8];
    rr_r[165] = reverse * mole_frac[4] * mole_frac[19];
  }
  // 166)  HO2 + NC3H7 <=> OH + CH2O + C2H5
  {
    double forward = 2.4e+13;
    double xik = cgspl[3] - cgspl[4] + cgspl[14] + cgspl[21] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[166] = forward * mole_frac[4];
    rr_r[166] = reverse * mole_frac[3] * mole_frac[11];
  }
  // 167)  CH3 + NC3H7 <=> CH4 + C3H6
  {
    double forward = 1.1e+13;
    double xik = -cgspl[11] + cgspl[12] + cgspl[25] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[167] = forward * mole_frac[9];
    rr_r[167] = reverse * mole_frac[10] * mole_frac[19];
  }
  // 168)  H + C4H7 (+M) <=> C4H81 (+M)
  {
    double rr_k0 = 3.01e+48 * exp(-9.32*vlntemp - 5.8336e+03*ortc);
    double rr_kinf = 3.6e+13;
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.502 * exp(-7.6103500761035e-04 * temperature) + 
      0.498 * exp(-7.6103500761035e-04 * temperature) + exp(-5.0e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[28] + cgspl[29];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[168] = forward * mole_frac[1] * mole_frac[21];
    rr_r[168] = reverse * mole_frac[22];
  }
  // 169)  H + C4H7 <=> CH3 + AC3H5
  {
    double forward = 2.0e+21 * otc * otc * exp(-1.1e+04*ortc);
    double xik = -cgspl[1] + cgspl[11] + cgspl[24] - cgspl[28];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[169] = forward * mole_frac[1] * mole_frac[21];
    rr_r[169] = reverse * mole_frac[9] * mole_frac[18];
  }
  // 170)  HO2 + C4H7 <=> OH + CH2O + AC3H5
  {
    double forward = 2.4e+13;
    double xik = cgspl[3] - cgspl[4] + cgspl[14] + cgspl[24] - cgspl[28];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[170] = forward * mole_frac[4] * mole_frac[21];
    rr_r[170] = reverse * mole_frac[3] * mole_frac[11] * mole_frac[18];
  }
  // 171)  HCO + C4H7 <=> CO + C4H81
  {
    double forward = 6.0e+13;
    double xik = -cgspl[13] + cgspl[16] - cgspl[28] + cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[171] = forward * mole_frac[21];
    rr_r[171] = reverse * mole_frac[12] * mole_frac[22];
  }
  // 172)  H + C4H81 (+M) <=> PC4H9 (+M)
  {
    double rr_k0 = 6.26e+38 * exp(-6.66*vlntemp - 7.0e+03*ortc);
    double rr_kinf = 1.33e+13 * exp(-3.2607e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.0 * exp(-1.0e-03 * temperature) + 1.0 * 
      exp(-7.633587786259542e-04 * temperature) + exp(-4.8097e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[29] + cgspl[30];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[172] = forward * mole_frac[1] * mole_frac[22];
    rr_r[172] = reverse;
  }
  // 173)  H + C4H81 <=> C2H4 + C2H5
  {
    double forward = 1.6e+22 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[20] + cgspl[21] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[173] = forward * mole_frac[1] * mole_frac[22];
    rr_r[173] = reverse * mole_frac[15];
  }
  // 174)  H + C4H81 <=> CH3 + C3H6
  {
    double forward = 3.2e+22 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[11] + cgspl[25] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[174] = forward * mole_frac[1] * mole_frac[22];
    rr_r[174] = reverse * mole_frac[9] * mole_frac[19];
  }
  // 175)  H + C4H81 <=> H2 + C4H7
  {
    double forward = 6.5e+05 * exp(2.54*vlntemp - 6.756e+03*ortc);
    double xik = -cgspl[1] + cgspl[5] + cgspl[28] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[175] = forward * mole_frac[1] * mole_frac[22];
    rr_r[175] = reverse * mole_frac[5] * mole_frac[21];
  }
  // 176)  O + C4H81 <=> HCO + NC3H7
  {
    double forward = 3.3e+08 * exp(1.45*vlntemp + 402.0*ortc);
    double xik = -cgspl[2] + cgspl[13] + cgspl[26] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[176] = forward * mole_frac[2] * mole_frac[22];
    rr_r[176] = reverse;
  }
  // 177)  O + C4H81 <=> OH + C4H7
  {
    double forward = 1.5e+13 * exp(-5.76e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[28] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[177] = forward * mole_frac[2] * mole_frac[22];
    rr_r[177] = reverse * mole_frac[3] * mole_frac[21];
  }
  // 178)  O + C4H81 <=> OH + C4H7
  {
    double forward = 2.6e+13 * exp(-4.47e+03*ortc);
    double xik = -cgspl[2] + cgspl[3] + cgspl[28] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[178] = forward * mole_frac[2] * mole_frac[22];
    rr_r[178] = reverse * mole_frac[3] * mole_frac[21];
  }
  // 179)  OH + C4H81 <=> H2O + C4H7
  {
    double forward = 700.0 * exp(2.66*vlntemp - 527.0*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[28] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[179] = forward * mole_frac[3] * mole_frac[22];
    rr_r[179] = reverse * mole_frac[6] * mole_frac[21];
  }
  // 180)  O2 + C4H81 <=> HO2 + C4H7
  {
    double forward = 2.0e+13 * exp(-5.093e+04*ortc);
    double xik = cgspl[4] - cgspl[8] + cgspl[28] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[180] = forward * mole_frac[8] * mole_frac[22];
    rr_r[180] = reverse * mole_frac[4] * mole_frac[21];
  }
  // 181)  HO2 + C4H81 <=> H2O2 + C4H7
  {
    double forward = 1.0e+12 * exp(-1.434e+04*ortc);
    double xik = -cgspl[4] + cgspl[7] + cgspl[28] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[181] = forward * mole_frac[4] * mole_frac[22];
    rr_r[181] = reverse * mole_frac[7] * mole_frac[21];
  }
  // 182)  CH3 + C4H81 <=> CH4 + C4H7
  {
    double forward = 0.45 * exp(3.65*vlntemp - 7.153e+03*ortc);
    double xik = -cgspl[11] + cgspl[12] + cgspl[28] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[182] = forward * mole_frac[9] * mole_frac[22];
    rr_r[182] = reverse * mole_frac[10] * mole_frac[21];
  }
  // 183)  H + PC4H9 <=> 2 C2H5
  {
    double forward = 3.7e+24 * exp(-2.92*vlntemp - 1.2505e+04*ortc);
    double xik = -cgspl[1] + 2.0 * cgspl[21] - cgspl[30];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[183] = forward * mole_frac[1];
    rr_r[183] = reverse;
  }
  // 184)  H + PC4H9 <=> H2 + C4H81
  {
    double forward = 1.8e+12;
    double xik = -cgspl[1] + cgspl[5] + cgspl[29] - cgspl[30];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[184] = forward * mole_frac[1];
    rr_r[184] = reverse * mole_frac[5] * mole_frac[22];
  }
  // 185)  O + PC4H9 <=> CH2O + NC3H7
  {
    double forward = 9.6e+13;
    double xik = -cgspl[2] + cgspl[14] + cgspl[26] - cgspl[30];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[185] = forward * mole_frac[2];
    rr_r[185] = reverse * mole_frac[11];
  }
  // 186)  OH + PC4H9 <=> H2O + C4H81
  {
    double forward = 2.4e+13;
    double xik = -cgspl[3] + cgspl[6] + cgspl[29] - cgspl[30];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[186] = forward * mole_frac[3];
    rr_r[186] = reverse * mole_frac[6] * mole_frac[22];
  }
  // 187)  O2 + PC4H9 <=> HO2 + C4H81
  {
    double forward = 2.7e+11;
    double xik = cgspl[4] - cgspl[8] + cgspl[29] - cgspl[30];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[187] = forward * mole_frac[8];
    rr_r[187] = reverse * mole_frac[4] * mole_frac[22];
  }
  // 188)  HO2 + PC4H9 <=> OH + CH2O + NC3H7
  {
    double forward = 2.4e+13;
    double xik = cgspl[3] - cgspl[4] + cgspl[14] + cgspl[26] - cgspl[30];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[188] = forward * mole_frac[4];
    rr_r[188] = reverse * mole_frac[3] * mole_frac[11];
  }
  // 189)  CH3 + PC4H9 <=> CH4 + C4H81
  {
    double forward = 1.1e+13;
    double xik = -cgspl[11] + cgspl[12] + cgspl[29] - cgspl[30];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[189] = forward * mole_frac[9];
    rr_r[189] = reverse * mole_frac[10] * mole_frac[22];
  }
  // 190)  C5H9 => C2H4 + AC3H5
  {
    double forward = 2.5e+13 * exp(-3.001912e+04*ortc);
    rr_f[190] = forward * mole_frac[23];
    rr_r[190] = 0.0;
  }
  // 191)  C5H9 => C2H3 + C3H6
  {
    double forward = 2.5e+13 * exp(-3.001912e+04*ortc);
    rr_f[191] = forward * mole_frac[23];
    rr_r[191] = 0.0;
  }
  // 192)  H + C5H10 (+M) <=> PXC5H11 (+M)
  {
    double rr_k0 = 6.26e+38 * exp(-6.66*vlntemp - 7.0e+03*ortc);
    double rr_kinf = 1.33e+13 * exp(-3.2607e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.0 * exp(-1.0e-03 * temperature) + 1.0 * 
      exp(-7.633587786259542e-04 * temperature) + exp(-4.8097e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[32] + cgspl[33];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[192] = forward * mole_frac[1] * mole_frac[24];
    rr_r[192] = reverse;
  }
  // 193)  H + C5H10 <=> C2H4 + NC3H7
  {
    double forward = 8.0e+21 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[20] + cgspl[26] - cgspl[32];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[193] = forward * mole_frac[1] * mole_frac[24];
    rr_r[193] = reverse * mole_frac[15];
  }
  // 194)  H + C5H10 <=> C2H5 + C3H6
  {
    double forward = 1.6e+22 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[21] + cgspl[25] - cgspl[32];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[194] = forward * mole_frac[1] * mole_frac[24];
    rr_r[194] = reverse * mole_frac[19];
  }
  // 195)  C2H4 + NC3H7 <=> PXC5H11
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[20] - cgspl[26] + cgspl[33];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[195] = forward * mole_frac[15];
    rr_r[195] = reverse;
  }
  // 196)  H + C6H12 (+M) <=> PXC6H13 (+M)
  {
    double rr_k0 = 6.26e+38 * exp(-6.66*vlntemp - 7.0e+03*ortc);
    double rr_kinf = 1.33e+13 * exp(-3.2607e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.0 * exp(-1.0e-03 * temperature) + 1.0 * 
      exp(-7.633587786259542e-04 * temperature) + exp(-4.8097e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[34] + cgspl[35];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[196] = forward * mole_frac[1] * mole_frac[25];
    rr_r[196] = reverse;
  }
  // 197)  H + C6H12 <=> C2H4 + PC4H9
  {
    double forward = 8.0e+21 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[20] + cgspl[30] - cgspl[34];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[197] = forward * mole_frac[1] * mole_frac[25];
    rr_r[197] = reverse * mole_frac[15];
  }
  // 198)  H + C6H12 <=> C3H6 + NC3H7
  {
    double forward = 1.6e+22 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[25] + cgspl[26] - cgspl[34];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[198] = forward * mole_frac[1] * mole_frac[25];
    rr_r[198] = reverse * mole_frac[19];
  }
  // 199)  C2H4 + PC4H9 <=> PXC6H13
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[20] - cgspl[30] + cgspl[35];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[199] = forward * mole_frac[15];
    rr_r[199] = reverse;
  }
  // 200)  H + C7H14 (+M) <=> PXC7H15 (+M)
  {
    double rr_k0 = 6.26e+38 * exp(-6.66*vlntemp - 7.0e+03*ortc);
    double rr_kinf = 1.33e+13 * exp(-3.2607e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.0 * exp(-1.0e-03 * temperature) + 1.0 * 
      exp(-7.633587786259542e-04 * temperature) + exp(-4.8097e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[36] + cgspl[37];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[200] = forward * mole_frac[1] * mole_frac[26];
    rr_r[200] = reverse;
  }
  // 201)  H + C7H14 <=> C2H4 + PXC5H11
  {
    double forward = 8.0e+21 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[20] + cgspl[33] - cgspl[36];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[201] = forward * mole_frac[1] * mole_frac[26];
    rr_r[201] = reverse * mole_frac[15];
  }
  // 202)  H + C7H14 <=> C3H6 + PC4H9
  {
    double forward = 1.6e+22 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[25] + cgspl[30] - cgspl[36];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[202] = forward * mole_frac[1] * mole_frac[26];
    rr_r[202] = reverse * mole_frac[19];
  }
  // 203)  C2H4 + PXC5H11 <=> PXC7H15
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[20] - cgspl[33] + cgspl[37];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[203] = forward * mole_frac[15];
    rr_r[203] = reverse;
  }
  // 204)  H + C8H16 (+M) <=> PXC8H17 (+M)
  {
    double rr_k0 = 6.26e+38 * exp(-6.66*vlntemp - 7.0e+03*ortc);
    double rr_kinf = 1.33e+13 * exp(-3.2607e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.0 * exp(-1.0e-03 * temperature) + 1.0 * 
      exp(-7.633587786259542e-04 * temperature) + exp(-4.8097e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[38] + cgspl[39];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[204] = forward * mole_frac[1] * mole_frac[27];
    rr_r[204] = reverse;
  }
  // 205)  H + C8H16 <=> C2H4 + PXC6H13
  {
    double forward = 8.0e+21 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[20] + cgspl[35] - cgspl[38];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[205] = forward * mole_frac[1] * mole_frac[27];
    rr_r[205] = reverse * mole_frac[15];
  }
  // 206)  H + C8H16 <=> C3H6 + PXC5H11
  {
    double forward = 1.6e+22 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[25] + cgspl[33] - cgspl[38];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[206] = forward * mole_frac[1] * mole_frac[27];
    rr_r[206] = reverse * mole_frac[19];
  }
  // 207)  C2H4 + PXC6H13 <=> PXC8H17
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[20] - cgspl[35] + cgspl[39];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[207] = forward * mole_frac[15];
    rr_r[207] = reverse;
  }
  // 208)  H + C9H18 (+M) <=> PXC9H19 (+M)
  {
    double rr_k0 = 6.26e+38 * exp(-6.66*vlntemp - 7.0e+03*ortc);
    double rr_kinf = 1.33e+13 * exp(-3.2607e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.0 * exp(-1.0e-03 * temperature) + 1.0 * 
      exp(-7.633587786259542e-04 * temperature) + exp(-4.8097e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[40] + cgspl[41];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[208] = forward * mole_frac[1] * mole_frac[28];
    rr_r[208] = reverse * mole_frac[29];
  }
  // 209)  H + C9H18 <=> C2H4 + PXC7H15
  {
    double forward = 8.0e+21 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[20] + cgspl[37] - cgspl[40];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[209] = forward * mole_frac[1] * mole_frac[28];
    rr_r[209] = reverse * mole_frac[15];
  }
  // 210)  H + C9H18 <=> C3H6 + PXC6H13
  {
    double forward = 1.6e+22 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[25] + cgspl[35] - cgspl[40];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[210] = forward * mole_frac[1] * mole_frac[28];
    rr_r[210] = reverse * mole_frac[19];
  }
  // 211)  C2H4 + PXC7H15 <=> PXC9H19
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[20] - cgspl[37] + cgspl[41];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[211] = forward * mole_frac[15];
    rr_r[211] = reverse * mole_frac[29];
  }
  // 212)  H + C10H20 (+M) <=> PXC10H21 (+M)
  {
    double rr_k0 = 6.26e+38 * exp(-6.66*vlntemp - 7.0e+03*ortc);
    double rr_kinf = 1.33e+13 * exp(-3.2607e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[7];
    double fcent = log10(MAX(0.0 * exp(-1.0e-03 * temperature) + 1.0 * 
      exp(-7.633587786259542e-04 * temperature) + exp(-4.8097e+04 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[42] + cgspl[43];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[212] = forward * mole_frac[1] * mole_frac[30];
    rr_r[212] = reverse;
  }
  // 213)  H + C10H20 <=> C2H4 + PXC8H17
  {
    double forward = 8.0e+21 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[20] + cgspl[39] - cgspl[42];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[213] = forward * mole_frac[1] * mole_frac[30];
    rr_r[213] = reverse * mole_frac[15];
  }
  // 214)  H + C10H20 <=> C3H6 + PXC7H15
  {
    double forward = 1.6e+22 * exp(-2.39*vlntemp - 1.118e+04*ortc);
    double xik = -cgspl[1] + cgspl[25] + cgspl[37] - cgspl[42];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[214] = forward * mole_frac[1] * mole_frac[30];
    rr_r[214] = reverse * mole_frac[19];
  }
  // 215)  C2H4 + PXC8H17 <=> PXC10H21
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[20] - cgspl[39] + cgspl[43];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[215] = forward * mole_frac[15];
    rr_r[215] = reverse;
  }
  // 216)  C12H24 <=> C5H9 + PXC7H15
  {
    double forward = 3.5e+16 * exp(-7.093689999999999e+04*ortc);
    double xik = cgspl[31] + cgspl[37] - cgspl[47];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[216] = forward * mole_frac[31];
    rr_r[216] = reverse * mole_frac[23];
  }
  // 217)  C2H4 + PXC10H21 <=> PXC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[20] - cgspl[43] + cgspl[44];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[217] = forward * mole_frac[15];
    rr_r[217] = reverse;
  }
  // 218)  PXC12H25 <=> S3XC12H25
  {
    double forward = 3.67e+12 * exp(-0.6*vlntemp - 1.44e+04*ortc);
    double xik = -cgspl[44] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[218] = forward;
    rr_r[218] = reverse;
  }
  // 219)  C3H6 + PXC9H19 <=> SXC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[25] - cgspl[41] + cgspl[45];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[219] = forward * mole_frac[19] * mole_frac[29];
    rr_r[219] = reverse;
  }
  // 220)  C4H81 + PXC8H17 <=> SXC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[29] - cgspl[39] + cgspl[45];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[220] = forward * mole_frac[22];
    rr_r[220] = reverse;
  }
  // 221)  C5H10 + PXC7H15 <=> S3XC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[32] - cgspl[37] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[221] = forward * mole_frac[24];
    rr_r[221] = reverse;
  }
  // 222)  C2H5 + C10H20 <=> S3XC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[21] - cgspl[42] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[222] = forward * mole_frac[30];
    rr_r[222] = reverse;
  }
  // 223)  C6H12 + PXC6H13 <=> S3XC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[34] - cgspl[35] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[223] = forward * mole_frac[25];
    rr_r[223] = reverse;
  }
  // 224)  NC3H7 + C9H18 <=> S3XC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[26] - cgspl[40] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[224] = forward * mole_frac[28];
    rr_r[224] = reverse;
  }
  // 225)  PXC5H11 + C7H14 <=> S3XC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[33] - cgspl[36] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[225] = forward * mole_frac[26];
    rr_r[225] = reverse;
  }
  // 226)  PC4H9 + C8H16 <=> S3XC12H25
  {
    double forward = 3.0e+11 * exp(-7.3e+03*ortc);
    double xik = -cgspl[30] - cgspl[38] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[226] = forward * mole_frac[27];
    rr_r[226] = reverse;
  }
  // 227)  C2H5 + PXC10H21 <=> NC12H26
  {
    double forward = 1.88e+14 * exp(-0.5 * vlntemp);
    double xik = cgspl[0] - cgspl[21] - cgspl[43];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[227] = forward;
    rr_r[227] = reverse * mole_frac[0];
  }
  // 228)  NC3H7 + PXC9H19 <=> NC12H26
  {
    double forward = 1.88e+14 * exp(-0.5 * vlntemp);
    double xik = cgspl[0] - cgspl[26] - cgspl[41];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[228] = forward * mole_frac[29];
    rr_r[228] = reverse * mole_frac[0];
  }
  // 229)  PC4H9 + PXC8H17 <=> NC12H26
  {
    double forward = 1.88e+14 * exp(-0.5 * vlntemp);
    double xik = cgspl[0] - cgspl[30] - cgspl[39];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[229] = forward;
    rr_r[229] = reverse * mole_frac[0];
  }
  // 230)  PXC5H11 + PXC7H15 <=> NC12H26
  {
    double forward = 1.88e+14 * exp(-0.5 * vlntemp);
    double xik = cgspl[0] - cgspl[33] - cgspl[37];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[230] = forward;
    rr_r[230] = reverse * mole_frac[0];
  }
  // 231)  2 PXC6H13 <=> NC12H26
  {
    double forward = 1.88e+14 * exp(-0.5 * vlntemp);
    double xik = cgspl[0] - 2.0 * cgspl[35];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[231] = forward;
    rr_r[231] = reverse * mole_frac[0];
  }
  // 232)  NC12H26 + H <=> H2 + PXC12H25
  {
    double forward = 1.3e+06 * exp(2.54*vlntemp - 6.756e+03*ortc);
    double xik = -cgspl[0] - cgspl[1] + cgspl[5] + cgspl[44];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[232] = forward * mole_frac[0] * mole_frac[1];
    rr_r[232] = reverse * mole_frac[5];
  }
  // 233)  NC12H26 + H <=> H2 + SXC12H25
  {
    double forward = 2.6e+06 * exp(2.4*vlntemp - 4.471e+03*ortc);
    double xik = -cgspl[0] - cgspl[1] + cgspl[5] + cgspl[45];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[233] = forward * mole_frac[0] * mole_frac[1];
    rr_r[233] = reverse * mole_frac[5];
  }
  // 234)  NC12H26 + H <=> H2 + S3XC12H25
  {
    double forward = 3.9e+06 * exp(2.4*vlntemp - 4.471e+03*ortc);
    double xik = -cgspl[0] - cgspl[1] + cgspl[5] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[234] = forward * mole_frac[0] * mole_frac[1];
    rr_r[234] = reverse * mole_frac[5];
  }
  // 235)  NC12H26 + O <=> OH + PXC12H25
  {
    double forward = 1.9e+05 * exp(2.68*vlntemp - 3.716e+03*ortc);
    double xik = -cgspl[0] - cgspl[2] + cgspl[3] + cgspl[44];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[235] = forward * mole_frac[0] * mole_frac[2];
    rr_r[235] = reverse * mole_frac[3];
  }
  // 236)  NC12H26 + O <=> OH + SXC12H25
  {
    double forward = 9.52e+04 * exp(2.71*vlntemp - 2.106e+03*ortc);
    double xik = -cgspl[0] - cgspl[2] + cgspl[3] + cgspl[45];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[236] = forward * mole_frac[0] * mole_frac[2];
    rr_r[236] = reverse * mole_frac[3];
  }
  // 237)  NC12H26 + O <=> OH + S3XC12H25
  {
    double forward = 1.428e+05 * exp(2.71*vlntemp - 2.106e+03*ortc);
    double xik = -cgspl[0] - cgspl[2] + cgspl[3] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[237] = forward * mole_frac[0] * mole_frac[2];
    rr_r[237] = reverse * mole_frac[3];
  }
  // 238)  NC12H26 + OH <=> H2O + PXC12H25
  {
    double forward = 3.4e+03 * exp(2.66*vlntemp - 527.0*ortc);
    double xik = -cgspl[0] - cgspl[3] + cgspl[6] + cgspl[44];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[238] = forward * mole_frac[0] * mole_frac[3];
    rr_r[238] = reverse * mole_frac[6];
  }
  // 239)  NC12H26 + OH <=> H2O + SXC12H25
  {
    double forward = 7.4e+04 * exp(2.39*vlntemp - 393.0*ortc);
    double xik = -cgspl[0] - cgspl[3] + cgspl[6] + cgspl[45];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[239] = forward * mole_frac[0] * mole_frac[3];
    rr_r[239] = reverse * mole_frac[6];
  }
  // 240)  NC12H26 + OH <=> H2O + S3XC12H25
  {
    double forward = 1.01e+05 * exp(2.39*vlntemp - 393.0*ortc);
    double xik = -cgspl[0] - cgspl[3] + cgspl[6] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[240] = forward * mole_frac[0] * mole_frac[3];
    rr_r[240] = reverse * mole_frac[6];
  }
  // 241)  NC12H26 + O2 <=> HO2 + PXC12H25
  {
    double forward = 4.0e+13 * exp(-5.093e+04*ortc);
    double xik = -cgspl[0] + cgspl[4] - cgspl[8] + cgspl[44];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[241] = forward * mole_frac[0] * mole_frac[8];
    rr_r[241] = reverse * mole_frac[4];
  }
  // 242)  NC12H26 + O2 <=> HO2 + SXC12H25
  {
    double forward = 8.0e+13 * exp(-4.759e+04*ortc);
    double xik = -cgspl[0] + cgspl[4] - cgspl[8] + cgspl[45];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[242] = forward * mole_frac[0] * mole_frac[8];
    rr_r[242] = reverse * mole_frac[4];
  }
  // 243)  NC12H26 + O2 <=> HO2 + S3XC12H25
  {
    double forward = 1.2e+14 * exp(-4.759e+04*ortc);
    double xik = -cgspl[0] + cgspl[4] - cgspl[8] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[243] = forward * mole_frac[0] * mole_frac[8];
    rr_r[243] = reverse * mole_frac[4];
  }
  // 244)  NC12H26 + HO2 <=> H2O2 + PXC12H25
  {
    double forward = 6.76e+04 * exp(2.55*vlntemp - 1.649e+04*ortc);
    double xik = -cgspl[0] - cgspl[4] + cgspl[7] + cgspl[44];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[244] = forward * mole_frac[0] * mole_frac[4];
    rr_r[244] = reverse * mole_frac[7];
  }
  // 245)  NC12H26 + HO2 <=> H2O2 + SXC12H25
  {
    double forward = 8.9e+04 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double xik = -cgspl[0] - cgspl[4] + cgspl[7] + cgspl[45];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[245] = forward * mole_frac[0] * mole_frac[4];
    rr_r[245] = reverse * mole_frac[7];
  }
  // 246)  NC12H26 + HO2 <=> H2O2 + S3XC12H25
  {
    double forward = 8.85e+04 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double xik = -cgspl[0] - cgspl[4] + cgspl[7] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[246] = forward * mole_frac[0] * mole_frac[4];
    rr_r[246] = reverse * mole_frac[7];
  }
  // 247)  NC12H26 + CH3 <=> CH4 + PXC12H25
  {
    double forward = 1.81 * exp(3.65*vlntemp - 7.153e+03*ortc);
    double xik = -cgspl[0] - cgspl[11] + cgspl[12] + cgspl[44];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[247] = forward * mole_frac[0] * mole_frac[9];
    rr_r[247] = reverse * mole_frac[10];
  }
  // 248)  NC12H26 + CH3 <=> CH4 + SXC12H25
  {
    double forward = 6.0 * exp(3.46*vlntemp - 5.48e+03*ortc);
    double xik = -cgspl[0] - cgspl[11] + cgspl[12] + cgspl[45];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[248] = forward * mole_frac[0] * mole_frac[9];
    rr_r[248] = reverse * mole_frac[10];
  }
  // 249)  NC12H26 + CH3 <=> CH4 + S3XC12H25
  {
    double forward = 9.0 * exp(3.46*vlntemp - 5.48e+03*ortc);
    double xik = -cgspl[0] - cgspl[11] + cgspl[12] + cgspl[46];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[249] = forward * mole_frac[0] * mole_frac[9];
    rr_r[249] = reverse * mole_frac[10];
  }
  // 250)  O2 + PXC12H25 => C12H25O2
  {
    double forward = 5.0e+13;
    rr_f[250] = forward * mole_frac[8];
    rr_r[250] = 0.0;
  }
  // 251)  C12H25O2 => O2 + PXC12H25
  {
    double forward = 2.75e+13 * exp(-2.74e+04*ortc);
    rr_f[251] = forward * mole_frac[32];
    rr_r[251] = 0.0;
  }
  // 252)  O2 + SXC12H25 => C12H25O2
  {
    double forward = 5.0e+13;
    rr_f[252] = forward * mole_frac[8];
    rr_r[252] = 0.0;
  }
  // 253)  C12H25O2 => O2 + SXC12H25
  {
    double forward = 2.75e+13 * exp(-2.74e+04*ortc);
    rr_f[253] = forward * mole_frac[32];
    rr_r[253] = 0.0;
  }
  // 254)  O2 + S3XC12H25 => C12H25O2
  {
    double forward = 5.0e+13;
    rr_f[254] = forward * mole_frac[8];
    rr_r[254] = 0.0;
  }
  // 255)  C12H25O2 => O2 + S3XC12H25
  {
    double forward = 2.75e+13 * exp(-2.74e+04*ortc);
    rr_f[255] = forward * mole_frac[32];
    rr_r[255] = 0.0;
  }
  // 256)  C12H25O2 => C12OOH
  {
    double forward = 1.51e+12 * exp(-1.9e+04*ortc);
    rr_f[256] = forward * mole_frac[32];
    rr_r[256] = 0.0;
  }
  // 257)  C12OOH => C12H25O2
  {
    double forward = 1.0e+11 * exp(-1.15e+04*ortc);
    rr_f[257] = forward;
    rr_r[257] = 0.0;
  }
  // 258)  O2 + PXC12H25 => HO2 + C12H24
  {
    double forward = 3.5e+11 * exp(-6.0e+03*ortc);
    rr_f[258] = forward * mole_frac[8];
    rr_r[258] = 0.0;
  }
  // 259)  HO2 + C12H24 => O2 + PXC12H25
  {
    double forward = 3.16e+11 * exp(-1.95e+04*ortc);
    rr_f[259] = forward * mole_frac[4] * mole_frac[31];
    rr_r[259] = 0.0;
  }
  // 260)  O2 + SXC12H25 => HO2 + C12H24
  {
    double forward = 3.5e+11 * exp(-6.0e+03*ortc);
    rr_f[260] = forward * mole_frac[8];
    rr_r[260] = 0.0;
  }
  // 261)  HO2 + C12H24 => O2 + SXC12H25
  {
    double forward = 3.16e+11 * exp(-1.95e+04*ortc);
    rr_f[261] = forward * mole_frac[4] * mole_frac[31];
    rr_r[261] = 0.0;
  }
  // 262)  O2 + S3XC12H25 => HO2 + C12H24
  {
    double forward = 3.5e+11 * exp(-6.0e+03*ortc);
    rr_f[262] = forward * mole_frac[8];
    rr_r[262] = 0.0;
  }
  // 263)  HO2 + C12H24 => O2 + S3XC12H25
  {
    double forward = 3.16e+11 * exp(-1.95e+04*ortc);
    rr_f[263] = forward * mole_frac[4] * mole_frac[31];
    rr_r[263] = 0.0;
  }
  // 264)  O2 + C12OOH => O2C12H24OOH
  {
    double forward = 4.6e+10;
    rr_f[264] = forward * mole_frac[8];
    rr_r[264] = 0.0;
  }
  // 265)  O2C12H24OOH => O2 + C12OOH
  {
    double forward = 2.51e+13 * exp(-2.74e+04*ortc);
    rr_f[265] = forward;
    rr_r[265] = 0.0;
  }
  // 266)  O2C12H24OOH <=> OH + OC12H23OOH
  {
    double forward = 8.9e+10 * exp(-1.7e+04*ortc);
    double xik = cgspl[3] - cgspl[50] + cgspl[51];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[266] = forward;
    rr_r[266] = reverse * mole_frac[3] * mole_frac[33];
  }
  // 267)  OC12H23OOH => OH + 3 C2H4 + C2H5 + 2 CH2CHO
  {
    double forward = 1.8e+15 * exp(-4.2065e+04*ortc);
    rr_f[267] = forward * mole_frac[33];
    rr_r[267] = 0.0;
  }
  // Unimportant reaction rates
  rr_f[46] = 0.0;
  rr_f[104] = 0.0;
  rr_f[105] = 0.0;
  rr_f[109] = 0.0;
  rr_f[134] = 0.0;
  rr_f[135] = 0.0;
  rr_f[227] = 0.0;
  rr_f[229] = 0.0;
  rr_f[230] = 0.0;
  rr_f[231] = 0.0;
  rr_r[153] = 0.0;
  rr_r[176] = 0.0;
  rr_r[183] = 0.0;
  // QSSA connected component
  {
    double a1_0, a1_2, a1_3;
    {
      double den = rr_f[39] + rr_f[40] + rr_f[41] + rr_f[42] + rr_f[43] + 
        rr_f[44] + rr_f[45] + rr_f[75] + rr_f[88] + rr_f[120] + rr_r[47] + 
        rr_r[54] + rr_r[55] + rr_r[56] + rr_r[66] + rr_r[91] + rr_r[117]; 
      a1_0 = (rr_f[66] + rr_f[91] + rr_f[117] + rr_r[39] + rr_r[41] + rr_r[42] + 
        rr_r[44] + rr_r[45] + rr_r[46] + rr_r[46] + rr_r[75] + rr_r[88] + 
        rr_r[120])/den; 
      a1_2 = (rr_f[47] + rr_f[54] + rr_f[55] + rr_f[56])/den;
      a1_3 = (rr_r[40] + rr_r[43])/den;
    }
    double a2_0, a2_1, a2_3, a2_4, a2_6;
    {
      double den = rr_f[47] + rr_f[48] + rr_f[49] + rr_f[50] + rr_f[51] + 
        rr_f[52] + rr_f[53] + rr_f[54] + rr_f[55] + rr_f[56] + rr_f[57] + 
        rr_f[76] + rr_f[89] + rr_f[121] + rr_f[139] + rr_r[67] + rr_r[81]; 
      a2_0 = (rr_f[67] + rr_r[48] + rr_r[50] + rr_r[51] + rr_r[52] + rr_r[53] + 
        rr_r[57] + rr_r[76] + rr_r[89] + rr_r[121])/den; 
      a2_1 = (rr_r[47] + rr_r[54] + rr_r[55] + rr_r[56])/den;
      a2_3 = (rr_r[49])/den;
      a2_4 = (rr_f[81])/den;
      a2_6 = (rr_r[139])/den;
    }
    double a3_0, a3_1, a3_2, a3_5, a3_6;
    {
      double den = rr_f[30] + rr_f[31] + rr_f[32] + rr_f[33] + rr_f[34] + 
        rr_f[35] + rr_f[36] + rr_f[38] + rr_f[73] + rr_f[93] + rr_f[119] + 
        rr_f[147] + rr_f[171] + rr_r[63] + rr_r[40] + rr_r[43] + rr_r[49] + 
        rr_r[59] + rr_r[60] + rr_r[61] + rr_r[62] + rr_r[74] + rr_r[101] + 
        rr_r[111] + rr_r[116] + rr_r[158]; 
      a3_0 = (rr_f[63] + rr_f[59] + rr_f[60] + rr_f[61] + rr_f[62] + rr_f[74] + 
        rr_f[111] + rr_f[116] + rr_f[153] + rr_f[158] + rr_f[176] + rr_r[30] + 
        rr_r[31] + rr_r[32] + rr_r[33] + rr_r[34] + rr_r[35] + rr_r[36] + 
        rr_r[38] + rr_r[73] + rr_r[104] + rr_r[105] + rr_r[147] + 
        rr_r[171])/den; 
      a3_1 = (rr_f[40] + rr_f[43])/den;
      a3_2 = (rr_f[49])/den;
      a3_5 = (rr_f[101] + rr_r[93])/den;
      a3_6 = (rr_r[119])/den;
    }
    double a4_0, a4_2;
    {
      double den = rr_f[79] + rr_f[80] + rr_f[81] + rr_f[82] + rr_f[83] + 
        rr_f[84] + rr_r[58] + rr_r[68] + rr_r[71]; 
      a4_0 = (rr_f[58] + rr_f[68] + rr_f[71] + rr_r[79] + rr_r[80] + rr_r[82] + 
        rr_r[83] + rr_r[84])/den; 
      a4_2 = (rr_r[81])/den;
    }
    double a5_0, a5_3;
    {
      double den = rr_f[90] + rr_f[95] + rr_f[96] + rr_f[97] + rr_f[98] + 
        rr_f[99] + rr_f[100] + rr_f[101] + rr_f[102] + rr_f[103] + rr_f[106] + 
        rr_f[107] + rr_f[108] + rr_f[125] + rr_r[93] + rr_r[114] + rr_r[115] + 
        rr_r[118] + rr_r[122] + rr_r[124] + rr_r[146] + rr_r[159] + rr_r[160] + 
        rr_r[191]; 
      a5_0 = (rr_f[114] + rr_f[115] + rr_f[118] + rr_f[122] + rr_f[124] + 
        rr_f[146] + rr_f[159] + rr_f[160] + rr_f[191] + rr_r[90] + rr_r[95] + 
        rr_r[96] + rr_r[97] + rr_r[98] + rr_r[99] + rr_r[100] + rr_r[102] + 
        rr_r[103] + rr_r[104] + rr_r[105] + rr_r[106] + rr_r[107] + rr_r[108] + 
        rr_r[109] + rr_r[109] + rr_r[125] + rr_r[134] + rr_r[135])/den; 
      a5_3 = (rr_f[93] + rr_r[101])/den;
    }
    double a6_0, a6_2, a6_3, a6_7, a6_8, a6_16;
    {
      double den = rr_f[129] + rr_f[130] + rr_f[131] + rr_f[126] + rr_f[127] + 
        rr_f[128] + rr_f[132] + rr_f[133] + rr_f[222] + rr_r[78] + rr_r[113] + 
        rr_r[119] + rr_r[267] + rr_r[136] + rr_r[137] + rr_r[138] + rr_r[139] + 
        rr_r[140] + rr_r[161] + rr_r[163] + rr_r[166] + rr_r[173] + rr_r[194]; 
      a6_0 = (rr_f[78] + rr_f[113] + rr_f[267] + rr_f[136] + rr_f[137] + 
        rr_f[138] + rr_f[140] + rr_f[153] + rr_f[173] + rr_f[194] + rr_r[129] + 
        rr_r[130] + rr_r[131] + rr_r[126] + rr_r[127] + rr_r[128] + rr_r[132] + 
        rr_r[133] + rr_r[134] + rr_r[135] + rr_r[227])/den; 
      a6_2 = (rr_f[139])/den;
      a6_3 = (rr_f[119])/den;
      a6_7 = (rr_f[161] + rr_f[163] + rr_f[166])/den;
      a6_8 = (rr_f[183] + rr_f[183])/den;
      a6_16 = (rr_r[222])/den;
    }
    double a7_0, a7_6, a7_8, a7_9, a7_16;
    {
      double den = rr_f[123] + rr_f[161] + rr_f[162] + rr_f[163] + rr_f[164] + 
        rr_f[165] + rr_f[166] + rr_f[167] + rr_f[195] + rr_f[224] + rr_f[228] + 
        rr_r[149] + rr_r[185] + rr_r[188] + rr_r[193] + rr_r[198]; 
      a7_0 = (rr_f[149] + rr_f[176] + rr_f[193] + rr_f[198] + rr_r[123] + 
        rr_r[162] + rr_r[164] + rr_r[165] + rr_r[167] + rr_r[228])/den; 
      a7_6 = (rr_r[161] + rr_r[163] + rr_r[166])/den;
      a7_8 = (rr_f[185] + rr_f[188])/den;
      a7_9 = (rr_r[195])/den;
      a7_16 = (rr_r[224])/den;
    }
    double a8_0, a8_7, a8_10, a8_16;
    {
      double den = rr_f[183] + rr_f[184] + rr_f[185] + rr_f[186] + rr_f[187] + 
        rr_f[188] + rr_f[189] + rr_f[199] + rr_f[226] + rr_r[172] + rr_r[197] + 
        rr_r[202]; 
      a8_0 = (rr_f[172] + rr_f[197] + rr_f[202] + rr_r[184] + rr_r[186] + 
        rr_r[187] + rr_r[189] + rr_r[229])/den; 
      a8_7 = (rr_r[185] + rr_r[188])/den;
      a8_10 = (rr_r[199])/den;
      a8_16 = (rr_r[226])/den;
    }
    double a9_0, a9_7, a9_11, a9_16;
    {
      double den = rr_f[203] + rr_f[225] + rr_r[192] + rr_r[195] + rr_r[201] + 
        rr_r[206]; 
      a9_0 = (rr_f[192] + rr_f[201] + rr_f[206] + rr_r[230])/den;
      a9_7 = (rr_f[195])/den;
      a9_11 = (rr_r[203])/den;
      a9_16 = (rr_r[225])/den;
    }
    double a10_0, a10_8, a10_12, a10_16;
    {
      double den = rr_f[207] + rr_f[223] + rr_r[196] + rr_r[199] + rr_r[205] + 
        rr_r[210]; 
      a10_0 = (rr_f[196] + rr_f[205] + rr_f[210] + rr_r[231] + rr_r[231])/den;
      a10_8 = (rr_f[199])/den;
      a10_12 = (rr_r[207])/den;
      a10_16 = (rr_r[223])/den;
    }
    double a11_0, a11_9, a11_16;
    {
      double den = rr_f[211] + rr_f[221] + rr_r[200] + rr_r[203] + rr_r[209] + 
        rr_r[214] + rr_r[216]; 
      a11_0 = (rr_f[200] + rr_f[209] + rr_f[214] + rr_f[216] + rr_r[211] + 
        rr_r[230])/den; 
      a11_9 = (rr_f[203])/den;
      a11_16 = (rr_r[221])/den;
    }
    double a12_0, a12_10, a12_13, a12_15;
    {
      double den = rr_f[215] + rr_f[220] + rr_r[204] + rr_r[207] + rr_r[213];
      a12_0 = (rr_f[204] + rr_f[213] + rr_r[229])/den;
      a12_10 = (rr_f[207])/den;
      a12_13 = (rr_r[215])/den;
      a12_15 = (rr_r[220])/den;
    }
    double a13_0, a13_12, a13_14;
    {
      double den = rr_f[217] + rr_r[212] + rr_r[215];
      a13_0 = (rr_f[212] + rr_r[227])/den;
      a13_12 = (rr_f[215])/den;
      a13_14 = (rr_r[217])/den;
    }
    double a14_0, a14_13, a14_16;
    {
      double den = rr_f[218] + rr_f[250] + rr_f[258] + rr_r[217] + rr_r[232] + 
        rr_r[235] + rr_r[238] + rr_r[241] + rr_r[244] + rr_r[247] + rr_r[251] + 
        rr_r[259]; 
      a14_0 = (rr_f[232] + rr_f[235] + rr_f[238] + rr_f[241] + rr_f[244] + 
        rr_f[247] + rr_f[251] + rr_f[259] + rr_r[250] + rr_r[258])/den; 
      a14_13 = (rr_f[217])/den;
      a14_16 = (rr_r[218])/den;
    }
    double a15_0, a15_12;
    {
      double den = rr_f[252] + rr_f[260] + rr_r[219] + rr_r[220] + rr_r[233] + 
        rr_r[236] + rr_r[239] + rr_r[242] + rr_r[245] + rr_r[248] + rr_r[253] + 
        rr_r[261]; 
      a15_0 = (rr_f[219] + rr_f[233] + rr_f[236] + rr_f[239] + rr_f[242] + 
        rr_f[245] + rr_f[248] + rr_f[253] + rr_f[261] + rr_r[252] + 
        rr_r[260])/den; 
      a15_12 = (rr_f[220])/den;
    }
    double a16_0, a16_6, a16_7, a16_8, a16_9, a16_10, a16_11, a16_14;
    {
      double den = rr_f[254] + rr_f[262] + rr_r[218] + rr_r[221] + rr_r[222] + 
        rr_r[223] + rr_r[224] + rr_r[225] + rr_r[226] + rr_r[234] + rr_r[237] + 
        rr_r[240] + rr_r[243] + rr_r[246] + rr_r[249] + rr_r[255] + rr_r[263]; 
      a16_0 = (rr_f[234] + rr_f[237] + rr_f[240] + rr_f[243] + rr_f[246] + 
        rr_f[249] + rr_f[255] + rr_f[263] + rr_r[254] + rr_r[262])/den; 
      a16_6 = (rr_f[222])/den;
      a16_7 = (rr_f[224])/den;
      a16_8 = (rr_f[226])/den;
      a16_9 = (rr_f[225])/den;
      a16_10 = (rr_f[223])/den;
      a16_11 = (rr_f[221])/den;
      a16_14 = (rr_f[218])/den;
    }
    double den, xq_1, xq_2, xq_3, xq_4, xq_5, xq_6, xq_7, xq_8, xq_9, xq_10, 
      xq_11, xq_12, xq_13, xq_14, xq_15, xq_16; 
    a12_0 = a12_0 + a12_15 * a15_0;
    den = 1.0/(1.0 - a15_12*a12_15);
    a12_0 = a12_0*den;
    a12_10 = a12_10*den;
    a12_13 = a12_13*den;
    a3_0 = a3_0 + a3_5 * a5_0;
    den = 1.0/(1.0 - a5_3*a3_5);
    a3_0 = a3_0*den;
    a3_2 = a3_2*den;
    a3_6 = a3_6*den;
    a3_1 = a3_1*den;
    a2_0 = a2_0 + a2_4 * a4_0;
    den = 1.0/(1.0 - a4_2*a2_4);
    a2_0 = a2_0*den;
    a2_3 = a2_3*den;
    a2_6 = a2_6*den;
    a2_1 = a2_1*den;
    a12_0 = a12_0 + a12_13 * a13_0;
    den = 1.0/(1.0 - a13_12*a12_13);
    double a12_14 = a12_13 * a13_14;
    a12_0 = a12_0*den;
    a12_10 = a12_10*den;
    a12_14 = a12_14*den;
    a14_0 = a14_0 + a14_13 * a13_0;
    den = 1.0/(1.0 - a13_14*a14_13);
    double a14_12 = a14_13 * a13_12;
    a14_0 = a14_0*den;
    a14_16 = a14_16*den;
    a14_12 = a14_12*den;
    a16_0 = a16_0 + a16_14 * a14_0;
    den = 1.0/(1.0 - a14_16*a16_14);
    double a16_12 = a16_14 * a14_12;
    a16_0 = a16_0*den;
    a16_6 = a16_6*den;
    a16_7 = a16_7*den;
    a16_9 = a16_9*den;
    a16_8 = a16_8*den;
    a16_12 = a16_12*den;
    a16_10 = a16_10*den;
    a16_11 = a16_11*den;
    a12_0 = a12_0 + a12_14 * a14_0;
    den = 1.0/(1.0 - a14_12*a12_14);
    double a12_16 = a12_14 * a14_16;
    a12_0 = a12_0*den;
    a12_16 = a12_16*den;
    a12_10 = a12_10*den;
    a16_0 = a16_0 + a16_11 * a11_0;
    den = 1.0/(1.0 - a11_16*a16_11);
    a16_9 = a16_9 + a16_11 * a11_9;
    a16_0 = a16_0*den;
    a16_6 = a16_6*den;
    a16_7 = a16_7*den;
    a16_9 = a16_9*den;
    a16_8 = a16_8*den;
    a16_12 = a16_12*den;
    a16_10 = a16_10*den;
    a9_0 = a9_0 + a9_11 * a11_0;
    den = 1.0/(1.0 - a11_9*a9_11);
    a9_16 = a9_16 + a9_11 * a11_16;
    a9_0 = a9_0*den;
    a9_16 = a9_16*den;
    a9_7 = a9_7*den;
    a16_0 = a16_0 + a16_10 * a10_0;
    den = 1.0/(1.0 - a10_16*a16_10);
    a16_8 = a16_8 + a16_10 * a10_8;
    a16_12 = a16_12 + a16_10 * a10_12;
    a16_0 = a16_0*den;
    a16_6 = a16_6*den;
    a16_7 = a16_7*den;
    a16_9 = a16_9*den;
    a16_8 = a16_8*den;
    a16_12 = a16_12*den;
    a8_0 = a8_0 + a8_10 * a10_0;
    den = 1.0/(1.0 - a10_8*a8_10);
    a8_16 = a8_16 + a8_10 * a10_16;
    double a8_12 = a8_10 * a10_12;
    a8_0 = a8_0*den;
    a8_16 = a8_16*den;
    a8_7 = a8_7*den;
    a8_12 = a8_12*den;
    a12_0 = a12_0 + a12_10 * a10_0;
    den = 1.0/(1.0 - a10_12*a12_10);
    a12_16 = a12_16 + a12_10 * a10_16;
    double a12_8 = a12_10 * a10_8;
    a12_0 = a12_0*den;
    a12_16 = a12_16*den;
    a12_8 = a12_8*den;
    a16_0 = a16_0 + a16_12 * a12_0;
    den = 1.0/(1.0 - a12_16*a16_12);
    a16_8 = a16_8 + a16_12 * a12_8;
    a16_0 = a16_0*den;
    a16_6 = a16_6*den;
    a16_7 = a16_7*den;
    a16_9 = a16_9*den;
    a16_8 = a16_8*den;
    a8_0 = a8_0 + a8_12 * a12_0;
    den = 1.0/(1.0 - a12_8*a8_12);
    a8_16 = a8_16 + a8_12 * a12_16;
    a8_0 = a8_0*den;
    a8_16 = a8_16*den;
    a8_7 = a8_7*den;
    a2_0 = a2_0 + a2_1 * a1_0;
    den = 1.0/(1.0 - a1_2*a2_1);
    a2_3 = a2_3 + a2_1 * a1_3;
    a2_0 = a2_0*den;
    a2_3 = a2_3*den;
    a2_6 = a2_6*den;
    a3_0 = a3_0 + a3_1 * a1_0;
    den = 1.0/(1.0 - a1_3*a3_1);
    a3_2 = a3_2 + a3_1 * a1_2;
    a3_0 = a3_0*den;
    a3_2 = a3_2*den;
    a3_6 = a3_6*den;
    a16_0 = a16_0 + a16_8 * a8_0;
    den = 1.0/(1.0 - a8_16*a16_8);
    a16_7 = a16_7 + a16_8 * a8_7;
    a16_0 = a16_0*den;
    a16_6 = a16_6*den;
    a16_7 = a16_7*den;
    a16_9 = a16_9*den;
    a6_0 = a6_0 + a6_8 * a8_0;
    den = 1.0/(1.0);
    a6_16 = a6_16 + a6_8 * a8_16;
    a6_7 = a6_7 + a6_8 * a8_7;
    a7_0 = a7_0 + a7_8 * a8_0;
    den = 1.0/(1.0 - a8_7*a7_8);
    a7_16 = a7_16 + a7_8 * a8_16;
    a7_0 = a7_0*den;
    a7_16 = a7_16*den;
    a7_6 = a7_6*den;
    a7_9 = a7_9*den;
    a16_0 = a16_0 + a16_9 * a9_0;
    den = 1.0/(1.0 - a9_16*a16_9);
    a16_7 = a16_7 + a16_9 * a9_7;
    a16_0 = a16_0*den;
    a16_6 = a16_6*den;
    a16_7 = a16_7*den;
    a7_0 = a7_0 + a7_9 * a9_0;
    den = 1.0/(1.0 - a9_7*a7_9);
    a7_16 = a7_16 + a7_9 * a9_16;
    a7_0 = a7_0*den;
    a7_16 = a7_16*den;
    a7_6 = a7_6*den;
    a16_0 = a16_0 + a16_7 * a7_0;
    den = 1.0/(1.0 - a7_16*a16_7);
    a16_6 = a16_6 + a16_7 * a7_6;
    a16_0 = a16_0*den;
    a16_6 = a16_6*den;
    a6_0 = a6_0 + a6_7 * a7_0;
    den = 1.0/(1.0 - a7_6*a6_7);
    a6_16 = a6_16 + a6_7 * a7_16;
    a6_0 = a6_0*den;
    a6_16 = a6_16*den;
    a6_2 = a6_2*den;
    a6_3 = a6_3*den;
    a16_0 = a16_0 + a16_6 * a6_0;
    den = 1.0/(1.0 - a6_16*a16_6);
    double a16_2 = a16_6 * a6_2;
    double a16_3 = a16_6 * a6_3;
    a16_0 = a16_0*den;
    a16_2 = a16_2*den;
    a16_3 = a16_3*den;
    a2_0 = a2_0 + a2_6 * a6_0;
    den = 1.0/(1.0 - a6_2*a2_6);
    double a2_16 = a2_6 * a6_16;
    a2_3 = a2_3 + a2_6 * a6_3;
    a2_0 = a2_0*den;
    a2_16 = a2_16*den;
    a2_3 = a2_3*den;
    a3_0 = a3_0 + a3_6 * a6_0;
    den = 1.0/(1.0 - a6_3*a3_6);
    double a3_16 = a3_6 * a6_16;
    a3_2 = a3_2 + a3_6 * a6_2;
    a3_0 = a3_0*den;
    a3_16 = a3_16*den;
    a3_2 = a3_2*den;
    a16_0 = a16_0 + a16_3 * a3_0;
    den = 1.0/(1.0 - a3_16*a16_3);
    a16_2 = a16_2 + a16_3 * a3_2;
    a16_0 = a16_0*den;
    a16_2 = a16_2*den;
    a2_0 = a2_0 + a2_3 * a3_0;
    den = 1.0/(1.0 - a3_2*a2_3);
    a2_16 = a2_16 + a2_3 * a3_16;
    a2_0 = a2_0*den;
    a2_16 = a2_16*den;
    a16_0 = a16_0 + a16_2 * a2_0;
    den = 1.0/(1.0 - a2_16*a16_2);
    a16_0 = a16_0*den;
    xq_16 = a16_0;
    xq_2 = a2_0 + a2_16*xq_16;
    xq_3 = a3_0 + a3_2*xq_2 + a3_16*xq_16;
    xq_6 = a6_0 + a6_2*xq_2 + a6_3*xq_3 + a6_16*xq_16;
    xq_7 = a7_0 + a7_6*xq_6 + a7_16*xq_16;
    xq_9 = a9_0 + a9_7*xq_7 + a9_16*xq_16;
    xq_8 = a8_0 + a8_7*xq_7 + a8_16*xq_16;
    xq_1 = a1_0 + a1_2*xq_2 + a1_3*xq_3;
    xq_12 = a12_0 + a12_8*xq_8 + a12_16*xq_16;
    xq_10 = a10_0 + a10_8*xq_8 + a10_12*xq_12 + a10_16*xq_16;
    xq_11 = a11_0 + a11_9*xq_9 + a11_16*xq_16;
    xq_14 = a14_0 + a14_12*xq_12 + a14_16*xq_16;
    xq_13 = a13_0 + a13_12*xq_12 + a13_14*xq_14;
    xq_4 = a4_0 + a4_2*xq_2;
    xq_5 = a5_0 + a5_3*xq_3;
    xq_15 = a15_0 + a15_12*xq_12;
    rr_f[39] *= xq_1;
    rr_f[40] *= xq_1;
    rr_f[41] *= xq_1;
    rr_f[42] *= xq_1;
    rr_f[43] *= xq_1;
    rr_f[44] *= xq_1;
    rr_f[45] *= xq_1;
    rr_f[75] *= xq_1;
    rr_f[88] *= xq_1;
    rr_f[120] *= xq_1;
    rr_r[47] *= xq_1;
    rr_r[54] *= xq_1;
    rr_r[55] *= xq_1;
    rr_r[56] *= xq_1;
    rr_r[66] *= xq_1;
    rr_r[91] *= xq_1;
    rr_r[117] *= xq_1;
    rr_f[47] *= xq_2;
    rr_f[48] *= xq_2;
    rr_f[49] *= xq_2;
    rr_f[50] *= xq_2;
    rr_f[51] *= xq_2;
    rr_f[52] *= xq_2;
    rr_f[53] *= xq_2;
    rr_f[54] *= xq_2;
    rr_f[55] *= xq_2;
    rr_f[56] *= xq_2;
    rr_f[57] *= xq_2;
    rr_f[76] *= xq_2;
    rr_f[89] *= xq_2;
    rr_f[121] *= xq_2;
    rr_f[139] *= xq_2;
    rr_r[67] *= xq_2;
    rr_r[81] *= xq_2;
    rr_f[30] *= xq_3;
    rr_f[31] *= xq_3;
    rr_f[32] *= xq_3;
    rr_f[33] *= xq_3;
    rr_f[34] *= xq_3;
    rr_f[35] *= xq_3;
    rr_f[36] *= xq_3;
    rr_f[38] *= xq_3;
    rr_f[73] *= xq_3;
    rr_f[93] *= xq_3;
    rr_f[119] *= xq_3;
    rr_f[147] *= xq_3;
    rr_f[171] *= xq_3;
    rr_r[63] *= xq_3;
    rr_r[40] *= xq_3;
    rr_r[43] *= xq_3;
    rr_r[49] *= xq_3;
    rr_r[59] *= xq_3;
    rr_r[60] *= xq_3;
    rr_r[61] *= xq_3;
    rr_r[62] *= xq_3;
    rr_r[74] *= xq_3;
    rr_r[101] *= xq_3;
    rr_r[111] *= xq_3;
    rr_r[116] *= xq_3;
    rr_r[158] *= xq_3;
    rr_f[79] *= xq_4;
    rr_f[80] *= xq_4;
    rr_f[81] *= xq_4;
    rr_f[82] *= xq_4;
    rr_f[83] *= xq_4;
    rr_f[84] *= xq_4;
    rr_r[58] *= xq_4;
    rr_r[68] *= xq_4;
    rr_r[71] *= xq_4;
    rr_f[90] *= xq_5;
    rr_f[95] *= xq_5;
    rr_f[96] *= xq_5;
    rr_f[97] *= xq_5;
    rr_f[98] *= xq_5;
    rr_f[99] *= xq_5;
    rr_f[100] *= xq_5;
    rr_f[101] *= xq_5;
    rr_f[102] *= xq_5;
    rr_f[103] *= xq_5;
    rr_f[106] *= xq_5;
    rr_f[107] *= xq_5;
    rr_f[108] *= xq_5;
    rr_f[125] *= xq_5;
    rr_r[93] *= xq_5;
    rr_r[114] *= xq_5;
    rr_r[115] *= xq_5;
    rr_r[118] *= xq_5;
    rr_r[122] *= xq_5;
    rr_r[124] *= xq_5;
    rr_r[146] *= xq_5;
    rr_r[159] *= xq_5;
    rr_r[160] *= xq_5;
    rr_r[191] *= xq_5;
    rr_f[129] *= xq_6;
    rr_f[130] *= xq_6;
    rr_f[131] *= xq_6;
    rr_f[126] *= xq_6;
    rr_f[127] *= xq_6;
    rr_f[128] *= xq_6;
    rr_f[132] *= xq_6;
    rr_f[133] *= xq_6;
    rr_f[222] *= xq_6;
    rr_r[78] *= xq_6;
    rr_r[113] *= xq_6;
    rr_r[119] *= xq_6;
    rr_r[267] *= xq_6;
    rr_r[136] *= xq_6;
    rr_r[137] *= xq_6;
    rr_r[138] *= xq_6;
    rr_r[139] *= xq_6;
    rr_r[140] *= xq_6;
    rr_r[161] *= xq_6;
    rr_r[163] *= xq_6;
    rr_r[166] *= xq_6;
    rr_r[173] *= xq_6;
    rr_r[194] *= xq_6;
    rr_f[123] *= xq_7;
    rr_f[161] *= xq_7;
    rr_f[162] *= xq_7;
    rr_f[163] *= xq_7;
    rr_f[164] *= xq_7;
    rr_f[165] *= xq_7;
    rr_f[166] *= xq_7;
    rr_f[167] *= xq_7;
    rr_f[195] *= xq_7;
    rr_f[224] *= xq_7;
    rr_f[228] *= xq_7;
    rr_r[149] *= xq_7;
    rr_r[185] *= xq_7;
    rr_r[188] *= xq_7;
    rr_r[193] *= xq_7;
    rr_r[198] *= xq_7;
    rr_f[183] *= xq_8;
    rr_f[184] *= xq_8;
    rr_f[185] *= xq_8;
    rr_f[186] *= xq_8;
    rr_f[187] *= xq_8;
    rr_f[188] *= xq_8;
    rr_f[189] *= xq_8;
    rr_f[199] *= xq_8;
    rr_f[226] *= xq_8;
    rr_r[172] *= xq_8;
    rr_r[197] *= xq_8;
    rr_r[202] *= xq_8;
    rr_f[203] *= xq_9;
    rr_f[225] *= xq_9;
    rr_r[192] *= xq_9;
    rr_r[195] *= xq_9;
    rr_r[201] *= xq_9;
    rr_r[206] *= xq_9;
    rr_f[207] *= xq_10;
    rr_f[223] *= xq_10;
    rr_r[196] *= xq_10;
    rr_r[199] *= xq_10;
    rr_r[205] *= xq_10;
    rr_r[210] *= xq_10;
    rr_f[211] *= xq_11;
    rr_f[221] *= xq_11;
    rr_r[200] *= xq_11;
    rr_r[203] *= xq_11;
    rr_r[209] *= xq_11;
    rr_r[214] *= xq_11;
    rr_r[216] *= xq_11;
    rr_f[215] *= xq_12;
    rr_f[220] *= xq_12;
    rr_r[204] *= xq_12;
    rr_r[207] *= xq_12;
    rr_r[213] *= xq_12;
    rr_f[217] *= xq_13;
    rr_r[212] *= xq_13;
    rr_r[215] *= xq_13;
    rr_f[218] *= xq_14;
    rr_f[250] *= xq_14;
    rr_f[258] *= xq_14;
    rr_r[217] *= xq_14;
    rr_r[232] *= xq_14;
    rr_r[235] *= xq_14;
    rr_r[238] *= xq_14;
    rr_r[241] *= xq_14;
    rr_r[244] *= xq_14;
    rr_r[247] *= xq_14;
    rr_r[251] *= xq_14;
    rr_r[259] *= xq_14;
    rr_f[252] *= xq_15;
    rr_f[260] *= xq_15;
    rr_r[219] *= xq_15;
    rr_r[220] *= xq_15;
    rr_r[233] *= xq_15;
    rr_r[236] *= xq_15;
    rr_r[239] *= xq_15;
    rr_r[242] *= xq_15;
    rr_r[245] *= xq_15;
    rr_r[248] *= xq_15;
    rr_r[253] *= xq_15;
    rr_r[261] *= xq_15;
    rr_f[254] *= xq_16;
    rr_f[262] *= xq_16;
    rr_r[218] *= xq_16;
    rr_r[221] *= xq_16;
    rr_r[222] *= xq_16;
    rr_r[223] *= xq_16;
    rr_r[224] *= xq_16;
    rr_r[225] *= xq_16;
    rr_r[226] *= xq_16;
    rr_r[234] *= xq_16;
    rr_r[237] *= xq_16;
    rr_r[240] *= xq_16;
    rr_r[243] *= xq_16;
    rr_r[246] *= xq_16;
    rr_r[249] *= xq_16;
    rr_r[255] *= xq_16;
    rr_r[263] *= xq_16;
  }
  // QSSA connected component
  {
    double a17_0, a17_18;
    {
      double den = rr_f[257] + rr_f[264] + rr_r[256] + rr_r[265];
      a17_0 = (rr_f[256] + rr_r[257])/den;
      a17_18 = (rr_f[265] + rr_r[264])/den;
    }
    double a18_0, a18_17;
    {
      double den = rr_f[265] + rr_f[266] + rr_r[264];
      a18_0 = (rr_r[266])/den;
      a18_17 = (rr_f[264] + rr_r[265])/den;
    }
    double den, xq_17, xq_18;
    a17_0 = a17_0 + a17_18 * a18_0;
    den = 1.0/(1.0 - a18_17*a17_18);
    a17_0 = a17_0*den;
    xq_17 = a17_0;
    xq_18 = a18_0 + a18_17*xq_17;
    rr_f[257] *= xq_17;
    rr_f[264] *= xq_17;
    rr_r[256] *= xq_17;
    rr_r[265] *= xq_17;
    rr_f[265] *= xq_18;
    rr_f[266] *= xq_18;
    rr_r[264] *= xq_18;
  }
  // Stiff species NC12H26
  {
    double ddot = rr_f[232] + rr_f[233] + rr_f[234] + rr_f[235] + rr_f[236] + 
      rr_f[237] + rr_f[238] + rr_f[239] + rr_f[240] + rr_f[241] + rr_f[242] + 
      rr_f[243] + rr_f[244] + rr_f[245] + rr_f[246] + rr_f[247] + rr_f[248] + 
      rr_f[249] + rr_r[227] + rr_r[228] + rr_r[229] + rr_r[230] + rr_r[231]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[0])
    {
      double cdot = rr_f[227] + rr_f[228] + rr_f[229] + rr_f[230] + rr_f[231] + 
        rr_r[232] + rr_r[233] + rr_r[234] + rr_r[235] + rr_r[236] + rr_r[237] + 
        rr_r[238] + rr_r[239] + rr_r[240] + rr_r[241] + rr_r[242] + rr_r[243] + 
        rr_r[244] + rr_r[245] + rr_r[246] + rr_r[247] + rr_r[248] + rr_r[249]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[0] * 5.870576564587907e-03;
      double c0 = mole_frac[0] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[0]- c0) / dt) 
        * recip_ddot; 
      rr_f[232] *= scale_r;
      rr_f[233] *= scale_r;
      rr_f[234] *= scale_r;
      rr_f[235] *= scale_r;
      rr_f[236] *= scale_r;
      rr_f[237] *= scale_r;
      rr_f[238] *= scale_r;
      rr_f[239] *= scale_r;
      rr_f[240] *= scale_r;
      rr_f[241] *= scale_r;
      rr_f[242] *= scale_r;
      rr_f[243] *= scale_r;
      rr_f[244] *= scale_r;
      rr_f[245] *= scale_r;
      rr_f[246] *= scale_r;
      rr_f[247] *= scale_r;
      rr_f[248] *= scale_r;
      rr_f[249] *= scale_r;
      rr_r[227] *= scale_r;
      rr_r[228] *= scale_r;
      rr_r[229] *= scale_r;
      rr_r[230] *= scale_r;
      rr_r[231] *= scale_r;
    }
  }
  // Stiff species H
  {
    double ddot = rr_f[0] + rr_f[4] + rr_f[5] + rr_f[8] + rr_f[12] + rr_f[13] + 
      rr_f[18] + rr_f[18] + rr_f[19] + rr_f[21] + rr_f[21] + rr_f[22] + rr_f[22] 
      + rr_f[23] + rr_f[23] + rr_f[24] + rr_f[31] + rr_f[38] + rr_f[39] + 
      rr_f[58] + rr_f[59] + rr_f[64] + rr_f[79] + rr_f[80] + rr_f[81] + rr_f[85] 
      + rr_f[95] + rr_f[96] + rr_f[111] + rr_f[113] + rr_f[114] + rr_f[126] + 
      rr_f[127] + rr_f[136] + rr_f[141] + rr_f[149] + rr_f[150] + rr_f[151] + 
      rr_f[158] + rr_f[161] + rr_f[162] + rr_f[168] + rr_f[169] + rr_f[172] + 
      rr_f[173] + rr_f[174] + rr_f[175] + rr_f[183] + rr_f[184] + rr_f[192] + 
      rr_f[193] + rr_f[194] + rr_f[196] + rr_f[197] + rr_f[198] + rr_f[200] + 
      rr_f[201] + rr_f[202] + rr_f[204] + rr_f[205] + rr_f[206] + rr_f[208] + 
      rr_f[209] + rr_f[210] + rr_f[212] + rr_f[213] + rr_f[214] + rr_f[232] + 
      rr_f[233] + rr_f[234] + rr_r[1] + rr_r[2] + rr_r[6] + rr_r[25] + rr_r[26] 
      + rr_r[30] + rr_r[33] + rr_r[36] + rr_r[40] + rr_r[41] + rr_r[42] + 
      rr_r[44] + rr_r[44] + rr_r[49] + rr_r[50] + rr_r[51] + rr_r[52] + rr_r[65] 
      + rr_r[75] + rr_r[76] + rr_r[78] + rr_r[90] + rr_r[108] + rr_r[120] + 
      rr_r[121] + rr_r[142] + rr_r[143] + rr_r[143] + rr_r[152] + rr_r[152]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[1])
    {
      double cdot = rr_f[1] + rr_f[2] + rr_f[6] + rr_f[25] + rr_f[26] + rr_f[30] 
        + rr_f[33] + rr_f[36] + rr_f[40] + rr_f[41] + rr_f[42] + rr_f[44] + 
        rr_f[44] + rr_f[49] + rr_f[50] + rr_f[51] + rr_f[52] + rr_f[65] + 
        rr_f[75] + rr_f[76] + rr_f[78] + rr_f[90] + rr_f[108] + rr_f[120] + 
        rr_f[121] + rr_f[142] + rr_f[143] + rr_f[143] + rr_f[152] + rr_f[152] + 
        rr_r[0] + rr_r[4] + rr_r[5] + rr_r[8] + rr_r[12] + rr_r[13] + rr_r[18] + 
        rr_r[18] + rr_r[19] + rr_r[21] + rr_r[21] + rr_r[22] + rr_r[22] + 
        rr_r[23] + rr_r[23] + rr_r[24] + rr_r[31] + rr_r[38] + rr_r[39] + 
        rr_r[58] + rr_r[59] + rr_r[64] + rr_r[79] + rr_r[80] + rr_r[81] + 
        rr_r[85] + rr_r[95] + rr_r[96] + rr_r[111] + rr_r[113] + rr_r[114] + 
        rr_r[126] + rr_r[127] + rr_r[136] + rr_r[141] + rr_r[149] + rr_r[150] + 
        rr_r[151] + rr_r[158] + rr_r[161] + rr_r[162] + rr_r[168] + rr_r[169] + 
        rr_r[172] + rr_r[173] + rr_r[174] + rr_r[175] + rr_r[183] + rr_r[184] + 
        rr_r[192] + rr_r[193] + rr_r[194] + rr_r[196] + rr_r[197] + rr_r[198] + 
        rr_r[200] + rr_r[201] + rr_r[202] + rr_r[204] + rr_r[205] + rr_r[206] + 
        rr_r[208] + rr_r[209] + rr_r[210] + rr_r[212] + rr_r[213] + rr_r[214] + 
        rr_r[232] + rr_r[233] + rr_r[234]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[1] * 0.9920930186414277;
      double c0 = mole_frac[1] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[1]- c0) / dt) 
        * recip_ddot; 
      rr_f[0] *= scale_r;
      rr_f[4] *= scale_r;
      rr_f[5] *= scale_r;
      rr_f[8] *= scale_r;
      rr_f[12] *= scale_r;
      rr_f[13] *= scale_r;
      rr_f[18] *= scale_r;
      rr_f[18] *= scale_r;
      rr_f[19] *= scale_r;
      rr_f[21] *= scale_r;
      rr_f[21] *= scale_r;
      rr_f[22] *= scale_r;
      rr_f[22] *= scale_r;
      rr_f[23] *= scale_r;
      rr_f[23] *= scale_r;
      rr_f[24] *= scale_r;
      rr_f[31] *= scale_r;
      rr_f[38] *= scale_r;
      rr_f[39] *= scale_r;
      rr_f[58] *= scale_r;
      rr_f[59] *= scale_r;
      rr_f[64] *= scale_r;
      rr_f[79] *= scale_r;
      rr_f[80] *= scale_r;
      rr_f[81] *= scale_r;
      rr_f[85] *= scale_r;
      rr_f[95] *= scale_r;
      rr_f[96] *= scale_r;
      rr_f[111] *= scale_r;
      rr_f[113] *= scale_r;
      rr_f[114] *= scale_r;
      rr_f[126] *= scale_r;
      rr_f[127] *= scale_r;
      rr_f[136] *= scale_r;
      rr_f[141] *= scale_r;
      rr_f[149] *= scale_r;
      rr_f[150] *= scale_r;
      rr_f[151] *= scale_r;
      rr_f[158] *= scale_r;
      rr_f[161] *= scale_r;
      rr_f[162] *= scale_r;
      rr_f[168] *= scale_r;
      rr_f[169] *= scale_r;
      rr_f[172] *= scale_r;
      rr_f[173] *= scale_r;
      rr_f[174] *= scale_r;
      rr_f[175] *= scale_r;
      rr_f[183] *= scale_r;
      rr_f[184] *= scale_r;
      rr_f[192] *= scale_r;
      rr_f[193] *= scale_r;
      rr_f[194] *= scale_r;
      rr_f[196] *= scale_r;
      rr_f[197] *= scale_r;
      rr_f[198] *= scale_r;
      rr_f[200] *= scale_r;
      rr_f[201] *= scale_r;
      rr_f[202] *= scale_r;
      rr_f[204] *= scale_r;
      rr_f[205] *= scale_r;
      rr_f[206] *= scale_r;
      rr_f[208] *= scale_r;
      rr_f[209] *= scale_r;
      rr_f[210] *= scale_r;
      rr_f[212] *= scale_r;
      rr_f[213] *= scale_r;
      rr_f[214] *= scale_r;
      rr_f[232] *= scale_r;
      rr_f[233] *= scale_r;
      rr_f[234] *= scale_r;
      rr_r[1] *= scale_r;
      rr_r[2] *= scale_r;
      rr_r[6] *= scale_r;
      rr_r[25] *= scale_r;
      rr_r[26] *= scale_r;
      rr_r[30] *= scale_r;
      rr_r[33] *= scale_r;
      rr_r[36] *= scale_r;
      rr_r[40] *= scale_r;
      rr_r[41] *= scale_r;
      rr_r[42] *= scale_r;
      rr_r[44] *= scale_r;
      rr_r[44] *= scale_r;
      rr_r[49] *= scale_r;
      rr_r[50] *= scale_r;
      rr_r[51] *= scale_r;
      rr_r[52] *= scale_r;
      rr_r[65] *= scale_r;
      rr_r[75] *= scale_r;
      rr_r[76] *= scale_r;
      rr_r[78] *= scale_r;
      rr_r[90] *= scale_r;
      rr_r[108] *= scale_r;
      rr_r[120] *= scale_r;
      rr_r[121] *= scale_r;
      rr_r[142] *= scale_r;
      rr_r[143] *= scale_r;
      rr_r[143] *= scale_r;
      rr_r[152] *= scale_r;
      rr_r[152] *= scale_r;
    }
  }
  // Stiff species O
  {
    double ddot = rr_f[1] + rr_f[9] + rr_f[14] + rr_f[20] + rr_f[20] + rr_f[24] 
      + rr_f[28] + rr_f[32] + rr_f[33] + rr_f[40] + rr_f[48] + rr_f[49] + 
      rr_f[60] + rr_f[65] + rr_f[82] + rr_f[86] + rr_f[91] + rr_f[97] + 
      rr_f[115] + rr_f[116] + rr_f[117] + rr_f[128] + rr_f[137] + rr_f[142] + 
      rr_f[152] + rr_f[153] + rr_f[154] + rr_f[159] + rr_f[163] + rr_f[176] + 
      rr_f[177] + rr_f[178] + rr_f[185] + rr_f[235] + rr_f[236] + rr_f[237] + 
      rr_r[0] + rr_r[3] + rr_r[8] + rr_r[29] + rr_r[68] + rr_r[100]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[2])
    {
      double cdot = rr_f[0] + rr_f[3] + rr_f[8] + rr_f[29] + rr_f[68] + 
        rr_f[100] + rr_r[1] + rr_r[9] + rr_r[14] + rr_r[20] + rr_r[20] + 
        rr_r[24] + rr_r[28] + rr_r[32] + rr_r[33] + rr_r[40] + rr_r[48] + 
        rr_r[49] + rr_r[60] + rr_r[65] + rr_r[82] + rr_r[86] + rr_r[91] + 
        rr_r[97] + rr_r[115] + rr_r[116] + rr_r[117] + rr_r[128] + rr_r[137] + 
        rr_r[142] + rr_r[152] + rr_r[153] + rr_r[154] + rr_r[159] + rr_r[163] + 
        rr_r[176] + rr_r[177] + rr_r[178] + rr_r[185] + rr_r[235] + rr_r[236] + 
        rr_r[237]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[2] * 0.06250234383789392;
      double c0 = mole_frac[2] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[2]- c0) / dt) 
        * recip_ddot; 
      rr_f[1] *= scale_r;
      rr_f[9] *= scale_r;
      rr_f[14] *= scale_r;
      rr_f[20] *= scale_r;
      rr_f[20] *= scale_r;
      rr_f[24] *= scale_r;
      rr_f[28] *= scale_r;
      rr_f[32] *= scale_r;
      rr_f[33] *= scale_r;
      rr_f[40] *= scale_r;
      rr_f[48] *= scale_r;
      rr_f[49] *= scale_r;
      rr_f[60] *= scale_r;
      rr_f[65] *= scale_r;
      rr_f[82] *= scale_r;
      rr_f[86] *= scale_r;
      rr_f[91] *= scale_r;
      rr_f[97] *= scale_r;
      rr_f[115] *= scale_r;
      rr_f[116] *= scale_r;
      rr_f[117] *= scale_r;
      rr_f[128] *= scale_r;
      rr_f[137] *= scale_r;
      rr_f[142] *= scale_r;
      rr_f[152] *= scale_r;
      rr_f[153] *= scale_r;
      rr_f[154] *= scale_r;
      rr_f[159] *= scale_r;
      rr_f[163] *= scale_r;
      rr_f[176] *= scale_r;
      rr_f[177] *= scale_r;
      rr_f[178] *= scale_r;
      rr_f[185] *= scale_r;
      rr_f[235] *= scale_r;
      rr_f[236] *= scale_r;
      rr_f[237] *= scale_r;
      rr_r[0] *= scale_r;
      rr_r[3] *= scale_r;
      rr_r[8] *= scale_r;
      rr_r[29] *= scale_r;
      rr_r[68] *= scale_r;
      rr_r[100] *= scale_r;
    }
  }
  // Stiff species OH
  {
    double ddot = rr_f[2] + rr_f[3] + rr_f[3] + rr_f[7] + rr_f[15] + rr_f[16] + 
      rr_f[17] + rr_f[17] + rr_f[19] + rr_f[25] + rr_f[26] + rr_f[34] + rr_f[41] 
      + rr_f[50] + rr_f[61] + rr_f[66] + rr_f[67] + rr_f[83] + rr_f[87] + 
      rr_f[92] + rr_f[98] + rr_f[118] + rr_f[138] + rr_f[143] + rr_f[155] + 
      rr_f[160] + rr_f[164] + rr_f[179] + rr_f[186] + rr_f[238] + rr_f[239] + 
      rr_f[240] + rr_r[0] + rr_r[1] + rr_r[5] + rr_r[5] + rr_r[9] + rr_r[12] + 
      rr_r[14] + rr_r[24] + rr_r[27] + rr_r[32] + rr_r[43] + rr_r[45] + rr_r[52] 
      + rr_r[60] + rr_r[69] + rr_r[71] + rr_r[80] + rr_r[82] + rr_r[86] + 
      rr_r[102] + rr_r[112] + rr_r[115] + rr_r[132] + rr_r[137] + rr_r[144] + 
      rr_r[146] + rr_r[154] + rr_r[159] + rr_r[166] + rr_r[170] + rr_r[177] + 
      rr_r[178] + rr_r[188] + rr_r[235] + rr_r[236] + rr_r[237] + rr_r[266]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[3])
    {
      double cdot = rr_f[0] + rr_f[1] + rr_f[5] + rr_f[5] + rr_f[9] + rr_f[12] + 
        rr_f[14] + rr_f[24] + rr_f[27] + rr_f[32] + rr_f[43] + rr_f[45] + 
        rr_f[52] + rr_f[60] + rr_f[69] + rr_f[71] + rr_f[80] + rr_f[82] + 
        rr_f[86] + rr_f[102] + rr_f[112] + rr_f[115] + rr_f[267] + rr_f[132] + 
        rr_f[137] + rr_f[144] + rr_f[146] + rr_f[154] + rr_f[159] + rr_f[166] + 
        rr_f[170] + rr_f[177] + rr_f[178] + rr_f[188] + rr_f[235] + rr_f[236] + 
        rr_f[237] + rr_f[266] + rr_r[2] + rr_r[3] + rr_r[3] + rr_r[7] + rr_r[15] 
        + rr_r[16] + rr_r[17] + rr_r[17] + rr_r[19] + rr_r[25] + rr_r[26] + 
        rr_r[34] + rr_r[41] + rr_r[50] + rr_r[61] + rr_r[66] + rr_r[67] + 
        rr_r[83] + rr_r[87] + rr_r[92] + rr_r[98] + rr_r[118] + rr_r[138] + 
        rr_r[143] + rr_r[155] + rr_r[160] + rr_r[164] + rr_r[179] + rr_r[186] + 
        rr_r[238] + rr_r[239] + rr_r[240]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[3] * 0.05879803873262004;
      double c0 = mole_frac[3] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[3]- c0) / dt) 
        * recip_ddot; 
      rr_f[2] *= scale_r;
      rr_f[3] *= scale_r;
      rr_f[3] *= scale_r;
      rr_f[7] *= scale_r;
      rr_f[15] *= scale_r;
      rr_f[16] *= scale_r;
      rr_f[17] *= scale_r;
      rr_f[17] *= scale_r;
      rr_f[19] *= scale_r;
      rr_f[25] *= scale_r;
      rr_f[26] *= scale_r;
      rr_f[34] *= scale_r;
      rr_f[41] *= scale_r;
      rr_f[50] *= scale_r;
      rr_f[61] *= scale_r;
      rr_f[66] *= scale_r;
      rr_f[67] *= scale_r;
      rr_f[83] *= scale_r;
      rr_f[87] *= scale_r;
      rr_f[92] *= scale_r;
      rr_f[98] *= scale_r;
      rr_f[118] *= scale_r;
      rr_f[138] *= scale_r;
      rr_f[143] *= scale_r;
      rr_f[155] *= scale_r;
      rr_f[160] *= scale_r;
      rr_f[164] *= scale_r;
      rr_f[179] *= scale_r;
      rr_f[186] *= scale_r;
      rr_f[238] *= scale_r;
      rr_f[239] *= scale_r;
      rr_f[240] *= scale_r;
      rr_r[0] *= scale_r;
      rr_r[1] *= scale_r;
      rr_r[5] *= scale_r;
      rr_r[5] *= scale_r;
      rr_r[9] *= scale_r;
      rr_r[12] *= scale_r;
      rr_r[14] *= scale_r;
      rr_r[24] *= scale_r;
      rr_r[27] *= scale_r;
      rr_r[32] *= scale_r;
      rr_r[43] *= scale_r;
      rr_r[45] *= scale_r;
      rr_r[52] *= scale_r;
      rr_r[60] *= scale_r;
      rr_r[69] *= scale_r;
      rr_r[71] *= scale_r;
      rr_r[80] *= scale_r;
      rr_r[82] *= scale_r;
      rr_r[86] *= scale_r;
      rr_r[102] *= scale_r;
      rr_r[112] *= scale_r;
      rr_r[115] *= scale_r;
      rr_r[132] *= scale_r;
      rr_r[137] *= scale_r;
      rr_r[144] *= scale_r;
      rr_r[146] *= scale_r;
      rr_r[154] *= scale_r;
      rr_r[159] *= scale_r;
      rr_r[166] *= scale_r;
      rr_r[170] *= scale_r;
      rr_r[177] *= scale_r;
      rr_r[178] *= scale_r;
      rr_r[188] *= scale_r;
      rr_r[235] *= scale_r;
      rr_r[236] *= scale_r;
      rr_r[237] *= scale_r;
      rr_r[266] *= scale_r;
    }
  }
  // Stiff species HO2
  {
    double ddot = rr_f[5] + rr_f[7] + rr_f[8] + rr_f[9] + rr_f[10] + rr_f[10] + 
      rr_f[11] + rr_f[11] + rr_f[27] + rr_f[63] + rr_f[45] + rr_f[130] + 
      rr_f[131] + rr_f[70] + rr_f[71] + rr_f[102] + rr_f[132] + rr_f[145] + 
      rr_f[146] + rr_f[156] + rr_f[166] + rr_f[170] + rr_f[181] + rr_f[188] + 
      rr_f[244] + rr_f[245] + rr_f[246] + rr_f[259] + rr_f[261] + rr_f[263] + 
      rr_r[4] + rr_r[6] + rr_r[13] + rr_r[14] + rr_r[15] + rr_r[16] + rr_r[35] + 
      rr_r[62] + rr_r[129] + rr_r[72] + rr_r[84] + rr_r[99] + rr_r[103] + 
      rr_r[124] + rr_r[133] + rr_r[165] + rr_r[180] + rr_r[187] + rr_r[241] + 
      rr_r[242] + rr_r[243]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[4])
    {
      double cdot = rr_f[4] + rr_f[6] + rr_f[13] + rr_f[14] + rr_f[15] + 
        rr_f[16] + rr_f[35] + rr_f[62] + rr_f[129] + rr_f[72] + rr_f[84] + 
        rr_f[99] + rr_f[103] + rr_f[124] + rr_f[133] + rr_f[165] + rr_f[180] + 
        rr_f[187] + rr_f[241] + rr_f[242] + rr_f[243] + rr_f[258] + rr_f[260] + 
        rr_f[262] + rr_r[5] + rr_r[7] + rr_r[8] + rr_r[9] + rr_r[10] + rr_r[10] 
        + rr_r[11] + rr_r[11] + rr_r[27] + rr_r[63] + rr_r[45] + rr_r[130] + 
        rr_r[131] + rr_r[70] + rr_r[71] + rr_r[102] + rr_r[132] + rr_r[145] + 
        rr_r[146] + rr_r[156] + rr_r[166] + rr_r[170] + rr_r[181] + rr_r[188] + 
        rr_r[244] + rr_r[245] + rr_r[246]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[4] * 0.03029681486555637;
      double c0 = mole_frac[4] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[4]- c0) / dt) 
        * recip_ddot; 
      rr_f[5] *= scale_r;
      rr_f[7] *= scale_r;
      rr_f[8] *= scale_r;
      rr_f[9] *= scale_r;
      rr_f[10] *= scale_r;
      rr_f[10] *= scale_r;
      rr_f[11] *= scale_r;
      rr_f[11] *= scale_r;
      rr_f[27] *= scale_r;
      rr_f[63] *= scale_r;
      rr_f[45] *= scale_r;
      rr_f[130] *= scale_r;
      rr_f[131] *= scale_r;
      rr_f[70] *= scale_r;
      rr_f[71] *= scale_r;
      rr_f[102] *= scale_r;
      rr_f[132] *= scale_r;
      rr_f[145] *= scale_r;
      rr_f[146] *= scale_r;
      rr_f[156] *= scale_r;
      rr_f[166] *= scale_r;
      rr_f[170] *= scale_r;
      rr_f[181] *= scale_r;
      rr_f[188] *= scale_r;
      rr_f[244] *= scale_r;
      rr_f[245] *= scale_r;
      rr_f[246] *= scale_r;
      rr_f[259] *= scale_r;
      rr_f[261] *= scale_r;
      rr_f[263] *= scale_r;
      rr_r[4] *= scale_r;
      rr_r[6] *= scale_r;
      rr_r[13] *= scale_r;
      rr_r[14] *= scale_r;
      rr_r[15] *= scale_r;
      rr_r[16] *= scale_r;
      rr_r[35] *= scale_r;
      rr_r[62] *= scale_r;
      rr_r[129] *= scale_r;
      rr_r[72] *= scale_r;
      rr_r[84] *= scale_r;
      rr_r[99] *= scale_r;
      rr_r[103] *= scale_r;
      rr_r[124] *= scale_r;
      rr_r[133] *= scale_r;
      rr_r[165] *= scale_r;
      rr_r[180] *= scale_r;
      rr_r[187] *= scale_r;
      rr_r[241] *= scale_r;
      rr_r[242] *= scale_r;
      rr_r[243] *= scale_r;
    }
  }
  // Stiff species H2
  {
    double ddot = rr_f[1] + rr_f[2] + rr_f[6] + rr_f[21] + rr_f[37] + rr_f[42] + 
      rr_f[51] + rr_r[13] + rr_r[18] + rr_r[21] + rr_r[21] + rr_r[22] + rr_r[23] 
      + rr_r[31] + rr_r[46] + rr_r[48] + rr_r[59] + rr_r[79] + rr_r[85] + 
      rr_r[96] + rr_r[114] + rr_r[127] + rr_r[136] + rr_r[151] + rr_r[162] + 
      rr_r[175] + rr_r[184] + rr_r[232] + rr_r[233] + rr_r[234]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[5])
    {
      double cdot = rr_f[13] + rr_f[18] + rr_f[21] + rr_f[21] + rr_f[22] + 
        rr_f[23] + rr_f[31] + rr_f[46] + rr_f[48] + rr_f[59] + rr_f[79] + 
        rr_f[85] + rr_f[96] + rr_f[114] + rr_f[127] + rr_f[136] + rr_f[151] + 
        rr_f[162] + rr_f[175] + rr_f[184] + rr_f[232] + rr_f[233] + rr_f[234] + 
        rr_r[1] + rr_r[2] + rr_r[6] + rr_r[21] + rr_r[37] + rr_r[42] + rr_r[51]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[5] * 0.4960465093207139;
      double c0 = mole_frac[5] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[5]- c0) / dt) 
        * recip_ddot; 
      rr_f[1] *= scale_r;
      rr_f[2] *= scale_r;
      rr_f[6] *= scale_r;
      rr_f[21] *= scale_r;
      rr_f[37] *= scale_r;
      rr_f[42] *= scale_r;
      rr_f[51] *= scale_r;
      rr_r[13] *= scale_r;
      rr_r[18] *= scale_r;
      rr_r[21] *= scale_r;
      rr_r[21] *= scale_r;
      rr_r[22] *= scale_r;
      rr_r[23] *= scale_r;
      rr_r[31] *= scale_r;
      rr_r[46] *= scale_r;
      rr_r[48] *= scale_r;
      rr_r[59] *= scale_r;
      rr_r[79] *= scale_r;
      rr_r[85] *= scale_r;
      rr_r[96] *= scale_r;
      rr_r[114] *= scale_r;
      rr_r[127] *= scale_r;
      rr_r[136] *= scale_r;
      rr_r[151] *= scale_r;
      rr_r[162] *= scale_r;
      rr_r[175] *= scale_r;
      rr_r[184] *= scale_r;
      rr_r[232] *= scale_r;
      rr_r[233] *= scale_r;
      rr_r[234] *= scale_r;
    }
  }
  // Stiff species H2O2
  {
    double ddot = rr_f[12] + rr_f[13] + rr_f[14] + rr_f[15] + rr_f[16] + 
      rr_f[72] + rr_f[103] + rr_f[133] + rr_r[10] + rr_r[11] + rr_r[17] + 
      rr_r[63] + rr_r[131] + rr_r[156] + rr_r[181] + rr_r[244] + rr_r[245] + 
      rr_r[246]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[7])
    {
      double cdot = rr_f[10] + rr_f[11] + rr_f[17] + rr_f[63] + rr_f[131] + 
        rr_f[156] + rr_f[181] + rr_f[244] + rr_f[245] + rr_f[246] + rr_r[12] + 
        rr_r[13] + rr_r[14] + rr_r[15] + rr_r[16] + rr_r[72] + rr_r[103] + 
        rr_r[133]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[7] * 0.02939901936631002;
      double c0 = mole_frac[7] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[7]- c0) / dt) 
        * recip_ddot; 
      rr_f[12] *= scale_r;
      rr_f[13] *= scale_r;
      rr_f[14] *= scale_r;
      rr_f[15] *= scale_r;
      rr_f[16] *= scale_r;
      rr_f[72] *= scale_r;
      rr_f[103] *= scale_r;
      rr_f[133] *= scale_r;
      rr_r[10] *= scale_r;
      rr_r[11] *= scale_r;
      rr_r[17] *= scale_r;
      rr_r[63] *= scale_r;
      rr_r[131] *= scale_r;
      rr_r[156] *= scale_r;
      rr_r[181] *= scale_r;
      rr_r[244] *= scale_r;
      rr_r[245] *= scale_r;
      rr_r[246] *= scale_r;
    }
  }
  // Stiff species CH3
  {
    double ddot = rr_f[64] + rr_f[65] + rr_f[66] + rr_f[67] + rr_f[68] + 
      rr_f[69] + rr_f[70] + rr_f[71] + rr_f[72] + rr_f[73] + rr_f[74] + rr_f[75] 
      + rr_f[76] + rr_f[77] + rr_f[77] + rr_f[78] + rr_f[78] + rr_f[94] + 
      rr_f[106] + rr_f[107] + rr_f[108] + rr_f[122] + rr_f[140] + rr_f[148] + 
      rr_f[157] + rr_f[167] + rr_f[182] + rr_f[189] + rr_f[247] + rr_f[248] + 
      rr_f[249] + rr_r[39] + rr_r[42] + rr_r[51] + rr_r[80] + rr_r[85] + 
      rr_r[86] + rr_r[87] + rr_r[88] + rr_r[88] + rr_r[89] + rr_r[89] + rr_r[92] 
      + rr_r[97] + rr_r[110] + rr_r[111] + rr_r[116] + rr_r[123] + rr_r[128] + 
      rr_r[132] + rr_r[135] + rr_r[139] + rr_r[150] + rr_r[161] + rr_r[169] + 
      rr_r[174]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[9])
    {
      double cdot = rr_f[39] + rr_f[42] + rr_f[51] + rr_f[80] + rr_f[85] + 
        rr_f[86] + rr_f[87] + rr_f[88] + rr_f[88] + rr_f[89] + rr_f[89] + 
        rr_f[92] + rr_f[97] + rr_f[110] + rr_f[111] + rr_f[116] + rr_f[123] + 
        rr_f[128] + rr_f[132] + rr_f[135] + rr_f[139] + rr_f[150] + rr_f[161] + 
        rr_f[169] + rr_f[174] + rr_r[64] + rr_r[65] + rr_r[66] + rr_r[67] + 
        rr_r[68] + rr_r[69] + rr_r[70] + rr_r[71] + rr_r[72] + rr_r[73] + 
        rr_r[74] + rr_r[75] + rr_r[76] + rr_r[77] + rr_r[77] + rr_r[78] + 
        rr_r[78] + rr_r[94] + rr_r[106] + rr_r[107] + rr_r[108] + rr_r[122] + 
        rr_r[140] + rr_r[148] + rr_r[157] + rr_r[167] + rr_r[182] + rr_r[189] + 
        rr_r[247] + rr_r[248] + rr_r[249]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[9] * 0.06651120780362699;
      double c0 = mole_frac[9] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[9]- c0) / dt) 
        * recip_ddot; 
      rr_f[64] *= scale_r;
      rr_f[65] *= scale_r;
      rr_f[66] *= scale_r;
      rr_f[67] *= scale_r;
      rr_f[68] *= scale_r;
      rr_f[69] *= scale_r;
      rr_f[70] *= scale_r;
      rr_f[71] *= scale_r;
      rr_f[72] *= scale_r;
      rr_f[73] *= scale_r;
      rr_f[74] *= scale_r;
      rr_f[75] *= scale_r;
      rr_f[76] *= scale_r;
      rr_f[77] *= scale_r;
      rr_f[77] *= scale_r;
      rr_f[78] *= scale_r;
      rr_f[78] *= scale_r;
      rr_f[94] *= scale_r;
      rr_f[106] *= scale_r;
      rr_f[107] *= scale_r;
      rr_f[108] *= scale_r;
      rr_f[122] *= scale_r;
      rr_f[140] *= scale_r;
      rr_f[148] *= scale_r;
      rr_f[157] *= scale_r;
      rr_f[167] *= scale_r;
      rr_f[182] *= scale_r;
      rr_f[189] *= scale_r;
      rr_f[247] *= scale_r;
      rr_f[248] *= scale_r;
      rr_f[249] *= scale_r;
      rr_r[39] *= scale_r;
      rr_r[42] *= scale_r;
      rr_r[51] *= scale_r;
      rr_r[80] *= scale_r;
      rr_r[85] *= scale_r;
      rr_r[86] *= scale_r;
      rr_r[87] *= scale_r;
      rr_r[88] *= scale_r;
      rr_r[88] *= scale_r;
      rr_r[89] *= scale_r;
      rr_r[89] *= scale_r;
      rr_r[92] *= scale_r;
      rr_r[97] *= scale_r;
      rr_r[110] *= scale_r;
      rr_r[111] *= scale_r;
      rr_r[116] *= scale_r;
      rr_r[123] *= scale_r;
      rr_r[128] *= scale_r;
      rr_r[132] *= scale_r;
      rr_r[135] *= scale_r;
      rr_r[139] *= scale_r;
      rr_r[150] *= scale_r;
      rr_r[161] *= scale_r;
      rr_r[169] *= scale_r;
      rr_r[174] *= scale_r;
    }
  }
  // Stiff species CH4
  {
    double ddot = rr_f[85] + rr_f[86] + rr_f[87] + rr_f[88] + rr_f[89] + 
      rr_r[64] + rr_r[70] + rr_r[72] + rr_r[73] + rr_r[74] + rr_r[106] + 
      rr_r[122] + rr_r[140] + rr_r[157] + rr_r[167] + rr_r[182] + rr_r[189] + 
      rr_r[247] + rr_r[248] + rr_r[249]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[10])
    {
      double cdot = rr_f[64] + rr_f[70] + rr_f[72] + rr_f[73] + rr_f[74] + 
        rr_f[106] + rr_f[122] + rr_f[140] + rr_f[157] + rr_f[167] + rr_f[182] + 
        rr_f[189] + rr_f[247] + rr_f[248] + rr_f[249] + rr_r[85] + rr_r[86] + 
        rr_r[87] + rr_r[88] + rr_r[89]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[10] * 0.06233236489615739;
      double c0 = mole_frac[10] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[10]- c0) / dt) 
        * recip_ddot; 
      rr_f[85] *= scale_r;
      rr_f[86] *= scale_r;
      rr_f[87] *= scale_r;
      rr_f[88] *= scale_r;
      rr_f[89] *= scale_r;
      rr_r[64] *= scale_r;
      rr_r[70] *= scale_r;
      rr_r[72] *= scale_r;
      rr_r[73] *= scale_r;
      rr_r[74] *= scale_r;
      rr_r[106] *= scale_r;
      rr_r[122] *= scale_r;
      rr_r[140] *= scale_r;
      rr_r[157] *= scale_r;
      rr_r[167] *= scale_r;
      rr_r[182] *= scale_r;
      rr_r[189] *= scale_r;
      rr_r[247] *= scale_r;
      rr_r[248] *= scale_r;
      rr_r[249] *= scale_r;
    }
  }
  // Stiff species CH2O
  {
    double ddot = rr_f[63] + rr_f[58] + rr_f[59] + rr_f[60] + rr_f[61] + 
      rr_f[62] + rr_f[74] + rr_r[37] + rr_r[38] + rr_r[41] + rr_r[45] + rr_r[50] 
      + rr_r[57] + rr_r[65] + rr_r[69] + rr_r[79] + rr_r[82] + rr_r[83] + 
      rr_r[84] + rr_r[101] + rr_r[112] + rr_r[117] + rr_r[128] + rr_r[132] + 
      rr_r[146] + rr_r[163] + rr_r[166] + rr_r[170] + rr_r[185] + rr_r[188]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[11])
    {
      double cdot = rr_f[37] + rr_f[38] + rr_f[41] + rr_f[45] + rr_f[50] + 
        rr_f[57] + rr_f[65] + rr_f[69] + rr_f[79] + rr_f[82] + rr_f[83] + 
        rr_f[84] + rr_f[101] + rr_f[112] + rr_f[117] + rr_f[128] + rr_f[132] + 
        rr_f[146] + rr_f[163] + rr_f[166] + rr_f[170] + rr_f[185] + rr_f[188] + 
        rr_r[63] + rr_r[58] + rr_r[59] + rr_r[60] + rr_r[61] + rr_r[62] + 
        rr_r[74]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[11] * 0.03330392596670473;
      double c0 = mole_frac[11] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[11]- c0) / dt) 
        * recip_ddot; 
      rr_f[63] *= scale_r;
      rr_f[58] *= scale_r;
      rr_f[59] *= scale_r;
      rr_f[60] *= scale_r;
      rr_f[61] *= scale_r;
      rr_f[62] *= scale_r;
      rr_f[74] *= scale_r;
      rr_r[37] *= scale_r;
      rr_r[38] *= scale_r;
      rr_r[41] *= scale_r;
      rr_r[45] *= scale_r;
      rr_r[50] *= scale_r;
      rr_r[57] *= scale_r;
      rr_r[65] *= scale_r;
      rr_r[69] *= scale_r;
      rr_r[79] *= scale_r;
      rr_r[82] *= scale_r;
      rr_r[83] *= scale_r;
      rr_r[84] *= scale_r;
      rr_r[101] *= scale_r;
      rr_r[112] *= scale_r;
      rr_r[117] *= scale_r;
      rr_r[128] *= scale_r;
      rr_r[132] *= scale_r;
      rr_r[146] *= scale_r;
      rr_r[163] *= scale_r;
      rr_r[166] *= scale_r;
      rr_r[170] *= scale_r;
      rr_r[185] *= scale_r;
      rr_r[188] *= scale_r;
    }
  }
  // Stiff species C2H4
  {
    double ddot = rr_f[113] + rr_f[114] + rr_f[115] + rr_f[116] + rr_f[117] + 
      rr_f[118] + rr_f[119] + rr_f[120] + rr_f[121] + rr_f[122] + rr_f[124] + 
      rr_f[125] + rr_f[195] + rr_f[199] + rr_f[203] + rr_f[207] + rr_f[211] + 
      rr_f[215] + rr_f[217] + rr_r[129] + rr_r[131] + rr_r[75] + rr_r[76] + 
      rr_r[95] + rr_r[103] + rr_r[104] + rr_r[109] + rr_r[123] + rr_r[127] + 
      rr_r[150] + rr_r[158] + rr_r[173] + rr_r[193] + rr_r[197] + rr_r[201] + 
      rr_r[205] + rr_r[209] + rr_r[213]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[15])
    {
      double cdot = rr_f[129] + rr_f[131] + rr_f[75] + rr_f[76] + rr_f[95] + 
        rr_f[103] + rr_f[104] + rr_f[109] + rr_f[123] + rr_f[127] + rr_f[267] + 
        rr_f[267] + rr_f[267] + rr_f[150] + rr_f[158] + rr_f[173] + rr_f[190] + 
        rr_f[193] + rr_f[197] + rr_f[201] + rr_f[205] + rr_f[209] + rr_f[213] + 
        rr_r[113] + rr_r[114] + rr_r[115] + rr_r[116] + rr_r[117] + rr_r[118] + 
        rr_r[119] + rr_r[120] + rr_r[121] + rr_r[122] + rr_r[124] + rr_r[125] + 
        rr_r[195] + rr_r[199] + rr_r[203] + rr_r[207] + rr_r[211] + rr_r[215] + 
        rr_r[217]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[15] * 0.03564531203549703;
      double c0 = mole_frac[15] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[15]- c0) / dt) 
        * recip_ddot; 
      rr_f[113] *= scale_r;
      rr_f[114] *= scale_r;
      rr_f[115] *= scale_r;
      rr_f[116] *= scale_r;
      rr_f[117] *= scale_r;
      rr_f[118] *= scale_r;
      rr_f[119] *= scale_r;
      rr_f[120] *= scale_r;
      rr_f[121] *= scale_r;
      rr_f[122] *= scale_r;
      rr_f[124] *= scale_r;
      rr_f[125] *= scale_r;
      rr_f[195] *= scale_r;
      rr_f[199] *= scale_r;
      rr_f[203] *= scale_r;
      rr_f[207] *= scale_r;
      rr_f[211] *= scale_r;
      rr_f[215] *= scale_r;
      rr_f[217] *= scale_r;
      rr_r[129] *= scale_r;
      rr_r[131] *= scale_r;
      rr_r[75] *= scale_r;
      rr_r[76] *= scale_r;
      rr_r[95] *= scale_r;
      rr_r[103] *= scale_r;
      rr_r[104] *= scale_r;
      rr_r[109] *= scale_r;
      rr_r[123] *= scale_r;
      rr_r[127] *= scale_r;
      rr_r[150] *= scale_r;
      rr_r[158] *= scale_r;
      rr_r[173] *= scale_r;
      rr_r[193] *= scale_r;
      rr_r[197] *= scale_r;
      rr_r[201] *= scale_r;
      rr_r[205] *= scale_r;
      rr_r[209] *= scale_r;
      rr_r[213] *= scale_r;
    }
  }
  // Stiff species C2H6
  {
    double ddot = rr_f[136] + rr_f[137] + rr_f[138] + rr_f[139] + rr_f[140] + 
      rr_r[130] + rr_r[77] + rr_r[126] + rr_r[133]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[16])
    {
      double cdot = rr_f[130] + rr_f[77] + rr_f[126] + rr_f[133] + rr_r[136] + 
        rr_r[137] + rr_r[138] + rr_r[139] + rr_r[140]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[16] * 0.03325560390181349;
      double c0 = mole_frac[16] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[16]- c0) / dt) 
        * recip_ddot; 
      rr_f[136] *= scale_r;
      rr_f[137] *= scale_r;
      rr_f[138] *= scale_r;
      rr_f[139] *= scale_r;
      rr_f[140] *= scale_r;
      rr_r[130] *= scale_r;
      rr_r[77] *= scale_r;
      rr_r[126] *= scale_r;
      rr_r[133] *= scale_r;
    }
  }
  // Stiff species CH2CHO
  {
    double ddot = rr_f[110] + rr_f[111] + rr_f[112] + rr_r[100] + rr_r[102];
    if ((ddot * dt * 9.750641978812205) > mole_frac[17])
    {
      double cdot = rr_f[100] + rr_f[102] + rr_f[267] + rr_f[267] + rr_r[110] + 
        rr_r[111] + rr_r[112]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[17] * 0.02323117270262868;
      double c0 = mole_frac[17] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[17]- c0) / dt) 
        * recip_ddot; 
      rr_f[110] *= scale_r;
      rr_f[111] *= scale_r;
      rr_f[112] *= scale_r;
      rr_r[100] *= scale_r;
      rr_r[102] *= scale_r;
    }
  }
  // Stiff species aC3H5
  {
    double ddot = rr_f[141] + rr_f[142] + rr_f[143] + rr_f[144] + rr_f[145] + 
      rr_f[146] + rr_f[147] + rr_f[148] + rr_r[94] + rr_r[108] + rr_r[120] + 
      rr_r[121] + rr_r[135] + rr_r[151] + rr_r[154] + rr_r[155] + rr_r[156] + 
      rr_r[157] + rr_r[169] + rr_r[170]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[18])
    {
      double cdot = rr_f[94] + rr_f[108] + rr_f[120] + rr_f[121] + rr_f[135] + 
        rr_f[151] + rr_f[154] + rr_f[155] + rr_f[156] + rr_f[157] + rr_f[169] + 
        rr_f[170] + rr_f[190] + rr_r[141] + rr_r[142] + rr_r[143] + rr_r[144] + 
        rr_r[145] + rr_r[146] + rr_r[147] + rr_r[148]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[18] * 0.02434671672351625;
      double c0 = mole_frac[18] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[18]- c0) / dt) 
        * recip_ddot; 
      rr_f[141] *= scale_r;
      rr_f[142] *= scale_r;
      rr_f[143] *= scale_r;
      rr_f[144] *= scale_r;
      rr_f[145] *= scale_r;
      rr_f[146] *= scale_r;
      rr_f[147] *= scale_r;
      rr_f[148] *= scale_r;
      rr_r[94] *= scale_r;
      rr_r[108] *= scale_r;
      rr_r[120] *= scale_r;
      rr_r[121] *= scale_r;
      rr_r[135] *= scale_r;
      rr_r[151] *= scale_r;
      rr_r[154] *= scale_r;
      rr_r[155] *= scale_r;
      rr_r[156] *= scale_r;
      rr_r[157] *= scale_r;
      rr_r[169] *= scale_r;
      rr_r[170] *= scale_r;
    }
  }
  // Stiff species C3H6
  {
    double ddot = rr_f[149] + rr_f[150] + rr_f[151] + rr_f[152] + rr_f[153] + 
      rr_f[154] + rr_f[155] + rr_f[156] + rr_f[157] + rr_f[219] + rr_r[107] + 
      rr_r[141] + rr_r[145] + rr_r[147] + rr_r[162] + rr_r[164] + rr_r[165] + 
      rr_r[167] + rr_r[174] + rr_r[194] + rr_r[198] + rr_r[202] + rr_r[206] + 
      rr_r[210] + rr_r[214]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[19])
    {
      double cdot = rr_f[107] + rr_f[141] + rr_f[145] + rr_f[147] + rr_f[162] + 
        rr_f[164] + rr_f[165] + rr_f[167] + rr_f[174] + rr_f[191] + rr_f[194] + 
        rr_f[198] + rr_f[202] + rr_f[206] + rr_f[210] + rr_f[214] + rr_r[149] + 
        rr_r[150] + rr_r[151] + rr_r[152] + rr_r[153] + rr_r[154] + rr_r[155] + 
        rr_r[156] + rr_r[157] + rr_r[219]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[19] * 0.02376354135699802;
      double c0 = mole_frac[19] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[19]- c0) / dt) 
        * recip_ddot; 
      rr_f[149] *= scale_r;
      rr_f[150] *= scale_r;
      rr_f[151] *= scale_r;
      rr_f[152] *= scale_r;
      rr_f[153] *= scale_r;
      rr_f[154] *= scale_r;
      rr_f[155] *= scale_r;
      rr_f[156] *= scale_r;
      rr_f[157] *= scale_r;
      rr_f[219] *= scale_r;
      rr_r[107] *= scale_r;
      rr_r[141] *= scale_r;
      rr_r[145] *= scale_r;
      rr_r[147] *= scale_r;
      rr_r[162] *= scale_r;
      rr_r[164] *= scale_r;
      rr_r[165] *= scale_r;
      rr_r[167] *= scale_r;
      rr_r[174] *= scale_r;
      rr_r[194] *= scale_r;
      rr_r[198] *= scale_r;
      rr_r[202] *= scale_r;
      rr_r[206] *= scale_r;
      rr_r[210] *= scale_r;
      rr_r[214] *= scale_r;
    }
  }
  // Stiff species C2H3CHO
  {
    double ddot = rr_f[158] + rr_f[159] + rr_f[160] + rr_r[105] + rr_r[142] + 
      rr_r[143] + rr_r[144] + rr_r[152]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[20])
    {
      double cdot = rr_f[105] + rr_f[142] + rr_f[143] + rr_f[144] + rr_f[152] + 
        rr_r[158] + rr_r[159] + rr_r[160]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[20] * 0.01783652574443861;
      double c0 = mole_frac[20] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[20]- c0) / dt) 
        * recip_ddot; 
      rr_f[158] *= scale_r;
      rr_f[159] *= scale_r;
      rr_f[160] *= scale_r;
      rr_r[105] *= scale_r;
      rr_r[142] *= scale_r;
      rr_r[143] *= scale_r;
      rr_r[144] *= scale_r;
      rr_r[152] *= scale_r;
    }
  }
  // Stiff species C4H81
  {
    double ddot = rr_f[172] + rr_f[173] + rr_f[174] + rr_f[175] + rr_f[176] + 
      rr_f[177] + rr_f[178] + rr_f[179] + rr_f[180] + rr_f[181] + rr_f[182] + 
      rr_f[220] + rr_r[134] + rr_r[148] + rr_r[168] + rr_r[171] + rr_r[184] + 
      rr_r[186] + rr_r[187] + rr_r[189]; 
    if ((ddot * dt * 9.750641978812205) > mole_frac[22])
    {
      double cdot = rr_f[134] + rr_f[148] + rr_f[168] + rr_f[171] + rr_f[184] + 
        rr_f[186] + rr_f[187] + rr_f[189] + rr_r[172] + rr_r[173] + rr_r[174] + 
        rr_r[175] + rr_r[176] + rr_r[177] + rr_r[178] + rr_r[179] + rr_r[180] + 
        rr_r[181] + rr_r[182] + rr_r[220]; 
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[22] * 0.01782265601774851;
      double c0 = mole_frac[22] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[22]- c0) / dt) 
        * recip_ddot; 
      rr_f[172] *= scale_r;
      rr_f[173] *= scale_r;
      rr_f[174] *= scale_r;
      rr_f[175] *= scale_r;
      rr_f[176] *= scale_r;
      rr_f[177] *= scale_r;
      rr_f[178] *= scale_r;
      rr_f[179] *= scale_r;
      rr_f[180] *= scale_r;
      rr_f[181] *= scale_r;
      rr_f[182] *= scale_r;
      rr_f[220] *= scale_r;
      rr_r[134] *= scale_r;
      rr_r[148] *= scale_r;
      rr_r[168] *= scale_r;
      rr_r[171] *= scale_r;
      rr_r[184] *= scale_r;
      rr_r[186] *= scale_r;
      rr_r[187] *= scale_r;
      rr_r[189] *= scale_r;
    }
  }
  // Stiff species C5H9
  {
    double ddot = rr_f[190] + rr_f[191] + rr_r[216];
    if ((ddot * dt * 9.750641978812205) > mole_frac[23])
    {
      double cdot = rr_f[216];
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[23] * 0.01446602711396394;
      double c0 = mole_frac[23] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[23]- c0) / dt) 
        * recip_ddot; 
      rr_f[190] *= scale_r;
      rr_f[191] *= scale_r;
      rr_r[216] *= scale_r;
    }
  }
  // Stiff species PXC9H19
  {
    double ddot = rr_f[219] + rr_f[228] + rr_r[208] + rr_r[211];
    if ((ddot * dt * 9.750641978812205) > mole_frac[29])
    {
      double cdot = rr_f[208] + rr_f[211] + rr_r[219] + rr_r[228];
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[29] * 7.858436243485159e-03;
      double c0 = mole_frac[29] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[29]- c0) / dt) 
        * recip_ddot; 
      rr_f[219] *= scale_r;
      rr_f[228] *= scale_r;
      rr_r[208] *= scale_r;
      rr_r[211] *= scale_r;
    }
  }
  // Stiff species C12H24
  {
    double ddot = rr_f[216] + rr_f[259] + rr_f[261] + rr_f[263];
    if ((ddot * dt * 9.750641978812205) > mole_frac[31])
    {
      double cdot = rr_f[258] + rr_f[260] + rr_f[262] + rr_r[216];
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[31] * 5.940885339249504e-03;
      double c0 = mole_frac[31] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[31]- c0) / dt) 
        * recip_ddot; 
      rr_f[216] *= scale_r;
      rr_f[259] *= scale_r;
      rr_f[261] *= scale_r;
      rr_f[263] *= scale_r;
    }
  }
  // Stiff species C12H25O2
  {
    double ddot = rr_f[251] + rr_f[253] + rr_f[255] + rr_f[256];
    if ((ddot * dt * 9.750641978812205) > mole_frac[32])
    {
      double cdot = rr_f[257] + rr_f[250] + rr_f[252] + rr_f[254];
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[32] * 4.966924011277897e-03;
      double c0 = mole_frac[32] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[32]- c0) / dt) 
        * recip_ddot; 
      rr_f[251] *= scale_r;
      rr_f[253] *= scale_r;
      rr_f[255] *= scale_r;
      rr_f[256] *= scale_r;
    }
  }
  // Stiff species OC12H23OOH
  {
    double ddot = rr_f[267] + rr_r[266];
    if ((ddot * dt * 9.750641978812205) > mole_frac[33])
    {
      double cdot = rr_f[266];
      double recip_ddot = 1.0/ddot;
      double part_sum = cdot + diffusion[33] * 4.622710972207892e-03;
      double c0 = mole_frac[33] * part_sum * recip_ddot;
      double scale_r = (part_sum + 0.102557349779939 * (mole_frac[33]- c0) / dt) 
        * recip_ddot; 
      rr_f[267] *= scale_r;
      rr_r[266] *= scale_r;
    }
  }
  double ropl[268];
  for (int i = 0; i < 268; i++)
  {
    ropl[i] = rr_f[i] - rr_r[i];
  }
  // 0. NC12H26
  wdot[0] = ropl[227] + ropl[228] + ropl[229] + ropl[230] + ropl[231] - 
    ropl[232] - ropl[233] - ropl[234] - ropl[235] - ropl[236] - ropl[237] - 
    ropl[238] - ropl[239] - ropl[240] - ropl[241] - ropl[242] - ropl[243] - 
    ropl[244] - ropl[245] - ropl[246] - ropl[247] - ropl[248] - ropl[249]; 
  // 1. H
  wdot[1] = -ropl[0] + ropl[1] + ropl[2] - ropl[4] - ropl[5] + ropl[6] - ropl[8] 
    - ropl[12] - ropl[13] - 2.0*ropl[18] - ropl[19] - 2.0*ropl[21] - 
    2.0*ropl[22] - 2.0*ropl[23] - ropl[24] + ropl[25] + ropl[26] + ropl[30] - 
    ropl[31] + ropl[33] + ropl[36] - ropl[38] - ropl[39] + ropl[40] + ropl[41] + 
    ropl[42] + 2.0*ropl[44] + ropl[49] + ropl[50] + ropl[51] + ropl[52] - 
    ropl[58] - ropl[59] - ropl[64] + ropl[65] + ropl[75] + ropl[76] + ropl[78] - 
    ropl[79] - ropl[80] - ropl[81] - ropl[85] + ropl[90] - ropl[95] - ropl[96] + 
    ropl[108] - ropl[111] - ropl[113] - ropl[114] + ropl[120] + ropl[121] - 
    ropl[126] - ropl[127] - ropl[136] - ropl[141] + ropl[142] + 2.0*ropl[143] - 
    ropl[149] - ropl[150] - ropl[151] + 2.0*ropl[152] - ropl[158] - ropl[161] - 
    ropl[162] - ropl[168] - ropl[169] - ropl[172] - ropl[173] - ropl[174] - 
    ropl[175] - ropl[183] - ropl[184] - ropl[192] - ropl[193] - ropl[194] - 
    ropl[196] - ropl[197] - ropl[198] - ropl[200] - ropl[201] - ropl[202] - 
    ropl[204] - ropl[205] - ropl[206] - ropl[208] - ropl[209] - ropl[210] - 
    ropl[212] - ropl[213] - ropl[214] - ropl[232] - ropl[233] - ropl[234]; 
  // 2. O
  wdot[2] = ropl[0] - ropl[1] + ropl[3] + ropl[8] - ropl[9] - ropl[14] - 
    2.0*ropl[20] - ropl[24] - ropl[28] + ropl[29] - ropl[32] - ropl[33] - 
    ropl[40] - ropl[48] - ropl[49] - ropl[60] - ropl[65] + ropl[68] - ropl[82] - 
    ropl[86] - ropl[91] - ropl[97] + ropl[100] - ropl[115] - ropl[116] - 
    ropl[117] - ropl[128] - ropl[137] - ropl[142] - ropl[152] - ropl[153] - 
    ropl[154] - ropl[159] - ropl[163] - ropl[176] - ropl[177] - ropl[178] - 
    ropl[185] - ropl[235] - ropl[236] - ropl[237]; 
  // 3. OH
  wdot[3] = ropl[0] + ropl[1] - ropl[2] - 2.0*ropl[3] + 2.0*ropl[5] - ropl[7] + 
    ropl[9] + ropl[12] + ropl[14] - ropl[15] - ropl[16] - 2.0*ropl[17] - 
    ropl[19] + ropl[24] - ropl[25] - ropl[26] + ropl[27] + ropl[32] - ropl[34] - 
    ropl[41] + ropl[43] + ropl[45] - ropl[50] + ropl[52] + ropl[60] - ropl[61] - 
    ropl[66] - ropl[67] + ropl[69] + ropl[71] + ropl[80] + ropl[82] - ropl[83] + 
    ropl[86] - ropl[87] - ropl[92] - ropl[98] + ropl[102] + ropl[112] + 
    ropl[115] - ropl[118] + ropl[267] + ropl[132] + ropl[137] - ropl[138] - 
    ropl[143] + ropl[144] + ropl[146] + ropl[154] - ropl[155] + ropl[159] - 
    ropl[160] - ropl[164] + ropl[166] + ropl[170] + ropl[177] + ropl[178] - 
    ropl[179] - ropl[186] + ropl[188] + ropl[235] + ropl[236] + ropl[237] - 
    ropl[238] - ropl[239] - ropl[240] + ropl[266]; 
  // 4. HO2
  wdot[4] = ropl[4] - ropl[5] + ropl[6] - ropl[7] - ropl[8] - ropl[9] - 
    2.0*ropl[10] - 2.0*ropl[11] + ropl[13] + ropl[14] + ropl[15] + ropl[16] - 
    ropl[27] - ropl[63] + ropl[35] - ropl[45] + ropl[62] + ropl[129] - ropl[130] 
    - ropl[131] - ropl[70] - ropl[71] + ropl[72] + ropl[84] + ropl[99] - 
    ropl[102] + ropl[103] + ropl[124] - ropl[132] + ropl[133] - ropl[145] - 
    ropl[146] - ropl[156] + ropl[165] - ropl[166] - ropl[170] + ropl[180] - 
    ropl[181] + ropl[187] - ropl[188] + ropl[241] + ropl[242] + ropl[243] - 
    ropl[244] - ropl[245] - ropl[246] + ropl[258] - ropl[259] + ropl[260] - 
    ropl[261] + ropl[262] - ropl[263]; 
  // 5. H2
  wdot[5] = -ropl[1] - ropl[2] - ropl[6] + ropl[13] + ropl[18] + ropl[21] + 
    ropl[22] + ropl[23] + ropl[31] - ropl[37] - ropl[42] + ropl[46] + ropl[48] - 
    ropl[51] + ropl[59] + ropl[79] + ropl[85] + ropl[96] + ropl[114] + ropl[127] 
    + ropl[136] + ropl[151] + ropl[162] + ropl[175] + ropl[184] + ropl[232] + 
    ropl[233] + ropl[234]; 
  // 6. H2O
  wdot[6] = ropl[2] + ropl[3] + ropl[7] + ropl[8] + ropl[12] + ropl[15] + 
    ropl[16] + ropl[19] + ropl[34] + ropl[53] + ropl[61] + ropl[66] + ropl[67] + 
    ropl[81] + ropl[83] + ropl[87] + ropl[98] + ropl[118] + ropl[138] + 
    ropl[155] + ropl[160] + ropl[164] + ropl[179] + ropl[186] + ropl[238] + 
    ropl[239] + ropl[240]; 
  // 7. H2O2
  wdot[7] = ropl[10] + ropl[11] - ropl[12] - ropl[13] - ropl[14] - ropl[15] - 
    ropl[16] + ropl[17] + ropl[63] + ropl[131] - ropl[72] - ropl[103] - 
    ropl[133] + ropl[156] + ropl[181] + ropl[244] + ropl[245] + ropl[246]; 
  // 8. O2
  wdot[8] = -ropl[0] - ropl[4] - ropl[6] + ropl[7] + ropl[9] + ropl[10] + 
    ropl[11] + ropl[20] - ropl[29] - ropl[35] - ropl[43] - ropl[44] - ropl[52] - 
    ropl[53] - ropl[62] - ropl[129] + ropl[130] - ropl[68] - ropl[69] + ropl[70] 
    - ropl[84] - ropl[99] - ropl[100] - ropl[101] - ropl[112] - ropl[124] - 
    ropl[144] + ropl[145] - ropl[165] - ropl[180] - ropl[187] - ropl[241] - 
    ropl[242] - ropl[243] - ropl[250] + ropl[251] - ropl[252] + ropl[253] - 
    ropl[254] + ropl[255] - ropl[258] + ropl[259] - ropl[260] + ropl[261] - 
    ropl[262] + ropl[263] - ropl[264] + ropl[265]; 
  // 9. CH3
  wdot[9] = ropl[39] + ropl[42] + ropl[51] - ropl[64] - ropl[65] - ropl[66] - 
    ropl[67] - ropl[68] - ropl[69] - ropl[70] - ropl[71] - ropl[72] - ropl[73] - 
    ropl[74] - ropl[75] - ropl[76] - 2.0*ropl[77] - 2.0*ropl[78] + ropl[80] + 
    ropl[85] + ropl[86] + ropl[87] + 2.0*ropl[88] + 2.0*ropl[89] + ropl[92] - 
    ropl[94] + ropl[97] - ropl[106] - ropl[107] - ropl[108] + ropl[110] + 
    ropl[111] + ropl[116] - ropl[122] + ropl[123] + ropl[128] + ropl[132] + 
    ropl[135] + ropl[139] - ropl[140] - ropl[148] + ropl[150] - ropl[157] + 
    ropl[161] - ropl[167] + ropl[169] + ropl[174] - ropl[182] - ropl[189] - 
    ropl[247] - ropl[248] - ropl[249]; 
  // 10. CH4
  wdot[10] = ropl[64] + ropl[70] + ropl[72] + ropl[73] + ropl[74] - ropl[85] - 
    ropl[86] - ropl[87] - ropl[88] - ropl[89] + ropl[106] + ropl[122] + 
    ropl[140] + ropl[157] + ropl[167] + ropl[182] + ropl[189] + ropl[247] + 
    ropl[248] + ropl[249]; 
  // 11. CH2O
  wdot[11] = -ropl[63] + ropl[37] + ropl[38] + ropl[41] + ropl[45] + ropl[50] + 
    ropl[57] - ropl[58] - ropl[59] - ropl[60] - ropl[61] - ropl[62] + ropl[65] + 
    ropl[69] - ropl[74] + ropl[79] + ropl[82] + ropl[83] + ropl[84] + ropl[101] 
    + ropl[112] + ropl[117] + ropl[128] + ropl[132] + ropl[146] + ropl[163] + 
    ropl[166] + ropl[170] + ropl[185] + ropl[188]; 
  // 12. CO
  wdot[12] = -ropl[25] - ropl[26] - ropl[27] - ropl[28] - ropl[29] + ropl[30] + 
    ropl[31] + ropl[32] + ropl[34] + ropl[35] + ropl[36] - ropl[37] + ropl[48] + 
    ropl[52] + ropl[53] + ropl[57] + ropl[73] + ropl[91] + ropl[92] + ropl[93] + 
    ropl[97] + ropl[104] + ropl[110] + ropl[112] + ropl[119] + ropl[147] + 
    ropl[159] + ropl[160] + ropl[171]; 
  // 13. CO2
  wdot[13] = ropl[25] + ropl[26] + ropl[27] + ropl[28] + ropl[29] + ropl[33] + 
    ropl[44] - ropl[57]; 
  // 14. C2H2
  wdot[14] = ropl[46] + ropl[90] - ropl[91] - ropl[92] - ropl[93] - ropl[94] + 
    ropl[96] + ropl[98] + ropl[99] + ropl[106] + ropl[109]; 
  // 15. C2H4
  wdot[15] = ropl[129] + ropl[131] + ropl[75] + ropl[76] + ropl[95] + ropl[103] 
    + ropl[104] + ropl[109] - ropl[113] - ropl[114] - ropl[115] - ropl[116] - 
    ropl[117] - ropl[118] - ropl[119] - ropl[120] - ropl[121] - ropl[122] + 
    ropl[123] - ropl[124] - ropl[125] + ropl[127] + 3.0*ropl[267] + ropl[150] + 
    ropl[158] + ropl[173] + ropl[190] + ropl[193] - ropl[195] + ropl[197] - 
    ropl[199] + ropl[201] - ropl[203] + ropl[205] - ropl[207] + ropl[209] - 
    ropl[211] + ropl[213] - ropl[215] - ropl[217]; 
  // 16. C2H6
  wdot[16] = ropl[130] + ropl[77] + ropl[126] + ropl[133] - ropl[136] - 
    ropl[137] - ropl[138] - ropl[139] - ropl[140]; 
  // 17. CH2CHO
  wdot[17] = ropl[100] + ropl[102] - ropl[110] - ropl[111] - ropl[112] + 
    2.0*ropl[267]; 
  // 18. AC3H5
  wdot[18] = ropl[94] + ropl[108] + ropl[120] + ropl[121] + ropl[135] - 
    ropl[141] - ropl[142] - ropl[143] - ropl[144] - ropl[145] - ropl[146] - 
    ropl[147] - ropl[148] + ropl[151] + ropl[154] + ropl[155] + ropl[156] + 
    ropl[157] + ropl[169] + ropl[170] + ropl[190]; 
  // 19. C3H6
  wdot[19] = ropl[107] + ropl[141] + ropl[145] + ropl[147] - ropl[149] - 
    ropl[150] - ropl[151] - ropl[152] - ropl[153] - ropl[154] - ropl[155] - 
    ropl[156] - ropl[157] + ropl[162] + ropl[164] + ropl[165] + ropl[167] + 
    ropl[174] + ropl[191] + ropl[194] + ropl[198] + ropl[202] + ropl[206] + 
    ropl[210] + ropl[214] - ropl[219]; 
  // 20. C2H3CHO
  wdot[20] = ropl[105] + ropl[142] + ropl[143] + ropl[144] + ropl[152] - 
    ropl[158] - ropl[159] - ropl[160]; 
  // 21. C4H7
  wdot[21] = ropl[125] - ropl[168] - ropl[169] - ropl[170] - ropl[171] + 
    ropl[175] + ropl[177] + ropl[178] + ropl[179] + ropl[180] + ropl[181] + 
    ropl[182]; 
  // 22. C4H81
  wdot[22] = ropl[134] + ropl[148] + ropl[168] + ropl[171] - ropl[172] - 
    ropl[173] - ropl[174] - ropl[175] - ropl[176] - ropl[177] - ropl[178] - 
    ropl[179] - ropl[180] - ropl[181] - ropl[182] + ropl[184] + ropl[186] + 
    ropl[187] + ropl[189] - ropl[220]; 
  // 23. C5H9
  wdot[23] = -ropl[190] - ropl[191] + ropl[216];
  // 24. C5H10
  wdot[24] = -ropl[192] - ropl[193] - ropl[194] - ropl[221];
  // 25. C6H12
  wdot[25] = -ropl[196] - ropl[197] - ropl[198] - ropl[223];
  // 26. C7H14
  wdot[26] = -ropl[200] - ropl[201] - ropl[202] - ropl[225];
  // 27. C8H16
  wdot[27] = -ropl[204] - ropl[205] - ropl[206] - ropl[226];
  // 28. C9H18
  wdot[28] = -ropl[208] - ropl[209] - ropl[210] - ropl[224];
  // 29. PXC9H19
  wdot[29] = ropl[208] + ropl[211] - ropl[219] - ropl[228];
  // 30. C10H20
  wdot[30] = -ropl[212] - ropl[213] - ropl[214] - ropl[222];
  // 31. C12H24
  wdot[31] = -ropl[216] + ropl[258] - ropl[259] + ropl[260] - ropl[261] + 
    ropl[262] - ropl[263]; 
  // 32. C12H25O2
  wdot[32] = ropl[257] + ropl[250] - ropl[251] + ropl[252] - ropl[253] + 
    ropl[254] - ropl[255] - ropl[256]; 
  // 33. OC12H23OOH
  wdot[33] = -ropl[267] + ropl[266];
  // 34. N2
  wdot[34] = 0.0;
}

