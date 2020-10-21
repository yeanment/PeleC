#include "chemistry_file.H"
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

static AMREX_GPU_DEVICE_MANAGED double molecular_masses[21] = {2.01594, 1.00797, 15.9994, 31.9988, 
  17.00737, 18.01534, 33.00677, 14.02709, 14.02709, 15.03506, 16.04303, 
  28.01055, 44.00995, 29.01852, 30.02649, 31.03446, 28.05418, 29.06215, 
  30.07012, 28.0134, 39.948}; 


static AMREX_GPU_DEVICE_MANAGED double recip_molecular_masses[21] = {0.4960465093207139, 
  0.9920930186414277, 0.06250234383789392, 0.03125117191894696, 
  0.05879803873262004, 0.05550825019122593, 0.03029681486555637, 
  0.07129062407099405, 0.07129062407099405, 0.06651120780362699, 
  0.06233236489615739, 0.03570083414998991, 0.02272213442641948, 
  0.03446075127194632, 0.03330392596670473, 0.03222224585186918, 
  0.03564531203549703, 0.0344090165386938, 0.03325560390181349, 
  0.03569720205330306, 0.02503254230499649}; 

AMREX_GPU_HOST_DEVICE
void base_getrates(const double pressure, const double temperature, const double 
  avmolwt, const double *mass_frac, double *wdot) 
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
  
  double cgspl[21];
  // Gibbs computation
  {
    const double &tk1 = temperature;
    double tklog = log(tk1);
    double tk2 = tk1 * tk1;
    double tk3 = tk1 * tk2;
    double tk4 = tk1 * tk3;
    double tk5 = tk1 * tk4;
    
    // Species H2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[0] = 3.3372792*tk1*(1-tklog) + 2.470123655e-05*tk2 + 
          -8.324279633333334e-08*tk3 + 1.496386616666666e-11*tk4 + 
          -1.00127688e-15*tk5 + (-950.158922 - tk1*-3.20502331); 
      }
      else
      {
        cgspl[0] = 2.34433112*tk1*(1-tklog) + -3.990260375e-03*tk2 + 
          3.2463585e-06*tk3 + -1.67976745e-09*tk4 + 3.688058804999999e-13*tk5 + 
          (-917.935173 - tk1*0.683010238); 
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
    // Species O2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[3] = 3.28253784*tk1*(1-tklog) + -7.4154377e-04*tk2 + 
          1.263277781666667e-07*tk3 + -1.745587958333333e-11*tk4 + 
          1.08358897e-15*tk5 + (-1.08845772e+03 - tk1*5.45323129); 
      }
      else
      {
        cgspl[3] = 3.78245636*tk1*(1-tklog) + 1.49836708e-03*tk2 + 
          -1.641217001666667e-06*tk3 + 8.067745908333333e-10*tk4 + 
          -1.621864185e-13*tk5 + (-1.06394356e+03 - tk1*3.65767573); 
      }
    }
    // Species OH
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[4] = 3.09288767*tk1*(1-tklog) + -2.74214858e-04*tk2 + 
          -2.108420466666667e-08*tk3 + 7.328846299999998e-12*tk4 + 
          -5.870618799999999e-16*tk5 + (3.858657e+03 - tk1*4.4766961); 
      }
      else
      {
        cgspl[4] = 3.99201543*tk1*(1-tklog) + 1.20065876e-03*tk2 + 
          -7.696564016666666e-07*tk3 + 3.234277774999999e-10*tk4 + 
          -6.820573499999999e-14*tk5 + (3.61508056e+03 - tk1*-0.103925458); 
      }
    }
    // Species H2O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[5] = 3.03399249*tk1*(1-tklog) + -1.08845902e-03*tk2 + 
          2.734541966666667e-08*tk3 + 8.086832249999998e-12*tk4 + 
          -8.410049599999998e-16*tk5 + (-3.00042971e+04 - tk1*4.9667701); 
      }
      else
      {
        cgspl[5] = 4.19864056*tk1*(1-tklog) + 1.01821705e-03*tk2 + 
          -1.086733685e-06*tk3 + 4.573308849999999e-10*tk4 + 
          -8.859890849999997e-14*tk5 + (-3.02937267e+04 - tk1*-0.849032208); 
      }
    }
    // Species HO2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[6] = 4.0172109*tk1*(1-tklog) + -1.119910065e-03*tk2 + 
          1.056096916666667e-07*tk3 + -9.520530833333331e-12*tk4 + 
          5.395426749999999e-16*tk5 + (111.856713 - tk1*3.78510215); 
      }
      else
      {
        cgspl[6] = 4.30179801*tk1*(1-tklog) + 2.374560255e-03*tk2 + 
          -3.526381516666667e-06*tk3 + 2.02303245e-09*tk4 + -4.64612562e-13*tk5 
          + (294.80804 - tk1*3.71666245); 
      }
    }
    // Species CH2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[7] = 2.87410113*tk1*(1-tklog) + -1.82819646e-03*tk2 + 
          2.348243283333334e-07*tk3 + -2.168162908333333e-11*tk4 + 
          9.386378349999998e-16*tk5 + (4.6263604e+04 - tk1*6.17119324); 
      }
      else
      {
        cgspl[7] = 3.76267867*tk1*(1-tklog) + -4.844360715e-04*tk2 + 
          -4.658164016666667e-07*tk3 + 3.209092941666666e-10*tk4 + 
          -8.437085949999999e-14*tk5 + (4.60040401e+04 - tk1*1.56253185); 
      }
    }
    // Species CH2(S)
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[8] = 2.29203842*tk1*(1-tklog) + -2.327943185e-03*tk2 + 
          3.353199116666667e-07*tk3 + -3.482549999999999e-11*tk4 + 
          1.698581825e-15*tk5 + (5.09259997e+04 - tk1*8.62650169); 
      }
      else
      {
        cgspl[8] = 4.19860411*tk1*(1-tklog) + 1.183307095e-03*tk2 + 
          -1.372160366666667e-06*tk3 + 5.573466508333332e-10*tk4 + 
          -9.715736849999998e-14*tk5 + (5.04968163e+04 - tk1*-0.769118967); 
      }
    }
    // Species CH3
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[9] = 2.28571772*tk1*(1-tklog) + -3.619950185e-03*tk2 + 
          4.978572466666668e-07*tk3 + -4.964038699999999e-11*tk4 + 
          2.33577197e-15*tk5 + (1.67755843e+04 - tk1*8.48007179); 
      }
      else
      {
        cgspl[9] = 3.6735904*tk1*(1-tklog) + -1.005475875e-03*tk2 + 
          -9.550364266666668e-07*tk3 + 5.725978541666665e-10*tk4 + 
          -1.27192867e-13*tk5 + (1.64449988e+04 - tk1*1.60456433); 
      }
    }
    // Species CH4
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[10] = 0.074851495*tk1*(1-tklog) + -6.69547335e-03*tk2 + 
          9.554763483333335e-07*tk3 + -1.019104458333333e-10*tk4 + 
          5.090761499999999e-15*tk5 + (-9.468344590000001e+03 - tk1*18.437318); 
      }
      else
      {
        cgspl[10] = 5.14987613*tk1*(1-tklog) + 6.8354894e-03*tk2 + 
          -8.19667665e-06*tk3 + 4.039525216666666e-09*tk4 + 
          -8.334697799999998e-13*tk5 + (-1.02466476e+04 - tk1*-4.64130376); 
      }
    }
    // Species CO
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[11] = 2.71518561*tk1*(1-tklog) + -1.031263715e-03*tk2 + 
          1.664709618333334e-07*tk3 + -1.9171084e-11*tk4 + 1.01823858e-15*tk5 + 
          (-1.41518724e+04 - tk1*7.81868772); 
      }
      else
      {
        cgspl[11] = 3.57953347*tk1*(1-tklog) + 3.0517684e-04*tk2 + 
          -1.69469055e-07*tk3 + -7.558382366666664e-11*tk4 + 
          4.522122494999999e-14*tk5 + (-1.4344086e+04 - tk1*3.50840928); 
      }
    }
    // Species CO2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[12] = 3.85746029*tk1*(1-tklog) + -2.20718513e-03*tk2 + 
          3.691356733333334e-07*tk3 + -4.362418233333333e-11*tk4 + 
          2.36042082e-15*tk5 + (-4.8759166e+04 - tk1*2.27163806); 
      }
      else
      {
        cgspl[12] = 2.35677352*tk1*(1-tklog) + -4.492298385e-03*tk2 + 
          1.187260448333333e-06*tk3 + -2.049325183333333e-10*tk4 + 
          7.184977399999998e-15*tk5 + (-4.83719697e+04 - tk1*9.90105222); 
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
        cgspl[15] = 3.770799*tk1*(1-tklog) + -3.9357485e-03*tk2 + 
          4.427306666666667e-07*tk3 + -3.287025833333333e-11*tk4 + 
          1.056308e-15*tk5 + (127.83252 - tk1*2.929575); 
      }
      else
      {
        cgspl[15] = 2.106204*tk1*(1-tklog) + -3.6082975e-03*tk2 + 
          -8.897453333333335e-07*tk3 + 6.148029999999998e-10*tk4 + 
          -1.037805e-13*tk5 + (978.6011 - tk1*13.152177); 
      }
    }
    // Species C2H4
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[16] = 2.03611116*tk1*(1-tklog) + -7.32270755e-03*tk2 + 
          1.118463191666667e-06*tk3 + -1.226857691666667e-10*tk4 + 
          6.285303049999998e-15*tk5 + (4.93988614e+03 - tk1*10.3053693); 
      }
      else
      {
        cgspl[16] = 3.95920148*tk1*(1-tklog) + 3.785261235e-03*tk2 + 
          -9.516504866666668e-06*tk3 + 5.763239608333332e-09*tk4 + 
          -1.349421865e-12*tk5 + (5.08977593e+03 - tk1*4.09733096); 
      }
    }
    // Species C2H5
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[17] = 1.95465642*tk1*(1-tklog) + -8.698636100000001e-03*tk2 + 
          1.330344446666667e-06*tk3 + -1.460147408333333e-10*tk4 + 
          7.482078799999998e-15*tk5 + (1.285752e+04 - tk1*13.4624343); 
      }
      else
      {
        cgspl[17] = 4.30646568*tk1*(1-tklog) + 2.09329446e-03*tk2 + 
          -8.28571345e-06*tk3 + 4.992721716666666e-09*tk4 + -1.15254502e-12*tk5 
          + (1.28416265e+04 - tk1*4.70720924); 
      }
    }
    // Species C2H6
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[18] = 1.0718815*tk1*(1-tklog) + -0.01084263385*tk2 + 
          1.67093445e-06*tk3 + -1.845100008333333e-10*tk4 + 
          9.500144499999998e-15*tk5 + (-1.14263932e+04 - tk1*15.1156107); 
      }
      else
      {
        cgspl[18] = 4.29142492*tk1*(1-tklog) + 2.75077135e-03*tk2 + 
          -9.990638133333334e-06*tk3 + 5.903885708333332e-09*tk4 + 
          -1.343428855e-12*tk5 + (-1.15222055e+04 - tk1*2.66682316); 
      }
    }
    // Species N2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[19] = 2.92664*tk1*(1-tklog) + -7.439884e-04*tk2 + 
          9.474600000000001e-08*tk3 + -8.414198333333332e-12*tk4 + 
          3.376675499999999e-16*tk5 + (-922.7977 - tk1*5.980528); 
      }
      else
      {
        cgspl[19] = 3.298677*tk1*(1-tklog) + -7.041202e-04*tk2 + 6.60537e-07*tk3 
          + -4.7012625e-10*tk4 + 1.222427e-13*tk5 + (-1.0208999e+03 - 
          tk1*3.950372); 
      }
    }
    // Species AR
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[20] = 2.5*tk1*(1-tklog) + -0.0*tk2 + -0.0*tk3 + -0.0*tk4 + 
          -0.0*tk5 + (-745.375 - tk1*4.366); 
      }
      else
      {
        cgspl[20] = 2.5*tk1*(1-tklog) + -0.0*tk2 + -0.0*tk3 + -0.0*tk4 + 
          -0.0*tk5 + (-745.375 - tk1*4.366); 
      }
    }
  }
  
  double mole_frac[21];
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
  }
  
  double thbctemp[7];
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
    thbctemp[0] = ctot + mole_frac[0] + 5.0*mole_frac[5] + mole_frac[10] + 
      0.5*mole_frac[11] + mole_frac[12] + 2.0*mole_frac[18] - 0.3*mole_frac[20]; 
    thbctemp[1] = ctot + mole_frac[0] + 5.0*mole_frac[3] + 5.0*mole_frac[5] + 
      mole_frac[10] + 0.5*mole_frac[11] + 2.5*mole_frac[12] + 2.0*mole_frac[18] 
      - 0.5*mole_frac[20]; 
    thbctemp[2] = ctot - mole_frac[3] - mole_frac[5] - 0.25*mole_frac[11] + 
      0.5*mole_frac[12] + 0.5*mole_frac[18] - mole_frac[19] - mole_frac[20]; 
    thbctemp[3] = ctot - mole_frac[0] - mole_frac[5] + mole_frac[10] - 
      mole_frac[12] + 2.0*mole_frac[18] - 0.37*mole_frac[20]; 
    thbctemp[4] = ctot - 0.27*mole_frac[0] + 2.65*mole_frac[5] + mole_frac[10] + 
      2.0*mole_frac[18] - 0.62*mole_frac[20]; 
    thbctemp[5] = ctot + mole_frac[0] + 5.0*mole_frac[5] + mole_frac[10] + 
      0.5*mole_frac[11] + mole_frac[12] + 2.0*mole_frac[18]; 
    thbctemp[6] = ctot + mole_frac[0] - mole_frac[5] + mole_frac[10] + 
      0.5*mole_frac[11] + mole_frac[12] + 2.0*mole_frac[18]; 
  }
  
  double rr_f[84];
  double rr_r[84];
  //   0)  H + O + M <=> OH + M
  {
    double forward = 5.0e+17 * otc;
    double xik = -cgspl[1] - cgspl[2] + cgspl[4];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[0] = forward * mole_frac[1] * mole_frac[2];
    rr_r[0] = reverse * mole_frac[4];
    rr_f[0] *= thbctemp[0];
    rr_r[0] *= thbctemp[0];
  }
  //   1)  H2 + O <=> H + OH
  {
    double forward = 5.0e+04 * exp(2.67*vlntemp - 6.29e+03*ortc);
    double xik = -cgspl[0] + cgspl[1] - cgspl[2] + cgspl[4];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[1] = forward * mole_frac[0] * mole_frac[2];
    rr_r[1] = reverse * mole_frac[1] * mole_frac[4];
  }
  //   2)  O + HO2 <=> O2 + OH
  {
    double forward = 2.0e+13;
    double xik = -cgspl[2] + cgspl[3] + cgspl[4] - cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[2] = forward * mole_frac[2] * mole_frac[6];
    rr_r[2] = reverse * mole_frac[3] * mole_frac[4];
  }
  //   3)  O + CH2 <=> H + HCO
  {
    double forward = 8.0e+13;
    double xik = cgspl[1] - cgspl[2] - cgspl[7] + cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[3] = forward * mole_frac[2] * mole_frac[7];
    rr_r[3] = reverse * mole_frac[1] * mole_frac[13];
  }
  //   4)  O + CH2(S) <=> H + HCO
  {
    double forward = 1.5e+13;
    double xik = cgspl[1] - cgspl[2] - cgspl[8] + cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[4] = forward * mole_frac[2] * mole_frac[8];
    rr_r[4] = reverse * mole_frac[1] * mole_frac[13];
  }
  //   5)  O + CH3 <=> H + CH2O
  {
    double forward = 8.43e+13;
    double xik = cgspl[1] - cgspl[2] - cgspl[9] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[5] = forward * mole_frac[2] * mole_frac[9];
    rr_r[5] = reverse * mole_frac[1] * mole_frac[14];
  }
  //   6)  O + CH4 <=> OH + CH3
  {
    double forward = 1.02e+09 * exp(1.5*vlntemp - 8.6e+03*ortc);
    double xik = -cgspl[2] + cgspl[4] + cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[6] = forward * mole_frac[2] * mole_frac[10];
    rr_r[6] = reverse * mole_frac[4] * mole_frac[9];
  }
  //   7)  O + CO + M <=> CO2 + M
  {
    double forward = 6.02e+14 * exp(-3.0e+03*ortc);
    double xik = -cgspl[2] - cgspl[11] + cgspl[12];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[7] = forward * mole_frac[2] * mole_frac[11];
    rr_r[7] = reverse * mole_frac[12];
    rr_f[7] *= thbctemp[1];
    rr_r[7] *= thbctemp[1];
  }
  //   8)  O + HCO <=> OH + CO
  {
    double forward = 3.0e+13;
    double xik = -cgspl[2] + cgspl[4] + cgspl[11] - cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[8] = forward * mole_frac[2] * mole_frac[13];
    rr_r[8] = reverse * mole_frac[4] * mole_frac[11];
  }
  //   9)  O + HCO <=> H + CO2
  {
    double forward = 3.0e+13;
    double xik = cgspl[1] - cgspl[2] + cgspl[12] - cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[9] = forward * mole_frac[2] * mole_frac[13];
    rr_r[9] = reverse * mole_frac[1] * mole_frac[12];
  }
  //  10)  O + CH2O <=> OH + HCO
  {
    double forward = 3.9e+13 * exp(-3.54e+03*ortc);
    double xik = -cgspl[2] + cgspl[4] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[10] = forward * mole_frac[2] * mole_frac[14];
    rr_r[10] = reverse * mole_frac[4] * mole_frac[13];
  }
  //  11)  O + C2H4 <=> CH3 + HCO
  {
    double forward = 1.92e+07 * exp(1.83*vlntemp - 220.0*ortc);
    double xik = -cgspl[2] + cgspl[9] + cgspl[13] - cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[11] = forward * mole_frac[2] * mole_frac[16];
    rr_r[11] = reverse * mole_frac[9] * mole_frac[13];
  }
  //  12)  O + C2H5 <=> CH3 + CH2O
  {
    double forward = 1.32e+14;
    double xik = -cgspl[2] + cgspl[9] + cgspl[14] - cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[12] = forward * mole_frac[2] * mole_frac[17];
    rr_r[12] = reverse * mole_frac[9] * mole_frac[14];
  }
  //  13)  O + C2H6 <=> OH + C2H5
  {
    double forward = 8.98e+07 * exp(1.92*vlntemp - 5.69e+03*ortc);
    double xik = -cgspl[2] + cgspl[4] + cgspl[17] - cgspl[18];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[13] = forward * mole_frac[2] * mole_frac[18];
    rr_r[13] = reverse * mole_frac[4] * mole_frac[17];
  }
  //  14)  O2 + CO <=> O + CO2
  {
    double forward = 2.5e+12 * exp(-4.78e+04*ortc);
    double xik = cgspl[2] - cgspl[3] - cgspl[11] + cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[14] = forward * mole_frac[3] * mole_frac[11];
    rr_r[14] = reverse * mole_frac[2] * mole_frac[12];
  }
  //  15)  O2 + CH2O <=> HO2 + HCO
  {
    double forward = 1.0e+14 * exp(-4.0e+04*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[15] = forward * mole_frac[3] * mole_frac[14];
    rr_r[15] = reverse * mole_frac[6] * mole_frac[13];
  }
  //  16)  H + O2 + M <=> HO2 + M
  {
    double forward = 2.8e+18 * exp(-0.86 * vlntemp);
    double xik = -cgspl[1] - cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[16] = forward * mole_frac[1] * mole_frac[3];
    rr_r[16] = reverse * mole_frac[6];
    rr_f[16] *= thbctemp[2];
    rr_r[16] *= thbctemp[2];
  }
  //  17)  H + 2 O2 <=> O2 + HO2
  {
    double forward = 3.0e+20 * exp(-1.72 * vlntemp);
    double xik = -cgspl[1] - cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[17] = forward * mole_frac[1] * mole_frac[3] * mole_frac[3];
    rr_r[17] = reverse * mole_frac[3] * mole_frac[6];
  }
  //  18)  H + O2 + H2O <=> H2O + HO2
  {
    double forward = 9.38e+18 * exp(-0.76 * vlntemp);
    double xik = -cgspl[1] - cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[18] = forward * mole_frac[1] * mole_frac[3] * mole_frac[5];
    rr_r[18] = reverse * mole_frac[5] * mole_frac[6];
  }
  //  19)  H + O2 + N2 <=> HO2 + N2
  {
    double forward = 3.75e+20 * exp(-1.72 * vlntemp);
    double xik = -cgspl[1] - cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[19] = forward * mole_frac[1] * mole_frac[3] * mole_frac[19];
    rr_r[19] = reverse * mole_frac[6] * mole_frac[19];
  }
  //  20)  H + O2 + AR <=> HO2 + AR
  {
    double forward = 7.0e+17 * exp(-0.8 * vlntemp);
    double xik = -cgspl[1] - cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[20] = forward * mole_frac[1] * mole_frac[3] * mole_frac[20];
    rr_r[20] = reverse * mole_frac[6] * mole_frac[20];
  }
  //  21)  H + O2 <=> O + OH
  {
    double forward = 8.3e+13 * exp(-1.4413e+04*ortc);
    double xik = -cgspl[1] + cgspl[2] - cgspl[3] + cgspl[4];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[21] = forward * mole_frac[1] * mole_frac[3];
    rr_r[21] = reverse * mole_frac[2] * mole_frac[4];
  }
  //  22)  2 H + M <=> H2 + M
  {
    double forward = 1.0e+18 * otc;
    double xik = cgspl[0] - 2.0 * cgspl[1];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[22] = forward * mole_frac[1] * mole_frac[1];
    rr_r[22] = reverse * mole_frac[0];
    rr_f[22] *= thbctemp[3];
    rr_r[22] *= thbctemp[3];
  }
  //  23)  H2 + 2 H <=> 2 H2
  {
    double forward = 9.0e+16 * exp(-0.6 * vlntemp);
    double xik = cgspl[0] - 2.0 * cgspl[1];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[23] = forward * mole_frac[0] * mole_frac[1] * mole_frac[1];
    rr_r[23] = reverse * mole_frac[0] * mole_frac[0];
  }
  //  24)  2 H + H2O <=> H2 + H2O
  {
    double forward = 6.0e+19 * exp(-1.25 * vlntemp);
    double xik = cgspl[0] - 2.0 * cgspl[1];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[24] = forward * mole_frac[1] * mole_frac[1] * mole_frac[5];
    rr_r[24] = reverse * mole_frac[0] * mole_frac[5];
  }
  //  25)  2 H + CO2 <=> H2 + CO2
  {
    double forward = 5.5e+20 * otc * otc;
    double xik = cgspl[0] - 2.0 * cgspl[1];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[25] = forward * mole_frac[1] * mole_frac[1] * mole_frac[12];
    rr_r[25] = reverse * mole_frac[0] * mole_frac[12];
  }
  //  26)  H + OH + M <=> H2O + M
  {
    double forward = 2.2e+22 * otc * otc;
    double xik = -cgspl[1] - cgspl[4] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[26] = forward * mole_frac[1] * mole_frac[4];
    rr_r[26] = reverse * mole_frac[5];
    rr_f[26] *= thbctemp[4];
    rr_r[26] *= thbctemp[4];
  }
  //  27)  H + HO2 <=> H2 + O2
  {
    double forward = 2.8e+13 * exp(-1.068e+03*ortc);
    double xik = cgspl[0] - cgspl[1] + cgspl[3] - cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[27] = forward * mole_frac[1] * mole_frac[6];
    rr_r[27] = reverse * mole_frac[0] * mole_frac[3];
  }
  //  28)  H + HO2 <=> 2 OH
  {
    double forward = 1.34e+14 * exp(-635.0*ortc);
    double xik = -cgspl[1] + 2.0 * cgspl[4] - cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[28] = forward * mole_frac[1] * mole_frac[6];
    rr_r[28] = reverse * mole_frac[4] * mole_frac[4];
  }
  //  29)  H + CH2 (+M) <=> CH3 (+M)
  {
    double rr_k0 = 3.2e+27 * exp(-3.14*vlntemp - 1.23e+03*ortc);
    double rr_kinf = 2.5e+16 * exp(-0.8 * vlntemp);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.32 * exp(-0.01282051282051282 * temperature) + 
      0.68 * exp(-5.012531328320802e-04 * temperature) + exp(-5.59e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[7] + cgspl[9];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[29] = forward * mole_frac[1] * mole_frac[7];
    rr_r[29] = reverse * mole_frac[9];
  }
  //  30)  H + CH3 (+M) <=> CH4 (+M)
  {
    double rr_k0 = 2.477e+33 * exp(-4.76*vlntemp - 2.44e+03*ortc);
    double rr_kinf = 1.27e+16 * exp(-0.63*vlntemp - 383.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.217 * exp(-0.01351351351351351 * temperature) + 
      0.783 * exp(-3.400204012240735e-04 * temperature) + exp(-6.964e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[9] + cgspl[10];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[30] = forward * mole_frac[1] * mole_frac[9];
    rr_r[30] = reverse * mole_frac[10];
  }
  //  31)  H + CH4 <=> H2 + CH3
  {
    double forward = 6.6e+08 * exp(1.62*vlntemp - 1.084e+04*ortc);
    double xik = cgspl[0] - cgspl[1] + cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[31] = forward * mole_frac[1] * mole_frac[10];
    rr_r[31] = reverse * mole_frac[0] * mole_frac[9];
  }
  //  32)  H + HCO (+M) <=> CH2O (+M)
  {
    double rr_k0 = 1.35e+24 * exp(-2.57*vlntemp - 1.425e+03*ortc);
    double rr_kinf = 1.09e+12 * exp(0.48*vlntemp + 260.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
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
    rr_f[32] = forward * mole_frac[1] * mole_frac[13];
    rr_r[32] = reverse * mole_frac[14];
  }
  //  33)  H + HCO <=> H2 + CO
  {
    double forward = 7.34e+13;
    double xik = cgspl[0] - cgspl[1] + cgspl[11] - cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[33] = forward * mole_frac[1] * mole_frac[13];
    rr_r[33] = reverse * mole_frac[0] * mole_frac[11];
  }
  //  34)  H + CH2O (+M) <=> CH3O (+M)
  {
    double rr_k0 = 2.2e+30 * exp(-4.8*vlntemp - 5.56e+03*ortc);
    double rr_kinf = 5.4e+11 * exp(0.454*vlntemp - 2.6e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[5];
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
    rr_f[34] = forward * mole_frac[1] * mole_frac[14];
    rr_r[34] = reverse * mole_frac[15];
  }
  //  35)  H + CH2O <=> H2 + HCO
  {
    double forward = 2.3e+10 * exp(1.05*vlntemp - 3.275e+03*ortc);
    double xik = cgspl[0] - cgspl[1] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[35] = forward * mole_frac[1] * mole_frac[14];
    rr_r[35] = reverse * mole_frac[0] * mole_frac[13];
  }
  //  36)  H + CH3O <=> OH + CH3
  {
    double forward = 3.2e+13;
    double xik = -cgspl[1] + cgspl[4] + cgspl[9] - cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[36] = forward * mole_frac[1] * mole_frac[15];
    rr_r[36] = reverse * mole_frac[4] * mole_frac[9];
  }
  //  37)  H + C2H4 (+M) <=> C2H5 (+M)
  {
    double rr_k0 = 1.2e+42 * exp(-7.62*vlntemp - 6.97e+03*ortc);
    double rr_kinf = 1.08e+12 * exp(0.454*vlntemp - 1.82e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.02470000000000006 * exp(-4.761904761904762e-03 * 
      temperature) + 0.9752999999999999 * exp(-1.016260162601626e-03 * 
      temperature) + exp(-4.374e+03 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[16] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[37] = forward * mole_frac[1] * mole_frac[16];
    rr_r[37] = reverse * mole_frac[17];
  }
  //  38)  H + C2H5 (+M) <=> C2H6 (+M)
  {
    double rr_k0 = 1.99e+41 * exp(-7.08*vlntemp - 6.685e+03*ortc);
    double rr_kinf = 5.21e+17 * exp(-0.99*vlntemp - 1.58e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.1578000000000001 * exp(-8.0e-03 * temperature) + 
      0.8421999999999999 * exp(-4.506534474988734e-04 * temperature) + 
      exp(-6.882e+03 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[17] + cgspl[18];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[38] = forward * mole_frac[1] * mole_frac[17];
    rr_r[38] = reverse * mole_frac[18];
  }
  //  39)  H + C2H6 <=> H2 + C2H5
  {
    double forward = 1.15e+08 * exp(1.9*vlntemp - 7.53e+03*ortc);
    double xik = cgspl[0] - cgspl[1] + cgspl[17] - cgspl[18];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[39] = forward * mole_frac[1] * mole_frac[18];
    rr_r[39] = reverse * mole_frac[0] * mole_frac[17];
  }
  //  40)  H2 + CO (+M) <=> CH2O (+M)
  {
    double rr_k0 = 5.07e+27 * exp(-3.42*vlntemp - 8.435e+04*ortc);
    double rr_kinf = 4.3e+07 * exp(1.5*vlntemp - 7.96e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.06799999999999995 * exp(-5.076142131979695e-03 * 
      temperature) + 0.9320000000000001 * exp(-6.493506493506494e-04 * 
      temperature) + exp(-1.03e+04 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] - cgspl[11] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[40] = forward * mole_frac[0] * mole_frac[11];
    rr_r[40] = reverse * mole_frac[14];
  }
  //  41)  H2 + OH <=> H + H2O
  {
    double forward = 2.16e+08 * exp(1.51*vlntemp - 3.43e+03*ortc);
    double xik = -cgspl[0] + cgspl[1] - cgspl[4] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[41] = forward * mole_frac[0] * mole_frac[4];
    rr_r[41] = reverse * mole_frac[1] * mole_frac[5];
  }
  //  42)  2 OH <=> O + H2O
  {
    double forward = 3.57e+04 * exp(2.4*vlntemp + 2.11e+03*ortc);
    double xik = cgspl[2] - 2.0 * cgspl[4] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[42] = forward * mole_frac[4] * mole_frac[4];
    rr_r[42] = reverse * mole_frac[2] * mole_frac[5];
  }
  //  43)  OH + HO2 <=> O2 + H2O
  {
    double forward = 2.9e+13 * exp(500.0*ortc);
    double xik = cgspl[3] - cgspl[4] + cgspl[5] - cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[43] = forward * mole_frac[4] * mole_frac[6];
    rr_r[43] = reverse * mole_frac[3] * mole_frac[5];
  }
  //  44)  OH + CH2 <=> H + CH2O
  {
    double forward = 2.0e+13;
    double xik = cgspl[1] - cgspl[4] - cgspl[7] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[44] = forward * mole_frac[4] * mole_frac[7];
    rr_r[44] = reverse * mole_frac[1] * mole_frac[14];
  }
  //  45)  OH + CH2(S) <=> H + CH2O
  {
    double forward = 3.0e+13;
    double xik = cgspl[1] - cgspl[4] - cgspl[8] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[45] = forward * mole_frac[4] * mole_frac[8];
    rr_r[45] = reverse * mole_frac[1] * mole_frac[14];
  }
  //  46)  OH + CH3 <=> H2O + CH2
  {
    double forward = 5.6e+07 * exp(1.6*vlntemp - 5.42e+03*ortc);
    double xik = -cgspl[4] + cgspl[5] + cgspl[7] - cgspl[9];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[46] = forward * mole_frac[4] * mole_frac[9];
    rr_r[46] = reverse * mole_frac[5] * mole_frac[7];
  }
  //  47)  OH + CH3 <=> H2O + CH2(S)
  {
    double forward = 2.501e+13;
    double xik = -cgspl[4] + cgspl[5] + cgspl[8] - cgspl[9];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[47] = forward * mole_frac[4] * mole_frac[9];
    rr_r[47] = reverse * mole_frac[5] * mole_frac[8];
  }
  //  48)  OH + CH4 <=> H2O + CH3
  {
    double forward = 1.0e+08 * exp(1.6*vlntemp - 3.12e+03*ortc);
    double xik = -cgspl[4] + cgspl[5] + cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[48] = forward * mole_frac[4] * mole_frac[10];
    rr_r[48] = reverse * mole_frac[5] * mole_frac[9];
  }
  //  49)  OH + CO <=> H + CO2
  {
    double forward = 4.76e+07 * exp(1.228*vlntemp - 70.0*ortc);
    double xik = cgspl[1] - cgspl[4] - cgspl[11] + cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[49] = forward * mole_frac[4] * mole_frac[11];
    rr_r[49] = reverse * mole_frac[1] * mole_frac[12];
  }
  //  50)  OH + HCO <=> H2O + CO
  {
    double forward = 5.0e+13;
    double xik = -cgspl[4] + cgspl[5] + cgspl[11] - cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[50] = forward * mole_frac[4] * mole_frac[13];
    rr_r[50] = reverse * mole_frac[5] * mole_frac[11];
  }
  //  51)  OH + CH2O <=> H2O + HCO
  {
    double forward = 3.43e+09 * exp(1.18*vlntemp + 447.0*ortc);
    double xik = -cgspl[4] + cgspl[5] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[51] = forward * mole_frac[4] * mole_frac[14];
    rr_r[51] = reverse * mole_frac[5] * mole_frac[13];
  }
  //  52)  OH + C2H6 <=> H2O + C2H5
  {
    double forward = 3.54e+06 * exp(2.12*vlntemp - 870.0*ortc);
    double xik = -cgspl[4] + cgspl[5] + cgspl[17] - cgspl[18];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[52] = forward * mole_frac[4] * mole_frac[18];
    rr_r[52] = reverse * mole_frac[5] * mole_frac[17];
  }
  //  53)  HO2 + CH2 <=> OH + CH2O
  {
    double forward = 2.0e+13;
    double xik = cgspl[4] - cgspl[6] - cgspl[7] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[53] = forward * mole_frac[6] * mole_frac[7];
    rr_r[53] = reverse * mole_frac[4] * mole_frac[14];
  }
  //  54)  HO2 + CH3 <=> O2 + CH4
  {
    double forward = 1.0e+12;
    double xik = cgspl[3] - cgspl[6] - cgspl[9] + cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[54] = forward * mole_frac[6] * mole_frac[9];
    rr_r[54] = reverse * mole_frac[3] * mole_frac[10];
  }
  //  55)  HO2 + CH3 <=> OH + CH3O
  {
    double forward = 2.0e+13;
    double xik = cgspl[4] - cgspl[6] - cgspl[9] + cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[55] = forward * mole_frac[6] * mole_frac[9];
    rr_r[55] = reverse * mole_frac[4] * mole_frac[15];
  }
  //  56)  HO2 + CO <=> OH + CO2
  {
    double forward = 1.5e+14 * exp(-2.36e+04*ortc);
    double xik = cgspl[4] - cgspl[6] - cgspl[11] + cgspl[12];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[56] = forward * mole_frac[6] * mole_frac[11];
    rr_r[56] = reverse * mole_frac[4] * mole_frac[12];
  }
  //  57)  O2 + CH2 <=> OH + HCO
  {
    double forward = 1.32e+13 * exp(-1.5e+03*ortc);
    double xik = -cgspl[3] + cgspl[4] - cgspl[7] + cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[57] = forward * mole_frac[3] * mole_frac[7];
    rr_r[57] = reverse * mole_frac[4] * mole_frac[13];
  }
  //  58)  H2 + CH2 <=> H + CH3
  {
    double forward = 5.0e+05 * temperature * temperature * exp(-7.23e+03*ortc);
    double xik = -cgspl[0] + cgspl[1] - cgspl[7] + cgspl[9];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[58] = forward * mole_frac[0] * mole_frac[7];
    rr_r[58] = reverse * mole_frac[1] * mole_frac[9];
  }
  //  59)  CH2 + CH3 <=> H + C2H4
  {
    double forward = 4.0e+13;
    double xik = cgspl[1] - cgspl[7] - cgspl[9] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[59] = forward * mole_frac[7] * mole_frac[9];
    rr_r[59] = reverse * mole_frac[1] * mole_frac[16];
  }
  //  60)  CH2 + CH4 <=> 2 CH3
  {
    double forward = 2.46e+06 * temperature * temperature * exp(-8.27e+03*ortc);
    double xik = -cgspl[7] + 2.0 * cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[60] = forward * mole_frac[7] * mole_frac[10];
    rr_r[60] = reverse * mole_frac[9] * mole_frac[9];
  }
  //  61)  CH2(S) + N2 <=> CH2 + N2
  {
    double forward = 1.5e+13 * exp(-600.0*ortc);
    double xik = cgspl[7] - cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[61] = forward * mole_frac[8] * mole_frac[19];
    rr_r[61] = reverse * mole_frac[7] * mole_frac[19];
  }
  //  62)  CH2(S) + AR <=> CH2 + AR
  {
    double forward = 9.0e+12 * exp(-600.0*ortc);
    double xik = cgspl[7] - cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[62] = forward * mole_frac[8] * mole_frac[20];
    rr_r[62] = reverse * mole_frac[7] * mole_frac[20];
  }
  //  63)  O2 + CH2(S) <=> H + OH + CO
  {
    double forward = 2.8e+13;
    double xik = cgspl[1] - cgspl[3] + cgspl[4] - cgspl[8] + cgspl[11];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[63] = forward * mole_frac[3] * mole_frac[8];
    rr_r[63] = reverse * mole_frac[1] * mole_frac[4] * mole_frac[11];
  }
  //  64)  O2 + CH2(S) <=> H2O + CO
  {
    double forward = 1.2e+13;
    double xik = -cgspl[3] + cgspl[5] - cgspl[8] + cgspl[11];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[64] = forward * mole_frac[3] * mole_frac[8];
    rr_r[64] = reverse * mole_frac[5] * mole_frac[11];
  }
  //  65)  H2 + CH2(S) <=> H + CH3
  {
    double forward = 7.0e+13;
    double xik = -cgspl[0] + cgspl[1] - cgspl[8] + cgspl[9];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[65] = forward * mole_frac[0] * mole_frac[8];
    rr_r[65] = reverse * mole_frac[1] * mole_frac[9];
  }
  //  66)  H2O + CH2(S) <=> H2O + CH2
  {
    double forward = 3.0e+13;
    double xik = cgspl[7] - cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[66] = forward * mole_frac[5] * mole_frac[8];
    rr_r[66] = reverse * mole_frac[5] * mole_frac[7];
  }
  //  67)  CH2(S) + CH3 <=> H + C2H4
  {
    double forward = 1.2e+13 * exp(570.0*ortc);
    double xik = cgspl[1] - cgspl[8] - cgspl[9] + cgspl[16];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[67] = forward * mole_frac[8] * mole_frac[9];
    rr_r[67] = reverse * mole_frac[1] * mole_frac[16];
  }
  //  68)  CH2(S) + CH4 <=> 2 CH3
  {
    double forward = 1.6e+13 * exp(570.0*ortc);
    double xik = -cgspl[8] + 2.0 * cgspl[9] - cgspl[10];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[68] = forward * mole_frac[8] * mole_frac[10];
    rr_r[68] = reverse * mole_frac[9] * mole_frac[9];
  }
  //  69)  CH2(S) + CO <=> CH2 + CO
  {
    double forward = 9.0e+12;
    double xik = cgspl[7] - cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[69] = forward * mole_frac[8] * mole_frac[11];
    rr_r[69] = reverse * mole_frac[7] * mole_frac[11];
  }
  //  70)  CH2(S) + CO2 <=> CH2 + CO2
  {
    double forward = 7.0e+12;
    double xik = cgspl[7] - cgspl[8];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[70] = forward * mole_frac[8] * mole_frac[12];
    rr_r[70] = reverse * mole_frac[7] * mole_frac[12];
  }
  //  71)  CH2(S) + CO2 <=> CO + CH2O
  {
    double forward = 1.4e+13;
    double xik = -cgspl[8] + cgspl[11] - cgspl[12] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[71] = forward * mole_frac[8] * mole_frac[12];
    rr_r[71] = reverse * mole_frac[11] * mole_frac[14];
  }
  //  72)  O2 + CH3 <=> O + CH3O
  {
    double forward = 2.675e+13 * exp(-2.88e+04*ortc);
    double xik = cgspl[2] - cgspl[3] - cgspl[9] + cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[72] = forward * mole_frac[3] * mole_frac[9];
    rr_r[72] = reverse * mole_frac[2] * mole_frac[15];
  }
  //  73)  O2 + CH3 <=> OH + CH2O
  {
    double forward = 3.6e+10 * exp(-8.94e+03*ortc);
    double xik = -cgspl[3] + cgspl[4] - cgspl[9] + cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[73] = forward * mole_frac[3] * mole_frac[9];
    rr_r[73] = reverse * mole_frac[4] * mole_frac[14];
  }
  //  74)  2 CH3 (+M) <=> C2H6 (+M)
  {
    double rr_k0 = 1.77e+50 * exp(-9.67*vlntemp - 6.22e+03*ortc);
    double rr_kinf = 2.12e+16 * exp(-0.97*vlntemp - 620.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.4675 * exp(-6.622516556291391e-03 * temperature) 
      + 0.5325 * exp(-9.633911368015414e-04 * temperature) + exp(-4.97e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -2.0 * cgspl[9] + cgspl[18];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[74] = forward * mole_frac[9] * mole_frac[9];
    rr_r[74] = reverse * mole_frac[18];
  }
  //  75)  2 CH3 <=> H + C2H5
  {
    double forward = 4.99e+12 * exp(0.1*vlntemp - 1.06e+04*ortc);
    double xik = cgspl[1] - 2.0 * cgspl[9] + cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[75] = forward * mole_frac[9] * mole_frac[9];
    rr_r[75] = reverse * mole_frac[1] * mole_frac[17];
  }
  //  76)  CH3 + HCO <=> CH4 + CO
  {
    double forward = 2.648e+13;
    double xik = -cgspl[9] + cgspl[10] + cgspl[11] - cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[76] = forward * mole_frac[9] * mole_frac[13];
    rr_r[76] = reverse * mole_frac[10] * mole_frac[11];
  }
  //  77)  CH3 + CH2O <=> CH4 + HCO
  {
    double forward = 3.32e+03 * exp(2.81*vlntemp - 5.86e+03*ortc);
    double xik = -cgspl[9] + cgspl[10] + cgspl[13] - cgspl[14];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[77] = forward * mole_frac[9] * mole_frac[14];
    rr_r[77] = reverse * mole_frac[10] * mole_frac[13];
  }
  //  78)  CH3 + C2H6 <=> CH4 + C2H5
  {
    double forward = 6.14e+06 * exp(1.74*vlntemp - 1.045e+04*ortc);
    double xik = -cgspl[9] + cgspl[10] + cgspl[17] - cgspl[18];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[78] = forward * mole_frac[9] * mole_frac[18];
    rr_r[78] = reverse * mole_frac[10] * mole_frac[17];
  }
  //  79)  H2O + HCO <=> H + H2O + CO
  {
    double forward = 2.244e+18 * otc * exp(-1.7e+04*ortc);
    double xik = cgspl[1] + cgspl[11] - cgspl[13];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[79] = forward * mole_frac[5] * mole_frac[13];
    rr_r[79] = reverse * mole_frac[1] * mole_frac[5] * mole_frac[11];
  }
  //  80)  HCO + M <=> H + CO + M
  {
    double forward = 1.87e+17 * otc * exp(-1.7e+04*ortc);
    double xik = cgspl[1] + cgspl[11] - cgspl[13];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[80] = forward * mole_frac[13];
    rr_r[80] = reverse * mole_frac[1] * mole_frac[11];
    rr_f[80] *= thbctemp[6];
    rr_r[80] *= thbctemp[6];
  }
  //  81)  O2 + HCO <=> HO2 + CO
  {
    double forward = 7.6e+12 * exp(-400.0*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[11] - cgspl[13];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[81] = forward * mole_frac[3] * mole_frac[13];
    rr_r[81] = reverse * mole_frac[6] * mole_frac[11];
  }
  //  82)  O2 + CH3O <=> HO2 + CH2O
  {
    double forward = 4.28e-13 * exp(7.6*vlntemp + 3.53e+03*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[14] - cgspl[15];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[82] = forward * mole_frac[3] * mole_frac[15];
    rr_r[82] = reverse * mole_frac[6] * mole_frac[14];
  }
  //  83)  O2 + C2H5 <=> HO2 + C2H4
  {
    double forward = 8.4e+11 * exp(-3.875e+03*ortc);
    double xik = -cgspl[3] + cgspl[6] + cgspl[16] - cgspl[17];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[83] = forward * mole_frac[3] * mole_frac[17];
    rr_r[83] = reverse * mole_frac[6] * mole_frac[16];
  }
  double ropl[84];
  for (int i = 0; i < 84; i++)
  {
    ropl[i] = rr_f[i] - rr_r[i];
  }
  // 0. H2
  wdot[0] = -ropl[1] + ropl[22] + ropl[23] + ropl[24] + ropl[25] + ropl[27] + 
    ropl[31] + ropl[33] + ropl[35] + ropl[39] - ropl[40] - ropl[41] - ropl[58] - 
    ropl[65]; 
  // 1. H
  wdot[1] = -ropl[0] + ropl[1] + ropl[3] + ropl[4] + ropl[5] + ropl[9] - 
    ropl[16] - ropl[17] - ropl[18] - ropl[19] - ropl[20] - ropl[21] - 
    2.0*ropl[22] - 2.0*ropl[23] - 2.0*ropl[24] - 2.0*ropl[25] - ropl[26] - 
    ropl[27] - ropl[28] - ropl[29] - ropl[30] - ropl[31] - ropl[32] - ropl[33] - 
    ropl[34] - ropl[35] - ropl[36] - ropl[37] - ropl[38] - ropl[39] + ropl[41] + 
    ropl[44] + ropl[45] + ropl[49] + ropl[58] + ropl[59] + ropl[63] + ropl[65] + 
    ropl[67] + ropl[75] + ropl[79] + ropl[80]; 
  // 2. O
  wdot[2] = -ropl[0] - ropl[1] - ropl[2] - ropl[3] - ropl[4] - ropl[5] - ropl[6] 
    - ropl[7] - ropl[8] - ropl[9] - ropl[10] - ropl[11] - ropl[12] - ropl[13] + 
    ropl[14] + ropl[21] + ropl[42] + ropl[72]; 
  // 3. O2
  wdot[3] = ropl[2] - ropl[14] - ropl[15] - ropl[16] - ropl[17] - ropl[18] - 
    ropl[19] - ropl[20] - ropl[21] + ropl[27] + ropl[43] + ropl[54] - ropl[57] - 
    ropl[63] - ropl[64] - ropl[72] - ropl[73] - ropl[81] - ropl[82] - ropl[83]; 
  // 4. OH
  wdot[4] = ropl[0] + ropl[1] + ropl[2] + ropl[6] + ropl[8] + ropl[10] + 
    ropl[13] + ropl[21] - ropl[26] + 2.0*ropl[28] + ropl[36] - ropl[41] - 
    2.0*ropl[42] - ropl[43] - ropl[44] - ropl[45] - ropl[46] - ropl[47] - 
    ropl[48] - ropl[49] - ropl[50] - ropl[51] - ropl[52] + ropl[53] + ropl[55] + 
    ropl[56] + ropl[57] + ropl[63] + ropl[73]; 
  // 5. H2O
  wdot[5] = ropl[26] + ropl[41] + ropl[42] + ropl[43] + ropl[46] + ropl[47] + 
    ropl[48] + ropl[50] + ropl[51] + ropl[52] + ropl[64]; 
  // 6. HO2
  wdot[6] = -ropl[2] + ropl[15] + ropl[16] + ropl[17] + ropl[18] + ropl[19] + 
    ropl[20] - ropl[27] - ropl[28] - ropl[43] - ropl[53] - ropl[54] - ropl[55] - 
    ropl[56] + ropl[81] + ropl[82] + ropl[83]; 
  // 7. CH2
  wdot[7] = -ropl[3] - ropl[29] - ropl[44] + ropl[46] - ropl[53] - ropl[57] - 
    ropl[58] - ropl[59] - ropl[60] + ropl[61] + ropl[62] + ropl[66] + ropl[69] + 
    ropl[70]; 
  // 8. CH2(S)
  wdot[8] = -ropl[4] - ropl[45] + ropl[47] - ropl[61] - ropl[62] - ropl[63] - 
    ropl[64] - ropl[65] - ropl[66] - ropl[67] - ropl[68] - ropl[69] - ropl[70] - 
    ropl[71]; 
  // 9. CH3
  wdot[9] = -ropl[5] + ropl[6] + ropl[11] + ropl[12] + ropl[29] - ropl[30] + 
    ropl[31] + ropl[36] - ropl[46] - ropl[47] + ropl[48] - ropl[54] - ropl[55] + 
    ropl[58] - ropl[59] + 2.0*ropl[60] + ropl[65] - ropl[67] + 2.0*ropl[68] - 
    ropl[72] - ropl[73] - 2.0*ropl[74] - 2.0*ropl[75] - ropl[76] - ropl[77] - 
    ropl[78]; 
  // 10. CH4
  wdot[10] = -ropl[6] + ropl[30] - ropl[31] - ropl[48] + ropl[54] - ropl[60] - 
    ropl[68] + ropl[76] + ropl[77] + ropl[78]; 
  // 11. CO
  wdot[11] = -ropl[7] + ropl[8] - ropl[14] + ropl[33] - ropl[40] - ropl[49] + 
    ropl[50] - ropl[56] + ropl[63] + ropl[64] + ropl[71] + ropl[76] + ropl[79] + 
    ropl[80] + ropl[81]; 
  // 12. CO2
  wdot[12] = ropl[7] + ropl[9] + ropl[14] + ropl[49] + ropl[56] - ropl[71];
  // 13. HCO
  wdot[13] = ropl[3] + ropl[4] - ropl[8] - ropl[9] + ropl[10] + ropl[11] + 
    ropl[15] - ropl[32] - ropl[33] + ropl[35] - ropl[50] + ropl[51] + ropl[57] - 
    ropl[76] + ropl[77] - ropl[79] - ropl[80] - ropl[81]; 
  // 14. CH2O
  wdot[14] = ropl[5] - ropl[10] + ropl[12] - ropl[15] + ropl[32] - ropl[34] - 
    ropl[35] + ropl[40] + ropl[44] + ropl[45] - ropl[51] + ropl[53] + ropl[71] + 
    ropl[73] - ropl[77] + ropl[82]; 
  // 15. CH3O
  wdot[15] = ropl[34] - ropl[36] + ropl[55] + ropl[72] - ropl[82];
  // 16. C2H4
  wdot[16] = -ropl[11] - ropl[37] + ropl[59] + ropl[67] + ropl[83];
  // 17. C2H5
  wdot[17] = -ropl[12] + ropl[13] + ropl[37] - ropl[38] + ropl[39] + ropl[52] + 
    ropl[75] + ropl[78] - ropl[83]; 
  // 18. C2H6
  wdot[18] = -ropl[13] + ropl[38] - ropl[39] - ropl[52] + ropl[74] - ropl[78];
  // 19. N2
  wdot[19] = 0.0;
  // 20. AR
  wdot[20] = 0.0;
}

