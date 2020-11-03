#include "gpu_getrates.h"

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

static AMREX_GPU_DEVICE_MANAGED double molecular_masses[143] = {1.00797, 2.01594, 15.9994, 31.9988, 
  17.00737, 18.01534, 33.00677, 34.01474, 28.01055, 44.00995, 30.02649, 
  29.01852, 124.05058, 122.03464, 90.03584000000001, 31.03446, 31.03446, 
  48.04183, 47.03386, 16.04303, 15.03506, 14.02709, 14.02709, 30.07012, 
  29.06215, 28.05418, 27.04621, 26.03824, 44.05358, 43.04561, 42.03764, 
  41.02967, 45.06155, 61.06095000000001, 86.09122000000001, 58.08067, 57.0727, 
  112.12946, 110.11352, 57.0727, 43.08924, 42.08127, 41.0733, 41.0733, 41.0733, 
  40.06533, 40.06533, 39.05736, 38.04939, 57.0727, 75.08804000000001, 107.08684, 
  75.08804000000001, 90.07947, 56.06473, 57.11633, 57.11633, 57.11633, 56.10836, 
  55.10039, 89.11513000000001, 89.11513000000001, 89.11513000000001, 73.11573, 
  90.12310000000001, 142.19958, 72.10776, 71.09979, 121.11393, 104.10656, 
  144.21552, 142.19958, 140.18364, 138.1677, 87.09919000000001, 70.09182, 
  176.21432, 206.19718, 114.1454, 71.14342000000001, 70.13545000000001, 
  70.13545000000001, 70.13545000000001, 69.12748000000001, 69.12748000000001, 
  85.12688, 85.12688, 57.0727, 84.16254000000001, 84.16254000000001, 140.2709, 
  55.10039, 99.19760000000001, 98.18963000000001, 98.18963000000001, 
  97.18166000000001, 97.18166000000001, 113.18106, 131.1964, 82.14660000000001, 
  81.13863000000001, 81.13863000000001, 71.14342000000001, 103.14222, 103.14222, 
  86.13485, 85.12688, 114.23266, 113.22469, 113.22469, 113.22469, 113.22469, 
  112.21672, 112.21672, 145.22349, 145.22349, 145.22349, 145.22349, 146.23146, 
  129.22409, 145.22349, 145.22349, 145.22349, 145.22349, 145.22349, 145.22349, 
  145.22349, 145.22349, 128.21612, 128.21612, 128.21612, 177.22229, 177.22229, 
  177.22229, 177.22229, 177.22229, 160.21492, 160.21492, 160.21492, 160.21492, 
  113.18106, 113.18106, 28.0134}; 


static AMREX_GPU_DEVICE_MANAGED double recip_molecular_masses[143] = {0.9920930186414277, 
  0.4960465093207139, 0.06250234383789392, 0.03125117191894696, 
  0.05879803873262004, 0.05550825019122593, 0.03029681486555637, 
  0.02939901936631002, 0.03570083414998991, 0.02272213442641948, 
  0.03330392596670473, 0.03446075127194632, 8.061227928156403e-03, 
  8.194394640734796e-03, 0.01110668818106212, 0.03222224585186918, 
  0.03222224585186918, 0.02081519375927187, 0.021261278576753, 
  0.06233236489615739, 0.06651120780362699, 0.07129062407099405, 
  0.07129062407099405, 0.03325560390181349, 0.0344090165386938, 
  0.03564531203549703, 0.0369737571363973, 0.0384050534905585, 
  0.02269963076780593, 0.02323117270262868, 0.02378820504671527, 
  0.02437260645771706, 0.02219186867739791, 0.01637707896781822, 
  0.01161558635131434, 0.01721743223692151, 0.01752151203640269, 
  8.918262872219307e-03, 9.081536944782075e-03, 0.01752151203640269, 
  0.02320764998407955, 0.02376354135699802, 0.02434671672351625, 
  0.02434671672351625, 0.02434671672351625, 0.02495923532889907, 
  0.02495923532889907, 0.02560336899370566, 0.02628163027055098, 
  0.01752151203640269, 0.01331770012907515, 9.338215601468865e-03, 
  0.01331770012907515, 0.01110130865556824, 0.01783652574443861, 
  0.01750812771058645, 0.01750812771058645, 0.01750812771058645, 
  0.01782265601774851, 0.01814869186951308, 0.01122143905305418, 
  0.01122143905305418, 0.01122143905305418, 0.01367694749132642, 
  0.01109593433869896, 7.032369575212529e-03, 0.01386813291662368, 
  0.01406473915042506, 8.256688557625038e-03, 9.605542628629742e-03, 
  6.934066458311838e-03, 7.032369575212529e-03, 7.13350002896201e-03, 
  7.237581576591346e-03, 0.01148116302803734, 0.01426700005792402, 
  5.674907691951482e-03, 4.849726848834693e-03, 8.760756018201346e-03, 
  0.01405611369259448, 0.01425812481419881, 0.01425812481419881, 
  0.01425812481419881, 0.01446602711396394, 0.01446602711396394, 
  0.01174717081138179, 0.01174717081138179, 0.01752151203640269, 
  0.01188177067849901, 0.01188177067849901, 7.129062407099405e-03, 
  0.01814869186951308, 0.01008088905376743, 0.01018437486728486, 
  0.01018437486728486, 0.01029000739439931, 0.01029000739439931, 
  8.835400552000485e-03, 7.622160364156333e-03, 0.01217335836175812, 
  0.01232458571213243, 0.01232458571213243, 0.01405611369259448, 
  9.695350749673605e-03, 9.695350749673605e-03, 0.01160970269292859, 
  0.01174717081138179, 8.754063855293223e-03, 8.831995918911324e-03, 
  8.831995918911324e-03, 8.831995918911324e-03, 8.831995918911324e-03, 
  8.911328008874257e-03, 8.911328008874257e-03, 6.885938356115805e-03, 
  6.885938356115805e-03, 6.885938356115805e-03, 6.885938356115805e-03, 
  6.838473745663211e-03, 7.738495198534576e-03, 6.885938356115805e-03, 
  6.885938356115805e-03, 6.885938356115805e-03, 6.885938356115805e-03, 
  6.885938356115805e-03, 6.885938356115805e-03, 6.885938356115805e-03, 
  6.885938356115805e-03, 7.799331316530245e-03, 7.799331316530245e-03, 
  7.799331316530245e-03, 5.642631070843289e-03, 5.642631070843289e-03, 
  5.642631070843289e-03, 5.642631070843289e-03, 5.642631070843289e-03, 
  6.241615949376e-03, 6.241615949376e-03, 6.241615949376e-03, 
  6.241615949376e-03, 8.835400552000485e-03, 8.835400552000485e-03, 
  0.03569720205330306}; 


AMREX_GPU_GLOBAL 
void gpu_getrates(const double *temperature_array, const double *pressure_array, 
  const double *avmolwt_array, const double *mass_frac_array, const int 
  spec_stride/*NX*NY*NZ in number of doubles*/, double *wdot_array) 
{
  
  const double PA = 1.013250e+06;
  const double R0 = 8.314510e+07;
  const double R0c = 1.9872155832;
  const double DLn10 = 2.3025850929940459e0;
  
  {
    const int offset = (blockIdx.x*blockDim.x + threadIdx.x);
    temperature_array += offset;
    pressure_array += offset;
    avmolwt_array += offset;
    mass_frac_array += offset;
    wdot_array += offset;
  }
  double temperature;
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(temperature) : 
    "l"(temperature_array) : "memory"); 
  const double otc     = 1.0 / temperature;
  const double ortc    = 1.0 / (temperature * R0c);
  const double vlntemp = log(temperature);
  const double prt     = PA / (R0 * temperature);
  const double oprt    = 1.0 / prt;
  
  double mass_frac[143];
  double avmolwt;
  double pressure;
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[0]) : 
    "l"(mass_frac_array+0*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[1]) : 
    "l"(mass_frac_array+1*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[2]) : 
    "l"(mass_frac_array+2*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[3]) : 
    "l"(mass_frac_array+3*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[4]) : 
    "l"(mass_frac_array+4*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[5]) : 
    "l"(mass_frac_array+5*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[6]) : 
    "l"(mass_frac_array+6*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[7]) : 
    "l"(mass_frac_array+7*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[8]) : 
    "l"(mass_frac_array+8*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[9]) : 
    "l"(mass_frac_array+9*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[10]) : 
    "l"(mass_frac_array+10*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[11]) : 
    "l"(mass_frac_array+11*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[12]) : 
    "l"(mass_frac_array+12*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[13]) : 
    "l"(mass_frac_array+13*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[14]) : 
    "l"(mass_frac_array+14*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[15]) : 
    "l"(mass_frac_array+15*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[16]) : 
    "l"(mass_frac_array+16*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[17]) : 
    "l"(mass_frac_array+17*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[18]) : 
    "l"(mass_frac_array+18*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[19]) : 
    "l"(mass_frac_array+19*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[20]) : 
    "l"(mass_frac_array+20*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[21]) : 
    "l"(mass_frac_array+21*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[22]) : 
    "l"(mass_frac_array+22*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[23]) : 
    "l"(mass_frac_array+23*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[24]) : 
    "l"(mass_frac_array+24*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[25]) : 
    "l"(mass_frac_array+25*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[26]) : 
    "l"(mass_frac_array+26*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[27]) : 
    "l"(mass_frac_array+27*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[28]) : 
    "l"(mass_frac_array+28*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[29]) : 
    "l"(mass_frac_array+29*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[30]) : 
    "l"(mass_frac_array+30*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[31]) : 
    "l"(mass_frac_array+31*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[32]) : 
    "l"(mass_frac_array+32*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[33]) : 
    "l"(mass_frac_array+33*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[34]) : 
    "l"(mass_frac_array+34*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[35]) : 
    "l"(mass_frac_array+35*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[36]) : 
    "l"(mass_frac_array+36*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[37]) : 
    "l"(mass_frac_array+37*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[38]) : 
    "l"(mass_frac_array+38*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[39]) : 
    "l"(mass_frac_array+39*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[40]) : 
    "l"(mass_frac_array+40*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[41]) : 
    "l"(mass_frac_array+41*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[42]) : 
    "l"(mass_frac_array+42*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[43]) : 
    "l"(mass_frac_array+43*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[44]) : 
    "l"(mass_frac_array+44*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[45]) : 
    "l"(mass_frac_array+45*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[46]) : 
    "l"(mass_frac_array+46*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[47]) : 
    "l"(mass_frac_array+47*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[48]) : 
    "l"(mass_frac_array+48*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[49]) : 
    "l"(mass_frac_array+49*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[50]) : 
    "l"(mass_frac_array+50*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[51]) : 
    "l"(mass_frac_array+51*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[52]) : 
    "l"(mass_frac_array+52*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[53]) : 
    "l"(mass_frac_array+53*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[54]) : 
    "l"(mass_frac_array+54*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[55]) : 
    "l"(mass_frac_array+55*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[56]) : 
    "l"(mass_frac_array+56*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[57]) : 
    "l"(mass_frac_array+57*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[58]) : 
    "l"(mass_frac_array+58*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[59]) : 
    "l"(mass_frac_array+59*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[60]) : 
    "l"(mass_frac_array+60*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[61]) : 
    "l"(mass_frac_array+61*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[62]) : 
    "l"(mass_frac_array+62*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[63]) : 
    "l"(mass_frac_array+63*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[64]) : 
    "l"(mass_frac_array+64*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[65]) : 
    "l"(mass_frac_array+65*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[66]) : 
    "l"(mass_frac_array+66*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[67]) : 
    "l"(mass_frac_array+67*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[68]) : 
    "l"(mass_frac_array+68*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[69]) : 
    "l"(mass_frac_array+69*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[70]) : 
    "l"(mass_frac_array+70*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[71]) : 
    "l"(mass_frac_array+71*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[72]) : 
    "l"(mass_frac_array+72*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[73]) : 
    "l"(mass_frac_array+73*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[74]) : 
    "l"(mass_frac_array+74*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[75]) : 
    "l"(mass_frac_array+75*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[76]) : 
    "l"(mass_frac_array+76*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[77]) : 
    "l"(mass_frac_array+77*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[78]) : 
    "l"(mass_frac_array+78*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[79]) : 
    "l"(mass_frac_array+79*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[80]) : 
    "l"(mass_frac_array+80*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[81]) : 
    "l"(mass_frac_array+81*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[82]) : 
    "l"(mass_frac_array+82*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[83]) : 
    "l"(mass_frac_array+83*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[84]) : 
    "l"(mass_frac_array+84*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[85]) : 
    "l"(mass_frac_array+85*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[86]) : 
    "l"(mass_frac_array+86*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[87]) : 
    "l"(mass_frac_array+87*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[88]) : 
    "l"(mass_frac_array+88*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[89]) : 
    "l"(mass_frac_array+89*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[90]) : 
    "l"(mass_frac_array+90*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[91]) : 
    "l"(mass_frac_array+91*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[92]) : 
    "l"(mass_frac_array+92*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[93]) : 
    "l"(mass_frac_array+93*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[94]) : 
    "l"(mass_frac_array+94*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[95]) : 
    "l"(mass_frac_array+95*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[96]) : 
    "l"(mass_frac_array+96*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[97]) : 
    "l"(mass_frac_array+97*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[98]) : 
    "l"(mass_frac_array+98*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[99]) : 
    "l"(mass_frac_array+99*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[100]) : 
    "l"(mass_frac_array+100*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[101]) : 
    "l"(mass_frac_array+101*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[102]) : 
    "l"(mass_frac_array+102*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[103]) : 
    "l"(mass_frac_array+103*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[104]) : 
    "l"(mass_frac_array+104*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[105]) : 
    "l"(mass_frac_array+105*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[106]) : 
    "l"(mass_frac_array+106*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[107]) : 
    "l"(mass_frac_array+107*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[108]) : 
    "l"(mass_frac_array+108*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[109]) : 
    "l"(mass_frac_array+109*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[110]) : 
    "l"(mass_frac_array+110*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[111]) : 
    "l"(mass_frac_array+111*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[112]) : 
    "l"(mass_frac_array+112*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[113]) : 
    "l"(mass_frac_array+113*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[114]) : 
    "l"(mass_frac_array+114*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[115]) : 
    "l"(mass_frac_array+115*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[116]) : 
    "l"(mass_frac_array+116*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[117]) : 
    "l"(mass_frac_array+117*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[118]) : 
    "l"(mass_frac_array+118*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[119]) : 
    "l"(mass_frac_array+119*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[120]) : 
    "l"(mass_frac_array+120*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[121]) : 
    "l"(mass_frac_array+121*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[122]) : 
    "l"(mass_frac_array+122*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[123]) : 
    "l"(mass_frac_array+123*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[124]) : 
    "l"(mass_frac_array+124*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[125]) : 
    "l"(mass_frac_array+125*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[126]) : 
    "l"(mass_frac_array+126*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[127]) : 
    "l"(mass_frac_array+127*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[128]) : 
    "l"(mass_frac_array+128*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[129]) : 
    "l"(mass_frac_array+129*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[130]) : 
    "l"(mass_frac_array+130*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[131]) : 
    "l"(mass_frac_array+131*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[132]) : 
    "l"(mass_frac_array+132*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[133]) : 
    "l"(mass_frac_array+133*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[134]) : 
    "l"(mass_frac_array+134*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[135]) : 
    "l"(mass_frac_array+135*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[136]) : 
    "l"(mass_frac_array+136*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[137]) : 
    "l"(mass_frac_array+137*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[138]) : 
    "l"(mass_frac_array+138*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[139]) : 
    "l"(mass_frac_array+139*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[140]) : 
    "l"(mass_frac_array+140*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[141]) : 
    "l"(mass_frac_array+141*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[142]) : 
    "l"(mass_frac_array+142*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(avmolwt) : "l"(avmolwt_array) 
    : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(pressure) : 
    "l"(pressure_array) : "memory"); 
  double cgspl[143];
  // Gibbs computation
  {
    const double &tk1 = temperature;
    double tklog = log(tk1);
    double tk2 = tk1 * tk1;
    double tk3 = tk1 * tk2;
    double tk4 = tk1 * tk3;
    double tk5 = tk1 * tk4;
    
    // Species H
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[0] = 2.5*tk1*(1-tklog) + -0.0*tk2 + -0.0*tk3 + -0.0*tk4 + -0.0*tk5 
          + (2.547163e+04 - tk1*-0.4601176); 
      }
      else
      {
        cgspl[0] = 2.5*tk1*(1-tklog) + -0.0*tk2 + -0.0*tk3 + -0.0*tk4 + -0.0*tk5 
          + (2.547163e+04 - tk1*-0.4601176); 
      }
    }
    // Species H2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[1] = 2.991423*tk1*(1-tklog) + -3.500322e-04*tk2 + 9.389715e-09*tk3 
          + 7.692981666666665e-13*tk4 + -7.913759999999998e-17*tk5 + (-835.034 - 
          tk1*-1.35511); 
      }
      else
      {
        cgspl[1] = 3.298124*tk1*(1-tklog) + -4.124721e-04*tk2 + 
          1.357169166666667e-07*tk3 + 7.896194999999997e-12*tk4 + 
          -2.067436e-14*tk5 + (-1.012521e+03 - tk1*-3.294094); 
      }
    }
    // Species O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[2] = 2.54206*tk1*(1-tklog) + 1.377531e-05*tk2 + 
          5.171338333333334e-10*tk3 + -3.792555833333332e-13*tk4 + 
          2.184026e-17*tk5 + (2.92308e+04 - tk1*4.920308); 
      }
      else
      {
        cgspl[2] = 2.946429*tk1*(1-tklog) + 8.19083e-04*tk2 + 
          -4.035053333333334e-07*tk3 + 1.3357025e-10*tk4 + -1.945348e-14*tk5 + 
          (2.914764e+04 - tk1*2.963995); 
      }
    }
    // Species O2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[3] = 3.697578*tk1*(1-tklog) + -3.0675985e-04*tk2 + 2.09807e-08*tk3 
          + -1.479400833333333e-12*tk4 + 5.682175e-17*tk5 + (-1.23393e+03 - 
          tk1*3.189166); 
      }
      else
      {
        cgspl[3] = 3.212936*tk1*(1-tklog) + -5.63743e-04*tk2 + 
          9.593583333333335e-08*tk3 + -1.0948975e-10*tk4 + 
          4.384276999999999e-14*tk5 + (-1.005249e+03 - tk1*6.034738); 
      }
    }
    // Species OH
    {
      if (tk1 > 1.71e+03)
      {
        cgspl[4] = 2.8537604*tk1*(1-tklog) + -5.1497167e-04*tk2 + 
          3.877774616666667e-08*tk3 + -1.6145892e-12*tk4 + 
          1.578799234999999e-17*tk5 + (3.6994972e+03 - tk1*5.78756825); 
      }
      else
      {
        cgspl[4] = 3.41896226*tk1*(1-tklog) + -1.596279005e-04*tk2 + 
          5.138211950000001e-08*tk3 + -3.036729116666666e-11*tk4 + 
          5.009773949999999e-15*tk5 + (3.45264448e+03 - tk1*2.54433372); 
      }
    }
    // Species H2O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[5] = 2.672146*tk1*(1-tklog) + -1.5281465e-03*tk2 + 
          1.455043333333333e-07*tk3 + -1.00083e-11*tk4 + 
          3.195808999999999e-16*tk5 + (-2.989921e+04 - tk1*6.862817); 
      }
      else
      {
        cgspl[5] = 3.386842*tk1*(1-tklog) + -1.737491e-03*tk2 + 1.059116e-06*tk3 
          + -5.807150833333332e-10*tk4 + 1.253294e-13*tk5 + (-3.020811e+04 - 
          tk1*2.590233); 
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
    // Species H2O2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[7] = 4.573167*tk1*(1-tklog) + -2.168068e-03*tk2 + 2.457815e-07*tk3 
          + -1.957419999999999e-11*tk4 + 7.158269999999999e-16*tk5 + 
          (-1.800696e+04 - tk1*0.5011370000000001); 
      }
      else
      {
        cgspl[7] = 3.388754*tk1*(1-tklog) + -3.284613e-03*tk2 + 
          2.475021666666667e-08*tk3 + 3.854838333333332e-10*tk4 + 
          -1.2357575e-13*tk5 + (-1.766315e+04 - tk1*6.785363); 
      }
    }
    // Species CO
    {
      if (tk1 > 1.429e+03)
      {
        cgspl[8] = 3.1121689*tk1*(1-tklog) + -5.79741415e-04*tk2 + 
          5.641339366666667e-08*tk3 + -3.678359149999999e-12*tk4 + 
          1.06431114e-16*tk5 + (-1.42718539e+04 - tk1*5.71725177); 
      }
      else
      {
        cgspl[8] = 3.19036352*tk1*(1-tklog) + -4.47209986e-04*tk2 + 
          5.415459383333334e-09*tk3 + 8.716663916666665e-12*tk4 + 
          -1.209828465e-15*tk5 + (-1.42869054e+04 - tk1*5.33277914); 
      }
    }
    // Species CO2
    {
      if (tk1 > 1.38e+03)
      {
        cgspl[9] = 5.18953018*tk1*(1-tklog) + -1.03003238e-03*tk2 + 
          1.22262554e-07*tk3 + -9.750364499999997e-12*tk4 + 
          3.458646074999999e-16*tk5 + (-4.93178953e+04 - tk1*-5.18289303); 
      }
      else
      {
        cgspl[9] = 2.5793049*tk1*(1-tklog) + -4.123424935e-03*tk2 + 
          1.071193411666667e-06*tk3 + -2.1219752e-10*tk4 + 
          2.060152214999999e-14*tk5 + (-4.8416283e+04 - tk1*8.811410410000001); 
      }
    }
    // Species CH2O
    {
      if (tk1 > 1.486e+03)
      {
        cgspl[10] = 4.02068394*tk1*(1-tklog) + -2.549517085e-03*tk2 + 
          2.940508000000001e-07*tk3 + -2.300215658333333e-11*tk4 + 
          8.049902099999998e-16*tk5 + (-1.49287258e+04 - tk1*1.06525547); 
      }
      else
      {
        cgspl[10] = 3.00754197*tk1*(1-tklog) + -1.52364748e-03*tk2 + 
          -8.751820766666667e-07*tk3 + 4.266827341666666e-10*tk4 + 
          -6.356689749999998e-14*tk5 + (-1.41188397e+04 - tk1*8.10120233); 
      }
    }
    // Species HCO
    {
      if (tk1 > 1.69e+03)
      {
        cgspl[11] = 3.44148164*tk1*(1-tklog) + -1.760788595e-03*tk2 + 
          2.0689353e-07*tk3 + -1.644405366666666e-11*tk4 + 
          5.826930799999998e-16*tk5 + (3.97409684e+03 - tk1*6.24593456); 
      }
      else
      {
        cgspl[11] = 3.81049965*tk1*(1-tklog) + -4.066349125e-04*tk2 + 
          -5.219411683333334e-07*tk3 + 1.995652233333333e-10*tk4 + 
          -2.53447277e-14*tk5 + (4.03859901e+03 - tk1*4.94843165); 
      }
    }
    // Species HO2CHO
    {
      if (tk1 > 1.378e+03)
      {
        cgspl[12] = 9.875038780000001*tk1*(1-tklog) + -2.32331854e-03*tk2 + 
          2.787175366666667e-07*tk3 + -2.238536774999999e-11*tk4 + 
          7.979761599999998e-16*tk5 + (-3.80502496e+04 - tk1*-22.4939155); 
      }
      else
      {
        cgspl[12] = 2.42464726*tk1*(1-tklog) + -0.010985319*tk2 + 
          2.811759100000001e-06*tk3 + -5.213434949999999e-10*tk4 + 
          4.558229214999999e-14*tk5 + (-3.54828006e+04 - tk1*17.5027796); 
      }
    }
    // Species O2CHO
    {
      if (tk1 > 1.368e+03)
      {
        cgspl[13] = 7.24075139*tk1*(1-tklog) + -2.316564755e-03*tk2 + 
          2.72823325e-07*tk3 + -2.164222441666666e-11*tk4 + 
          7.648234949999998e-16*tk5 + (-1.87027618e+04 - tk1*-6.49547212); 
      }
      else
      {
        cgspl[13] = 3.96059309*tk1*(1-tklog) + -5.30011395e-03*tk2 + 
          8.761889183333334e-07*tk3 + -8.476393833333331e-11*tk4 + 
          1.43743801e-15*tk5 + (-1.73599383e+04 - tk1*11.7807483); 
      }
    }
    // Species OCHO
    {
      if (tk1 > 1.412e+03)
      {
        cgspl[14] = 6.12628782*tk1*(1-tklog) + -1.87801466e-03*tk2 + 
          2.3668392e-07*tk3 + -1.970243333333333e-11*tk4 + 
          7.208382549999998e-16*tk5 + (-2.17698466e+04 - tk1*-8.01574694); 
      }
      else
      {
        cgspl[14] = 1.35213452*tk1*(1-tklog) + -7.5041002e-03*tk2 + 
          1.83160235e-06*tk3 + -3.113998666666666e-10*tk4 + 
          2.405072489999999e-14*tk5 + (-2.02253647e+04 - tk1*17.4373147); 
      }
    }
    // Species CH2OH
    {
      if (tk1 > 1.399e+03)
      {
        cgspl[15] = 5.41875913*tk1*(1-tklog) + -2.83092622e-03*tk2 + 
          3.124518933333334e-07*tk3 + -2.370349483333333e-11*tk4 + 
          8.114989899999999e-16*tk5 + (-3.6147554e+03 - tk1*-3.49277963); 
      }
      else
      {
        cgspl[15] = 3.05674228*tk1*(1-tklog) + -5.9667817e-03*tk2 + 
          1.454168838333334e-06*tk3 + -3.189842116666666e-10*tk4 + 
          3.614439754999999e-14*tk5 + (-2.8314019e+03 - tk1*8.98878133); 
      }
    }
    // Species CH3O
    {
      if (tk1 > 1.509e+03)
      {
        cgspl[16] = 4.64787019*tk1*(1-tklog) + -3.454153415e-03*tk2 + 
          3.906746266666667e-07*tk3 + -3.016621416666666e-11*tk4 + 
          1.046267705e-15*tk5 + (-299.208881 - tk1*-1.57740193); 
      }
      else
      {
        cgspl[16] = 2.23058023*tk1*(1-tklog) + -4.26589293e-03*tk2 + 
          -1.702777066666667e-07*tk3 + 2.842057633333333e-10*tk4 + 
          -4.973455189999999e-14*tk5 + (945.939708 - tk1*12.8377569); 
      }
    }
    // Species CH3O2H
    {
      if (tk1 > 1.367e+03)
      {
        cgspl[17] = 8.80409289*tk1*(1-tklog) + -4.04713609e-03*tk2 + 
          4.764054566666668e-07*tk3 + -3.778081283333333e-11*tk4 + 
          1.334903535e-15*tk5 + (-1.98512174e+04 - tk1*-21.7000591); 
      }
      else
      {
        cgspl[17] = 2.83880024*tk1*(1-tklog) + -9.30481245e-03*tk2 + 
          1.41360902e-06*tk3 + -8.365620916666665e-11*tk4 + 
          -8.580621449999998e-15*tk5 + (-1.74033753e+04 - tk1*11.6092433); 
      }
    }
    // Species CH3O2
    {
      if (tk1 > 1.365e+03)
      {
        cgspl[18] = 6.34718801*tk1*(1-tklog) + -3.96044679e-03*tk2 + 
          4.610031883333334e-07*tk3 + -3.628005258333333e-11*tk4 + 
          1.27492381e-15*tk5 + (-1.83436055e+03 - tk1*-7.42552545); 
      }
      else
      {
        cgspl[18] = 3.8049759*tk1*(1-tklog) + -4.9039233e-03*tk2 + 
          6.515677066666668e-08*tk3 + 1.858938349999999e-10*tk4 + 
          -3.216554099999999e-14*tk5 + (-455.625796 - tk1*7.817891); 
      }
    }
    // Species CH4
    {
      if (tk1 > 1.462e+03)
      {
        cgspl[19] = 4.09617653*tk1*(1-tklog) + -3.721654225e-03*tk2 + 
          4.397865000000001e-07*tk3 + -3.496480033333333e-11*tk4 + 
          1.23754025e-15*tk5 + (-1.13835704e+04 - tk1*-4.67561383); 
      }
      else
      {
        cgspl[19] = 3.7211302*tk1*(1-tklog) + 1.251466445e-03*tk2 + 
          -3.170775566666667e-06*tk3 + 1.223927108333333e-09*tk4 + 
          -1.71895576e-13*tk5 + (-1.01424099e+04 - tk1*1.22776596); 
      }
    }
    // Species CH3
    {
      if (tk1 > 1.389e+03)
      {
        cgspl[20] = 3.51281376*tk1*(1-tklog) + -2.557063065e-03*tk2 + 
          2.7938675e-07*tk3 + -2.10412645e-11*tk4 + 7.165146149999999e-16*tk5 + 
          (1.61238027e+04 - tk1*1.62436112); 
      }
      else
      {
        cgspl[20] = 3.43858162*tk1*(1-tklog) + -2.03876332e-03*tk2 + 
          -5.330516566666667e-08*tk3 + 7.897244916666666e-11*tk4 + 
          -1.10914083e-14*tk5 + (1.63164018e+04 - tk1*2.52807406); 
      }
    }
    // Species CH2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[21] = 3.636408*tk1*(1-tklog) + -9.665285e-04*tk2 + 
          2.811693333333334e-08*tk3 + 8.415824999999999e-12*tk4 + 
          -9.041279999999997e-16*tk5 + (4.534134e+04 - tk1*2.156561); 
      }
      else
      {
        cgspl[21] = 3.762237*tk1*(1-tklog) + -5.799095e-04*tk2 + 
          -4.149308333333334e-08*tk3 + -7.334029999999998e-11*tk4 + 
          3.666217499999999e-14*tk5 + (4.536791e+04 - tk1*1.712578); 
      }
    }
    // Species CH2(S)
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[22] = 3.552889*tk1*(1-tklog) + -1.033394e-03*tk2 + 
          3.190193333333334e-08*tk3 + 9.205608333333331e-12*tk4 + 
          -1.010675e-15*tk5 + (4.984975e+04 - tk1*1.68657); 
      }
      else
      {
        cgspl[22] = 3.971265*tk1*(1-tklog) + 8.495445e-05*tk2 + 
          -1.708948333333334e-07*tk3 + -2.077125833333333e-10*tk4 + 
          9.906329999999997e-14*tk5 + (4.989368e+04 - tk1*0.05753207); 
      }
    }
    // Species C2H6
    {
      if (tk1 > 1.383e+03)
      {
        cgspl[23] = 6.0597263*tk1*(1-tklog) + -6.51914185e-03*tk2 + 
          7.468399033333334e-07*tk3 + -5.814684124999999e-11*tk4 + 
          2.028031765e-15*tk5 + (-1.35751226e+04 - tk1*-12.8608001); 
      }
      else
      {
        cgspl[23] = 0.0478623203*tk1*(1-tklog) + -0.01202845635*tk2 + 
          1.9192652e-06*tk3 + -2.07221865e-10*tk4 + 8.917197199999998e-15*tk5 + 
          (-1.10923014e+04 - tk1*20.6544071); 
      }
    }
    // Species C2H5
    {
      if (tk1 > 1.387e+03)
      {
        cgspl[24] = 5.8878439*tk1*(1-tklog) + -5.15383965e-03*tk2 + 
          5.780739933333334e-07*tk3 + -4.437493808333332e-11*tk4 + 
          1.532563254999999e-15*tk5 + (1.15065499e+04 - tk1*-8.496517709999999); 
      }
      else
      {
        cgspl[24] = 1.32730217*tk1*(1-tklog) + -8.83283765e-03*tk2 + 
          1.024877596666667e-06*tk3 + 2.509528883333333e-11*tk4 + 
          -2.193088874999999e-14*tk5 + (1.34284028e+04 - tk1*17.1789216); 
      }
    }
    // Species C2H4
    {
      if (tk1 > 1.395e+03)
      {
        cgspl[25] = 5.22176372*tk1*(1-tklog) + -4.480686515e-03*tk2 + 
          5.081148100000001e-07*tk3 + -3.928879366666666e-11*tk4 + 
          1.36369796e-15*tk5 + (3.60389679e+03 - tk1*-7.47789234); 
      }
      else
      {
        cgspl[25] = 0.233879687*tk1*(1-tklog) + -9.816732349999999e-03*tk2 + 
          1.947220233333333e-06*tk3 + -3.035387108333332e-10*tk4 + 
          2.387213575e-14*tk5 + (5.46489338e+03 - tk1*19.7084228); 
      }
    }
    // Species C2H3
    {
      if (tk1 > 1.395e+03)
      {
        cgspl[26] = 5.07331248*tk1*(1-tklog) + -3.29158139e-03*tk2 + 
          3.729382066666667e-07*tk3 + -2.881694824999999e-11*tk4 + 
          9.997024499999996e-16*tk5 + (3.37234748e+04 - tk1*-3.39792712); 
      }
      else
      {
        cgspl[26] = 1.25329724*tk1*(1-tklog) + -7.8129185e-03*tk2 + 
          1.796731316666667e-06*tk3 + -3.483788616666666e-10*tk4 + 
          3.506801809999999e-14*tk5 + (3.50734773e+04 - tk1*17.1341661); 
      }
    }
    // Species C2H2
    {
      if (tk1 > 1.407e+03)
      {
        cgspl[27] = 4.98265164*tk1*(1-tklog) + -2.12996465e-03*tk2 + 
          2.29139205e-07*tk3 + -1.705983033333333e-11*tk4 + 
          5.759586999999998e-16*tk5 + (2.52697118e+04 - tk1*-5.81321385); 
      }
      else
      {
        cgspl[27] = 2.06742667*tk1*(1-tklog) + -7.3284253e-03*tk2 + 
          2.549117716666667e-06*tk3 + -6.924714674999999e-10*tk4 + 
          8.646608749999997e-14*tk5 + (2.59578589e+04 - tk1*8.62758672); 
      }
    }
    // Species CH3CHO
    {
      if (tk1 > 1.377e+03)
      {
        cgspl[28] = 6.98518866*tk1*(1-tklog) + -4.839488935e-03*tk2 + 
          5.530699233333334e-07*tk3 + -4.300215841666666e-11*tk4 + 
          1.498629515e-15*tk5 + (-2.39807279e+04 - tk1*-12.7484852); 
      }
      else
      {
        cgspl[28] = 1.77060035*tk1*(1-tklog) + -9.223758049999999e-03*tk2 + 
          1.206896936666667e-06*tk3 + -1.953038008333333e-11*tk4 + 
          -1.677719455e-14*tk5 + (-2.1807885e+04 - tk1*16.5023437); 
      }
    }
    // Species CH3CO
    {
      if (tk1 > 1.371e+03)
      {
        cgspl[29] = 6.56682466*tk1*(1-tklog) + -3.776543345e-03*tk2 + 
          4.332779733333334e-07*tk3 + -3.377791616666666e-11*tk4 + 
          1.17938082e-15*tk5 + (-4.76690401e+03 - tk1*-8.833019650000001); 
      }
      else
      {
        cgspl[29] = 2.5288415*tk1*(1-tklog) + -6.85760865e-03*tk2 + 
          7.143457933333335e-07*tk3 + 6.430702316666666e-11*tk4 + 
          -2.419181899999999e-14*tk5 + (-3.02546532e+03 - tk1*14.0340315); 
      }
    }
    // Species CH2CO
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[30] = 6.038817*tk1*(1-tklog) + -2.90242e-03*tk2 + 3.20159e-07*tk3 
          + -2.328737499999999e-11*tk4 + 7.294339999999998e-16*tk5 + 
          (-8.583402e+03 - tk1*-7.657581); 
      }
      else
      {
        cgspl[30] = 2.974971*tk1*(1-tklog) + -6.059355e-03*tk2 + 
          3.908410000000001e-07*tk3 + 5.388904166666665e-10*tk4 + 
          -1.9528245e-13*tk5 + (-7.632637e+03 - tk1*8.673553); 
      }
    }
    // Species HCCO
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[31] = 6.758073*tk1*(1-tklog) + -1.0002e-03*tk2 + 3.379345e-08*tk3 
          + 8.676099999999999e-12*tk4 + -9.825824999999998e-16*tk5 + 
          (1.901513e+04 - tk1*-9.071262000000001); 
      }
      else
      {
        cgspl[31] = 5.047965*tk1*(1-tklog) + -2.226739e-03*tk2 + 
          -3.780471666666667e-08*tk3 + 1.235079166666666e-10*tk4 + 
          -1.125371e-14*tk5 + (1.965892e+04 - tk1*0.4818439); 
      }
    }
    // Species C2H5O
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[32] = 8.23717244*tk1*(1-tklog) + -5.54429395e-03*tk2 + 
          6.313472866666667e-07*tk3 + -4.896778941666665e-11*tk4 + 
          1.703564445e-15*tk5 + (-6.22948597e+03 - tk1*-19.3190543); 
      }
      else
      {
        cgspl[32] = 0.287429022*tk1*(1-tklog) + -0.0143250459*tk2 + 
          3.064283466666667e-06*tk3 + -5.025801491666666e-10*tk4 + 
          4.022813214999999e-14*tk5 + (-3.35717377e+03 - tk1*23.7513898); 
      }
    }
    // Species C2H5O2
    {
      if (tk1 > 1.387e+03)
      {
        cgspl[33] = 8.776413290000001*tk1*(1-tklog) + -5.93835815e-03*tk2 + 
          6.461384166666668e-07*tk3 + -4.860089616666666e-11*tk4 + 
          1.65503362e-15*tk5 + (-6.78748703e+03 - tk1*-18.3119972); 
      }
      else
      {
        cgspl[33] = 2.58630333*tk1*(1-tklog) + -0.0130918181*tk2 + 
          2.805103216666667e-06*tk3 + -5.072910791666666e-10*tk4 + 
          4.786510199999999e-14*tk5 + (-4.58588992e+03 - tk1*15.0486289); 
      }
    }
    // Species C2H3O1-2
    {
      if (tk1 > 1.492e+03)
      {
        cgspl[34] = 6.88486471*tk1*(1-tklog) + -3.473602505e-03*tk2 + 
          3.720244966666667e-07*tk3 + -2.768256391666666e-11*tk4 + 
          9.356227749999997e-16*tk5 + (1.264422e+04 - tk1*-12.384257); 
      }
      else
      {
        cgspl[34] = -1.62965122*tk1*(1-tklog) + -0.0146727743*tk2 + 
          4.0622925e-06*tk3 + -8.376860416666664e-10*tk4 + 
          8.062951799999998e-14*tk5 + (1.52459425e+04 - tk1*32.2782741); 
      }
    }
    // Species CH3COCH3
    {
      if (tk1 > 1.382e+03)
      {
        cgspl[35] = 9.626743790000001*tk1*(1-tklog) + -7.27596225e-03*tk2 + 
          8.295824283333334e-07*tk3 + -6.439954924999998e-11*tk4 + 
          2.241835825e-15*tk5 + (-3.11862263e+04 - tk1*-26.1613449); 
      }
      else
      {
        cgspl[35] = 1.24527408*tk1*(1-tklog) + -0.01498801275*tk2 + 
          2.333777683333333e-06*tk3 + -1.803779266666666e-10*tk4 + 
          -6.381864749999998e-15*tk5 + (-2.78348727e+04 - tk1*20.3682615); 
      }
    }
    // Species CH3COCH2
    {
      if (tk1 > 1.388e+03)
      {
        cgspl[36] = 10.8892477*tk1*(1-tklog) + -5.57703375e-03*tk2 + 
          6.42527975e-07*tk3 + -5.023617066666666e-11*tk4 + 1.757667245e-15*tk5 
          + (-1.00741464e+04 - tk1*-31.8043322); 
      }
      else
      {
        cgspl[36] = 1.22337251*tk1*(1-tklog) + -0.0162273371*tk2 + 
          3.559041966666667e-06*tk3 + -5.806481124999999e-10*tk4 + 
          4.495801494999999e-14*tk5 + (-6.59419324e+03 - tk1*20.5537233); 
      }
    }
    // Species C2H3CHO
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[37] = 10.4184959*tk1*(1-tklog) + -4.744816605e-03*tk2 + 
          5.488508816666668e-07*tk3 + -4.302326691666665e-11*tk4 + 
          1.507936455e-15*tk5 + (-1.49630281e+04 - tk1*-30.7235061); 
      }
      else
      {
        cgspl[37] = 0.292355162*tk1*(1-tklog) + -0.01771607085*tk2 + 
          4.9156054e-06*tk3 + -1.067501033333333e-09*tk4 + 1.13072054e-13*tk5 + 
          (-1.16521584e+04 - tk1*22.887828); 
      }
    }
    // Species C2H3CO
    {
      if (tk1 > 1.402e+03)
      {
        cgspl[38] = 9.37467676*tk1*(1-tklog) + -3.9564845e-03*tk2 + 
          4.453304666666667e-07*tk3 + -3.425961916666666e-11*tk4 + 
          1.184894905e-15*tk5 + (1.92969514e+03 - tk1*-24.0892696); 
      }
      else
      {
        cgspl[38] = 1.36242013*tk1*(1-tklog) + -0.0157636986*tk2 + 
          5.003648916666667e-06*tk3 + -1.234725933333333e-09*tk4 + 
          1.43985765e-13*tk5 + (4.25770215e+03 - tk1*17.2626546); 
      }
    }
    // Species C2H5CO
    {
      if (tk1 > 1.362e+03)
      {
        cgspl[39] = 10.0147418*tk1*(1-tklog) + -8.076398049999999e-03*tk2 + 
          1.104960203333333e-06*tk3 + -9.60755558333333e-11*tk4 + 
          3.606317289999999e-15*tk5 + (-1.00430305e+04 - tk1*-28.8570933); 
      }
      else
      {
        cgspl[39] = 8.353522460000001*tk1*(1-tklog) + 2.043698635e-03*tk2 + 
          -6.070299233333333e-06*tk3 + 2.270480941666666e-09*tk4 + 
          -3.03220133e-13*tk5 + (-6.58577307e+03 - tk1*-10.5948346); 
      }
    }
    // Species IC3H7
    {
      if (tk1 > 1.373e+03)
      {
        cgspl[40] = 8.14705217*tk1*(1-tklog) + -7.9363553e-03*tk2 + 
          9.076859016666668e-07*tk3 + -7.060064074999998e-11*tk4 + 
          2.460893759999999e-15*tk5 + (6.18073367e+03 - tk1*-19.198085); 
      }
      else
      {
        cgspl[40] = 1.63417589*tk1*(1-tklog) + -0.0120085686*tk2 + 
          7.880134450000001e-07*tk3 + 2.702955025e-10*tk4 + 
          -6.176952199999998e-14*tk5 + (9.20752889e+03 - tk1*18.3848082); 
      }
    }
    // Species C3H6
    {
      if (tk1 > 1.388e+03)
      {
        cgspl[41] = 8.015959580000001*tk1*(1-tklog) + -6.8511817e-03*tk2 + 
          7.770828883333334e-07*tk3 + -6.010453349999999e-11*tk4 + 
          2.086850629999999e-15*tk5 + (-1.76749303e+03 - tk1*-20.0160668); 
      }
      else
      {
        cgspl[41] = 0.394615444*tk1*(1-tklog) + -0.0144553831*tk2 + 
          2.5814468e-06*tk3 + -3.240118408333333e-10*tk4 + 1.68945176e-14*tk5 + 
          (1.17760132e+03 - tk1*21.9003736); 
      }
    }
    // Species C3H5-A
    {
      if (tk1 > 1.397e+03)
      {
        cgspl[42] = 8.458839579999999*tk1*(1-tklog) + -5.63477415e-03*tk2 + 
          6.396547733333333e-07*tk3 + -4.950492658333332e-11*tk4 + 
          1.71959015e-15*tk5 + (1.64683289e+04 - tk1*-23.2704266); 
      }
      else
      {
        cgspl[42] = -0.529131958*tk1*(1-tklog) + -0.016727955*tk2 + 
          4.223350450000001e-06*tk3 + -8.572146166666664e-10*tk4 + 
          8.662916999999998e-14*tk5 + (1.94941423e+04 - tk1*24.6172315); 
      }
    }
    // Species C3H5-S
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[43] = 7.88765879*tk1*(1-tklog) + -5.65062955e-03*tk2 + 
          6.403552166666667e-07*tk3 + -4.949855641666665e-11*tk4 + 
          1.717835874999999e-15*tk5 + (2.83484913e+04 - tk1*-17.4291589); 
      }
      else
      {
        cgspl[43] = 1.32807335*tk1*(1-tklog) + -0.0126553957*tk2 + 
          2.525507316666667e-06*tk3 + -3.952879708333333e-10*tk4 + 
          3.12333042e-14*tk5 + (3.079811e+04 - tk1*18.3328787); 
      }
    }
    // Species C3H5-T
    {
      if (tk1 > 1.382e+03)
      {
        cgspl[44] = 7.37492443*tk1*(1-tklog) + -5.87550305e-03*tk2 + 
          6.667021383333334e-07*tk3 + -5.157894958333332e-11*tk4 + 
          1.79107509e-15*tk5 + (2.73982108e+04 - tk1*-14.3478655); 
      }
      else
      {
        cgspl[44] = 2.17916644*tk1*(1-tklog) + -0.01019133115*tk2 + 
          1.319023056666667e-06*tk3 + -3.974218224999999e-11*tk4 + 
          -1.35199268e-14*tk5 + (2.96002535e+04 - tk1*14.8785684); 
      }
    }
    // Species C3H4-P
    {
      if (tk1 > 1.4e+03)
      {
        cgspl[45] = 9.768102000000001*tk1*(1-tklog) + -2.6095755e-03*tk2 + 
          6.255233333333335e-08*tk3 + 2.493492499999999e-11*tk4 + 
          -2.553939e-15*tk5 + (1.860277e+04 - tk1*-30.20678); 
      }
      else
      {
        cgspl[45] = 3.02973*tk1*(1-tklog) + -7.494805e-03*tk2 + 
          2.330833333333334e-07*tk3 + 3.308015833333333e-10*tk4 + 
          -6.941084999999999e-14*tk5 + (2.148408e+04 - tk1*8.004594000000001); 
      }
    }
    // Species C3H4-A
    {
      if (tk1 > 1.4e+03)
      {
        cgspl[46] = 9.776256*tk1*(1-tklog) + -2.651069e-03*tk2 + 6.16853e-08*tk3 
          + 2.521988333333333e-11*tk4 + -2.5447905e-15*tk5 + (1.954972e+04 - 
          tk1*-30.77061); 
      }
      else
      {
        cgspl[46] = 2.539831*tk1*(1-tklog) + -8.167185e-03*tk2 + 
          2.941583333333334e-07*tk3 + 3.872804166666666e-10*tk4 + 
          -8.645654999999998e-14*tk5 + (2.251243e+04 - tk1*9.935701999999999); 
      }
    }
    // Species C3H3
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[47] = 8.831047*tk1*(1-tklog) + -2.1785975e-03*tk2 + 
          6.848445000000001e-08*tk3 + 1.973935833333333e-11*tk4 + 
          -2.188259999999999e-15*tk5 + (3.84742e+04 - tk1*-21.77919); 
      }
      else
      {
        cgspl[47] = 4.7542*tk1*(1-tklog) + -5.54014e-03*tk2 + 
          -4.655538333333333e-08*tk3 + 4.566009999999999e-10*tk4 + 
          -9.748144999999997e-14*tk5 + (3.988883e+04 - tk1*0.5854549); 
      }
    }
    // Species C3H2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[48] = 7.670981*tk1*(1-tklog) + -1.3743745e-03*tk2 + 
          7.284905e-08*tk3 + 5.379665833333333e-12*tk4 + 
          -8.319434999999998e-16*tk5 + (6.259722e+04 - tk1*-12.3689); 
      }
      else
      {
        cgspl[48] = 3.166714*tk1*(1-tklog) + -0.01241286*tk2 + 
          7.652728333333333e-06*tk3 + -3.556682499999999e-09*tk4 + 
          7.410759999999998e-13*tk5 + (6.350421e+04 - tk1*8.869446); 
      }
    }
    // Species C3H5O
    {
      if (tk1 > 1.38e+03)
      {
        cgspl[49] = 10.2551752*tk1*(1-tklog) + -5.749186e-03*tk2 + 
          6.410760983333334e-07*tk3 + -4.907586216666666e-11*tk4 + 
          1.692789615e-15*tk5 + (6.2656081e+03 - tk1*-27.7655042); 
      }
      else
      {
        cgspl[49] = 1.19822582*tk1*(1-tklog) + -0.01527899185*tk2 + 
          3.0105046e-06*tk3 + -4.051250274999999e-10*tk4 + 2.09927281e-14*tk5 + 
          (9.58217784e+03 - tk1*21.5566221); 
      }
    }
    // Species C3H6OOH2-1
    {
      if (tk1 > 1.407e+03)
      {
        cgspl[50] = 14.2163221*tk1*(1-tklog) + -7.1691225e-03*tk2 + 
          7.966741283333335e-07*tk3 + -6.076109449999999e-11*tk4 + 
          2.088809864999999e-15*tk5 + (-5.6738162e+03 - tk1*-43.5770997); 
      }
      else
      {
        cgspl[50] = 2.0919395*tk1*(1-tklog) + -0.0234610197*tk2 + 
          6.504680516666667e-06*tk3 + -1.436512108333333e-09*tk4 + 
          1.539844895e-13*tk5 + (-1.89377918e+03 - tk1*20.0178282); 
      }
    }
    // Species C3H6OOH2-1O2
    {
      if (tk1 > 1.386e+03)
      {
        cgspl[51] = 19.1759159*tk1*(1-tklog) + -7.992850649999999e-03*tk2 + 
          9.355106300000001e-07*tk3 + -7.390670791666665e-11*tk4 + 
          2.604385199999999e-15*tk5 + (-2.64412115e+04 - tk1*-67.7512936); 
      }
      else
      {
        cgspl[51] = 2.65196584*tk1*(1-tklog) + -0.02873190745*tk2 + 
          7.869847783333334e-06*tk3 + -1.713262975e-09*tk4 + 1.843936935e-13*tk5 
          + (-2.08829371e+04 - tk1*20.1547955); 
      }
    }
    // Species IC3H7O2
    {
      if (tk1 > 1.388e+03)
      {
        cgspl[52] = 13.2493493*tk1*(1-tklog) + -8.204109500000001e-03*tk2 + 
          9.457201033333334e-07*tk3 + -7.394469499999998e-11*tk4 + 
          2.586807674999999e-15*tk5 + (-1.44109855e+04 - tk1*-42.9066213); 
      }
      else
      {
        cgspl[52] = 1.49941639*tk1*(1-tklog) + -0.02215406025*tk2 + 
          5.373574266666667e-06*tk3 + -1.080726133333333e-09*tk4 + 
          1.116852845e-13*tk5 + (-1.0258798e+04 - tk1*20.233649); 
      }
    }
    // Species C3KET21
    {
      if (tk1 > 1.371e+03)
      {
        cgspl[53] = 15.6377776*tk1*(1-tklog) + -7.2029671e-03*tk2 + 
          8.480134700000001e-07*tk3 + -6.725634324999998e-11*tk4 + 
          2.376478249999999e-15*tk5 + (-4.30657975e+04 - tk1*-51.3105869); 
      }
      else
      {
        cgspl[53] = 4.55686367*tk1*(1-tklog) + -0.01785384185*tk2 + 
          3.245200900000001e-06*tk3 + -3.922461924999999e-10*tk4 + 
          1.848769035e-14*tk5 + (-3.86710975e+04 - tk1*9.977616940000001); 
      }
    }
    // Species CH3CHCO
    {
      if (tk1 > 1.4e+03)
      {
        cgspl[54] = 10.0219123*tk1*(1-tklog) + -4.7848315e-03*tk2 + 
          5.437027400000001e-07*tk3 + -4.210264216666666e-11*tk4 + 
          1.462966285e-15*tk5 + (-1.42482738e+04 - tk1*-27.7829973); 
      }
      else
      {
        cgspl[54] = 1.48380119*tk1*(1-tklog) + -0.01611015065*tk2 + 
          4.504167216666667e-06*tk3 + -1.0041597e-09*tk4 + 1.091829655e-13*tk5 + 
          (-1.1527654e+04 - tk1*17.1552068); 
      }
    }
    // Species SC4H9
    {
      if (tk1 > 1.381e+03)
      {
        cgspl[55] = 11.6934304*tk1*(1-tklog) + -9.820114350000001e-03*tk2 + 
          1.108844195e-06*tk3 + -8.552657916666664e-11*tk4 + 
          2.964131469999999e-15*tk5 + (1.96382429e+03 - tk1*-36.1626672); 
      }
      else
      {
        cgspl[55] = 0.849159986*tk1*(1-tklog) + -0.019104266*tk2 + 
          2.49377995e-06*tk3 + -1.704160091666666e-11*tk4 + 
          -4.121272184999999e-14*tk5 + (6.38832956e+03 - tk1*24.4466606); 
      }
    }
    // Species IC4H9
    {
      if (tk1 > 1.386e+03)
      {
        cgspl[56] = 12.127693*tk1*(1-tklog) + -9.934474699999999e-03*tk2 + 
          1.14322834e-06*tk3 + -8.928460666666665e-11*tk4 + 
          3.120923044999999e-15*tk5 + (2.11952051e+03 - tk1*-40.8727278); 
      }
      else
      {
        cgspl[56] = -0.221457835*tk1*(1-tklog) + -0.0231878162*tk2 + 
          4.804715333333334e-06*tk3 + -8.001672049999998e-10*tk4 + 
          6.951051699999997e-14*tk5 + (6.76153637e+03 - tk1*26.480122); 
      }
    }
    // Species TC4H9
    {
      if (tk1 > 1.372e+03)
      {
        cgspl[57] = 10.5855083*tk1*(1-tklog) + -0.01059460275*tk2 + 
          1.219447266666667e-06*tk3 + -9.524681083333331e-11*tk4 + 
          3.329494174999999e-15*tk5 + (318.311189 - tk1*-32.0671258); 
      }
      else
      {
        cgspl[57] = 3.04300181*tk1*(1-tklog) + -0.01456381295*tk2 + 
          6.689779766666667e-07*tk3 + 4.085616449999999e-10*tk4 + 
          -8.200744549999999e-14*tk5 + (4.0478098e+03 - tk1*12.1127351); 
      }
    }
    // Species IC4H8
    {
      if (tk1 > 1.388e+03)
      {
        cgspl[58] = 11.225833*tk1*(1-tklog) + -9.089789900000001e-03*tk2 + 
          1.03391432e-06*tk3 + -8.012037149999999e-11*tk4 + 
          2.785440284999999e-15*tk5 + (-7.90617899e+03 - tk1*-36.6411888); 
      }
      else
      {
        cgspl[58] = 0.938433173*tk1*(1-tklog) + -0.01952736435*tk2 + 
          3.6072858e-06*tk3 + -4.893892308333332e-10*tk4 + 3.072177395e-14*tk5 + 
          (-3.95452013e+03 - tk1*19.8337802); 
      }
    }
    // Species IC4H7
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[59] = 11.6382753*tk1*(1-tklog) + -7.88406495e-03*tk2 + 
          8.975647633333334e-07*tk3 + -6.959774391666665e-11*tk4 + 
          2.420705414999999e-15*tk5 + (1.0340823e+04 - tk1*-39.025989); 
      }
      else
      {
        cgspl[59] = -7.20881697e-04*tk1*(1-tklog) + -0.0218247865*tk2 + 
          5.27309795e-06*tk3 + -1.033208191666666e-09*tk4 + 1.0218918e-13*tk5 + 
          (1.43654373e+04 - tk1*23.323434); 
      }
    }
    // Species TC4H9O2
    {
      if (tk1 > 1.388e+03)
      {
        cgspl[60] = 16.7061556*tk1*(1-tklog) + -0.010366389*tk2 + 
          1.195993615e-06*tk3 + -9.356872749999999e-11*tk4 + 
          3.274706929999999e-15*tk5 + (-2.04046924e+04 - tk1*-63.5558608); 
      }
      else
      {
        cgspl[60] = 1.08742583*tk1*(1-tklog) + -0.02913903545*tk2 + 
          7.221544950000001e-06*tk3 + -1.474104458333333e-09*tk4 + 
          1.53384803e-13*tk5 + (-1.4945664e+04 - tk1*20.1871963); 
      }
    }
    // Species IC4H9O2
    {
      if (tk1 > 1.387e+03)
      {
        cgspl[61] = 15.9741221*tk1*(1-tklog) + -0.010676737*tk2 + 
          1.231668508333333e-06*tk3 + -9.635367583333332e-11*tk4 + 
          3.372040229999999e-15*tk5 + (-1.72329304e+04 - tk1*-56.5302409); 
      }
      else
      {
        cgspl[61] = 1.21434293*tk1*(1-tklog) + -0.02726941555*tk2 + 
          6.116693216666668e-06*tk3 + -1.117758683333333e-09*tk4 + 
          1.058708965e-13*tk5 + (-1.1848245e+04 - tk1*23.4153048); 
      }
    }
    // Species IC4H8O2H-I
    {
      if (tk1 > 1.388e+03)
      {
        cgspl[62] = 18.0246456*tk1*(1-tklog) + -9.683413199999999e-03*tk2 + 
          1.124460826666667e-06*tk3 + -8.836677416666664e-11*tk4 + 
          3.102532225e-15*tk5 + (-1.00858977e+04 - tk1*-65.76296929999999); 
      }
      else
      {
        cgspl[62] = 0.994784793*tk1*(1-tklog) + -0.029460612*tk2 + 
          7.086703750000001e-06*tk3 + -1.344754783333333e-09*tk4 + 
          1.27952451e-13*tk5 + (-4.08029057e+03 - tk1*25.895088); 
      }
    }
    // Species TC4H9O
    {
      if (tk1 > 1.397e+03)
      {
        cgspl[63] = 15.0819361*tk1*(1-tklog) + -9.722706399999999e-03*tk2 + 
          1.102222683333334e-06*tk3 + -8.523263333333331e-11*tk4 + 
          2.958984639999999e-15*tk5 + (-1.88111456e+04 - tk1*-57.1658947); 
      }
      else
      {
        cgspl[63] = -0.532084074*tk1*(1-tklog) + -0.02858067345*tk2 + 
          7.002953800000002e-06*tk3 + -1.364545208333333e-09*tk4 + 
          1.3206235e-13*tk5 + (-1.34963439e+04 - tk1*26.2776957); 
      }
    }
    // Species TC4H9O2H
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[64] = 19.1617041*tk1*(1-tklog) + -0.01045786245*tk2 + 
          1.212244701666667e-06*tk3 + -9.515082416666664e-11*tk4 + 
          3.337885674999999e-15*tk5 + (-3.83201687e+04 - 
          tk1*-77.82308740000001); 
      }
      else
      {
        cgspl[64] = 0.114505932*tk1*(1-tklog) + -0.0335753927*tk2 + 
          8.592973500000001e-06*tk3 + -1.752136375e-09*tk4 + 1.78233512e-13*tk5 
          + (-3.17928725e+04 - tk1*24.001103); 
      }
    }
    // Species IC4H7O
    {
      if (tk1 > 1.386e+03)
      {
        cgspl[65] = 13.3457615*tk1*(1-tklog) + -8.0609294e-03*tk2 + 
          9.07294005e-07*tk3 + -6.984994783333332e-11*tk4 + 
          2.418041399999999e-15*tk5 + (611.4436439999999 - tk1*-43.6818838); 
      }
      else
      {
        cgspl[65] = 1.74700687*tk1*(1-tklog) + -0.0203891718*tk2 + 
          4.079170716666667e-06*tk3 + -5.887524649999999e-10*tk4 + 
          3.757852944999999e-14*tk5 + (4.86979233e+03 - tk1*19.4535999); 
      }
    }
    // Species IC3H7CHO
    {
      if (tk1 > 1.391e+03)
      {
        cgspl[66] = 13.7501656*tk1*(1-tklog) + -9.156336100000001e-03*tk2 + 
          1.047621048333334e-06*tk3 + -8.152089633333331e-11*tk4 + 
          2.842693264999999e-15*tk5 + (-3.26936771e+04 - tk1*-47.7270548); 
      }
      else
      {
        cgspl[66] = -0.273021382*tk1*(1-tklog) + -0.02448481535*tk2 + 
          5.212834150000001e-06*tk3 + -8.337745416666665e-10*tk4 + 
          6.375603699999999e-14*tk5 + (-2.76054737e+04 - tk1*28.3451139); 
      }
    }
    // Species TC3H6CHO
    {
      if (tk1 > 1.389e+03)
      {
        cgspl[67] = 13.1013047*tk1*(1-tklog) + -8.31959325e-03*tk2 + 
          9.474293716666668e-07*tk3 + -7.348402924999998e-11*tk4 + 
          2.556450804999999e-15*tk5 + (-1.30638647e+04 - tk1*-44.2705813); 
      }
      else
      {
        cgspl[67] = 1.87052762*tk1*(1-tklog) + -0.02074348385*tk2 + 
          4.44692835e-06*tk3 + -7.512763416666665e-10*tk4 + 
          6.393531649999998e-14*tk5 + (-8.97730744e+03 - tk1*16.6174178); 
      }
    }
    // Species IC4H8OOH-IO2
    {
      if (tk1 > 1.385e+03)
      {
        cgspl[68] = 21.8969581*tk1*(1-tklog) + -0.0104818937*tk2 + 
          1.2244415e-06*tk3 + -9.660549499999998e-11*tk4 + 
          3.401127064999999e-15*tk5 + (-2.92664889e+04 - tk1*-82.0540807); 
      }
      else
      {
        cgspl[68] = 2.39424426*tk1*(1-tklog) + -0.03382862745*tk2 + 
          8.618061366666668e-06*tk3 + -1.756633674999999e-09*tk4 + 
          1.799801865e-13*tk5 + (-2.24787495e+04 - tk1*22.5029839); 
      }
    }
    // Species IC4KETII
    {
      if (tk1 > 1.387e+03)
      {
        cgspl[69] = 19.5143059*tk1*(1-tklog) + -9.11886975e-03*tk2 + 
          1.064847676666667e-06*tk3 + -8.400130916666664e-11*tk4 + 
          2.957201749999999e-15*tk5 + (-4.46884836e+04 - tk1*-71.7167584); 
      }
      else
      {
        cgspl[69] = 1.15501614*tk1*(1-tklog) + -0.03053111725*tk2 + 
          7.495188716666668e-06*tk3 + -1.42095545e-09*tk4 + 1.32974301e-13*tk5 + 
          (-3.82747956e+04 - tk1*26.9612235); 
      }
    }
    // Species IC4H7OH
    {
      if (tk1 > 1.384e+03)
      {
        cgspl[70] = 13.5043419*tk1*(1-tklog) + -8.9323484e-03*tk2 + 
          9.988406183333333e-07*tk3 + -7.655980341666664e-11*tk4 + 
          2.642176509999999e-15*tk5 + (-2.58255688e+04 - tk1*-44.4645715); 
      }
      else
      {
        cgspl[70] = 1.69099899*tk1*(1-tklog) + -0.02135844455*tk2 + 
          4.154694916666667e-06*tk3 + -5.841346016666665e-10*tk4 + 
          3.616314139999999e-14*tk5 + (-2.14512334e+04 - tk1*19.9500833); 
      }
    }
    // Species IC4H6OH
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[71] = 14.0310926*tk1*(1-tklog) + -7.76587705e-03*tk2 + 
          8.879249350000001e-07*tk3 + -6.906549183333331e-11*tk4 + 
          2.407726284999999e-15*tk5 + (-7.69378228e+03 - tk1*-47.6555306); 
      }
      else
      {
        cgspl[71] = 0.863371227*tk1*(1-tklog) + -0.0234355641*tk2 + 
          5.726338983333334e-06*tk3 + -1.1085921e-09*tk4 + 1.069574875e-13*tk5 + 
          (-3.14948305e+03 - tk1*22.9075523); 
      }
    }
    // Species IC3H5CHO
    {
      if (tk1 > 1.396e+03)
      {
        cgspl[72] = 13.6203958*tk1*(1-tklog) + -6.8958596e-03*tk2 + 
          7.889501966666667e-07*tk3 + -6.138793549999998e-11*tk4 + 
          2.10048987e-15*tk5 + (-2.00025274e+04 - tk1*-47.3184531); 
      }
      else
      {
        cgspl[72] = 0.627183793*tk1*(1-tklog) + -0.0233390127*tk2 + 
          6.240510516666667e-06*tk3 + -1.319421183333333e-09*tk4 + 
          1.369760775e-13*tk5 + (-1.57203117e+04 - tk1*21.6034294); 
      }
    }
    // Species IC3H5CO
    {
      if (tk1 > 1.397e+03)
      {
        cgspl[73] = 13.0667437*tk1*(1-tklog) + -5.8352122e-03*tk2 + 
          6.651775383333335e-07*tk3 + -5.162484566666666e-11*tk4 + 
          1.796741244999999e-15*tk5 + (-3.36519344e+03 - tk1*-43.580309); 
      }
      else
      {
        cgspl[73] = 1.85097069*tk1*(1-tklog) + -0.0209427923*tk2 + 
          6.042562183333335e-06*tk3 + -1.380755491666666e-09*tk4 + 
          1.52925423e-13*tk5 + (170.381441 - tk1*15.3014433); 
      }
    }
    // Species TC3H6OCHO
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[74] = 17.0371287*tk1*(1-tklog) + -7.72003225e-03*tk2 + 
          8.805548100000001e-07*tk3 + -6.842377891666666e-11*tk4 + 
          2.384492144999999e-15*tk5 + (-2.75871941e+04 - tk1*-63.727123); 
      }
      else
      {
        cgspl[74] = 0.370830259*tk1*(1-tklog) + -0.02692378305*tk2 + 
          6.374626083333334e-06*tk3 + -1.107351975e-09*tk4 + 
          8.961436499999999e-14*tk5 + (-2.18391262e+04 - tk1*25.8142112); 
      }
    }
    // Species IC3H6CO
    {
      if (tk1 > 1.397e+03)
      {
        cgspl[75] = 13.2548232*tk1*(1-tklog) + -7.00713935e-03*tk2 + 
          7.981836916666668e-07*tk3 + -6.191036183333333e-11*tk4 + 
          2.153687829999999e-15*tk5 + (-2.00529779e+04 - tk1*-44.4810221); 
      }
      else
      {
        cgspl[75] = 2.28039055*tk1*(1-tklog) + -0.02085084945*tk2 + 
          5.418161016666668e-06*tk3 + -1.143695158333333e-09*tk4 + 
          1.20286566e-13*tk5 + (-1.63939712e+04 - tk1*13.8187714); 
      }
    }
    // Species IC4H7OOH
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[76] = 16.9234564*tk1*(1-tklog) + -8.91983845e-03*tk2 + 
          1.023788798333334e-06*tk3 + -7.982458566666666e-11*tk4 + 
          2.78719152e-15*tk5 + (-2.00040686e+04 - tk1*-59.474607); 
      }
      else
      {
        cgspl[76] = 2.99117402*tk1*(1-tklog) + -0.0251674639*tk2 + 
          5.938001016666667e-06*tk3 + -1.11626795e-09*tk4 + 1.055267045e-13*tk5 
          + (-1.51095046e+04 - tk1*15.4537413); 
      }
    }
    // Species TC3H6O2CHO
    {
      if (tk1 > 1.386e+03)
      {
        cgspl[77] = 18.5534443*tk1*(1-tklog) + -8.43871945e-03*tk2 + 
          9.845882750000002e-07*tk3 + -7.762650708333332e-11*tk4 + 
          2.731725934999999e-15*tk5 + (-2.85447191e+04 - tk1*-68.2486667); 
      }
      else
      {
        cgspl[77] = 2.17883383*tk1*(1-tklog) + -0.0270797916*tk2 + 
          6.390598100000001e-06*tk3 + -1.152567533333333e-09*tk4 + 
          1.020950735e-13*tk5 + (-2.27394154e+04 - tk1*20.0751264); 
      }
    }
    // Species CH2CCH2OH
    {
      if (tk1 > 1.372e+03)
      {
        cgspl[78] = 9.707020269999999*tk1*(1-tklog) + -5.698633e-03*tk2 + 
          6.299899366666667e-07*tk3 + -4.793410641666666e-11*tk4 + 
          1.646145625e-15*tk5 + (9.132128839999999e+03 - tk1*-22.5012933); 
      }
      else
      {
        cgspl[78] = 2.88422544*tk1*(1-tklog) + -0.01212140355*tk2 + 
          1.9025378e-06*tk3 + -1.431461116666666e-10*tk4 + 
          -7.108872699999999e-15*tk5 + (1.17935615e+04 - tk1*15.2102335); 
      }
    }
    // Species BC5H11
    {
      if (tk1 > 1.378e+03)
      {
        cgspl[79] = 13.8978149*tk1*(1-tklog) + -0.01274819545*tk2 + 
          1.463553595e-06*tk3 + -1.141204091666666e-10*tk4 + 
          3.984645104999999e-15*tk5 + (-3.85892889e+03 - tk1*-47.3886894); 
      }
      else
      {
        cgspl[79] = 2.35820469*tk1*(1-tklog) + -0.0215377222*tk2 + 
          2.383591483333333e-06*tk3 + 8.700975249999998e-11*tk4 + 
          -5.305232949999999e-14*tk5 + (1.20929341e+03 - tk1*18.1181461); 
      }
    }
    // Species AC5H10
    {
      if (tk1 > 1.391e+03)
      {
        cgspl[80] = 14.5614087*tk1*(1-tklog) + -0.0112685269*tk2 + 
          1.2835001e-06*tk3 + -9.955655833333332e-11*tk4 + 3.46352404e-15*tk5 + 
          (-1.1800452e+04 - tk1*-53.0651959); 
      }
      else
      {
        cgspl[80] = -0.276630603*tk1*(1-tklog) + -0.027676964*tk2 + 
          5.908327e-06*tk3 + -9.968904499999998e-10*tk4 + 
          8.493912099999998e-14*tk5 + (-6.39547297e+03 - tk1*27.3911578); 
      }
    }
    // Species BC5H10
    {
      if (tk1 > 1.385e+03)
      {
        cgspl[81] = 13.9945443*tk1*(1-tklog) + -0.0114723291*tk2 + 
          1.304323588333333e-06*tk3 + -1.010501916666666e-10*tk4 + 
          3.512592284999999e-15*tk5 + (-1.31576943e+04 - tk1*-50.4083848); 
      }
      else
      {
        cgspl[81] = 1.20274147*tk1*(1-tklog) + -0.02394360535*tk2 + 
          4.135746100000001e-06*tk3 + -4.779286291666665e-10*tk4 + 
          1.956333425e-14*tk5 + (-8.15212335e+03 - tk1*20.1592579); 
      }
    }
    // Species CC5H10
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[82] = 14.6491909*tk1*(1-tklog) + -0.01127110305*tk2 + 
          1.286755001666667e-06*tk3 + -9.99655033333333e-11*tk4 + 
          3.481588724999999e-15*tk5 + (-1.11652332e+04 - tk1*-54.2252285); 
      }
      else
      {
        cgspl[82] = -1.83685605*tk1*(1-tklog) + -0.0303854733*tk2 + 
          7.026542933333334e-06*tk3 + -1.300866433333333e-09*tk4 + 
          1.21571383e-13*tk5 + (-5.34864282e+03 - tk1*34.5203543); 
      }
    }
    // Species AC5H9-C
    {
      if (tk1 > 1.391e+03)
      {
        cgspl[83] = 14.1589519*tk1*(1-tklog) + -0.0104238078*tk2 + 
          1.189871558333333e-06*tk3 + -9.243556916666664e-11*tk4 + 
          3.219347354999999e-15*tk5 + (5.2956702e+03 - tk1*-52.1277925); 
      }
      else
      {
        cgspl[83] = -0.580051463*tk1*(1-tklog) + -0.0268157419*tk2 + 
          5.82159235e-06*tk3 + -9.887379916666663e-10*tk4 + 
          8.388876299999997e-14*tk5 + (1.06267222e+04 - tk1*27.693703); 
      }
    }
    // Species CC5H9-B
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[84] = 13.7488177*tk1*(1-tklog) + -0.0106484334*tk2 + 
          1.219445175e-06*tk3 + -9.49397583333333e-11*tk4 + 
          3.311614459999999e-15*tk5 + (4.95421077e+03 - tk1*-50.4875121); 
      }
      else
      {
        cgspl[84] = -1.34298383*tk1*(1-tklog) + -0.02700409335*tk2 + 
          5.6542683e-06*tk3 + -9.033629583333331e-10*tk4 + 
          7.053734349999998e-14*tk5 + (1.04932842e+04 - tk1*31.5398815); 
      }
    }
    // Species AC5H9O-C
    {
      if (tk1 > 1.395e+03)
      {
        cgspl[85] = 18.5587275*tk1*(1-tklog) + -9.20333345e-03*tk2 + 
          1.022502243333333e-06*tk3 + -7.809719491666666e-11*tk4 + 
          2.689858504999999e-15*tk5 + (-6.48338717e+03 - tk1*-71.2211423); 
      }
      else
      {
        cgspl[85] = -1.71420068*tk1*(1-tklog) + -0.03453987335*tk2 + 
          9.118300550000001e-06*tk3 + -1.842190908333333e-09*tk4 + 
          1.79033907e-13*tk5 + (131.695143 - tk1*36.3260717); 
      }
    }
    // Species CC5H9O-B
    {
      if (tk1 > 1.462e+03)
      {
        cgspl[86] = 21.6481795*tk1*(1-tklog) + -9.020308649999999e-03*tk2 + 
          1.083637896666667e-06*tk3 + -8.711142666666665e-11*tk4 + 
          3.107179989999999e-15*tk5 + (-9.811130590000001e+03 - 
          tk1*-70.7098941); 
      }
      else
      {
        cgspl[86] = 3.1937726*tk1*(1-tklog) + -0.02183600245*tk2 + 
          1.16392922e-06*tk3 + 8.743297583333331e-10*tk4 + 
          -1.945908009999999e-13*tk5 + (-1.8262988e+03 - tk1*34.3211965); 
      }
    }
    // Species CH3CHCHO
    {
      if (tk1 > 1.253e+03)
      {
        cgspl[87] = 8.2777209*tk1*(1-tklog) + -9.7843594e-03*tk2 + 
          1.412625148333334e-06*tk3 + -1.277081066666666e-10*tk4 + 
          4.932207709999999e-15*tk5 + (-8.34456062e+03 - tk1*-21.7652483); 
      }
      else
      {
        cgspl[87] = -2.72811212*tk1*(1-tklog) + -0.01901772415*tk2 + 
          3.2797022e-06*tk3 + -4.246288083333332e-10*tk4 + 
          3.643945459999999e-14*tk5 + (-3.51122739e+03 - tk1*40.4496677); 
      }
    }
    // Species BC6H12
    {
      if (tk1 > 1.389e+03)
      {
        cgspl[88] = 17.3194968*tk1*(1-tklog) + -0.01365717275*tk2 + 
          1.554709006666667e-06*tk3 + -1.205530825e-10*tk4 + 
          4.193096209999999e-15*tk5 + (-1.72536853e+04 - 
          tk1*-66.77084019999999); 
      }
      else
      {
        cgspl[88] = 6.61236759e-03*tk1*(1-tklog) + -0.03205083235*tk2 + 
          6.417533616666668e-06*tk3 + -9.799851499999997e-10*tk4 + 
          7.310914649999999e-14*tk5 + (-1.0802194e+04 - tk1*27.6294362); 
      }
    }
    // Species CC6H12
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[89] = 17.2953713*tk1*(1-tklog) + -0.0137452233*tk2 + 
          1.570356453333334e-06*tk3 + -1.220535241666667e-10*tk4 + 
          4.252137859999999e-15*tk5 + (-1.65683634e+04 - tk1*-67.2858572); 
      }
      else
      {
        cgspl[89] = -1.31890347*tk1*(1-tklog) + -0.03429872595*tk2 + 
          7.331361233333335e-06*tk3 + -1.227432541666666e-09*tk4 + 
          1.028972525e-13*tk5 + (-9.793955250000001e+03 - tk1*33.6472333); 
      }
    }
    // Species C5H10-2
    {
      if (tk1 > 1.389e+03)
      {
        cgspl[90] = 14.1109267*tk1*(1-tklog) + -0.0114174136*tk2 + 
          1.297711391666667e-06*tk3 + -1.005229091666666e-10*tk4 + 
          3.493979914999999e-15*tk5 + (-1.14336507e+04 - tk1*-50.1601163); 
      }
      else
      {
        cgspl[90] = -0.541560551*tk1*(1-tklog) + -0.0269814959*tk2 + 
          5.391812300000001e-06*tk3 + -8.145133641666665e-10*tk4 + 
          5.926733399999998e-14*tk5 + (-5.98606169e+03 - tk1*29.7142748); 
      }
    }
    // Species IC4H7-I1
    {
      if (tk1 > 1.389e+03)
      {
        cgspl[91] = 11.09576*tk1*(1-tklog) + -7.89050245e-03*tk2 + 
          8.973660250000001e-07*tk3 + -6.953001008333331e-11*tk4 + 
          2.417007404999999e-15*tk5 + (2.24175827e+04 - tk1*-34.0426822); 
      }
      else
      {
        cgspl[91] = 1.87632434*tk1*(1-tklog) + -0.01772430535*tk2 + 
          3.55174915e-06*tk3 + -5.611670766666665e-10*tk4 + 
          4.515831694999999e-14*tk5 + (2.58712914e+04 - tk1*16.2429161); 
      }
    }
    // Species YC7H15
    {
      if (tk1 > 1.384e+03)
      {
        cgspl[92] = 20.4581471*tk1*(1-tklog) + -0.01715381805*tk2 + 
          1.96837725e-06*tk3 + -1.534413425e-10*tk4 + 5.356690299999999e-15*tk5 
          + (-1.34978774e+04 - tk1*-81.8212264); 
      }
      else
      {
        cgspl[92] = 1.30897106*tk1*(1-tklog) + -0.0348068221*tk2 + 
          5.519167616666667e-06*tk3 + -4.857402133333333e-10*tk4 + 
          -1.772136569999999e-15*tk5 + (-5.78512513e+03 - tk1*24.5658235); 
      }
    }
    // Species XC7H14
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[93] = 21.016403*tk1*(1-tklog) + -0.01576072985*tk2 + 
          1.80123245e-06*tk3 + -1.400334925e-10*tk4 + 4.879461564999999e-15*tk5 
          + (-2.1211705e+04 - tk1*-86.3818785); 
      }
      else
      {
        cgspl[93] = -1.33081497*tk1*(1-tklog) + -0.0410541176*tk2 + 
          9.152055050000001e-06*tk3 + -1.619041233333333e-09*tk4 + 
          1.44443271e-13*tk5 + (-1.32021384e+04 - tk1*34.3549746); 
      }
    }
    // Species YC7H14
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[94] = 20.5074323*tk1*(1-tklog) + -0.01598220835*tk2 + 
          1.826979633333334e-06*tk3 + -1.420531133333333e-10*tk4 + 
          4.950180599999999e-15*tk5 + (-2.23945609e+04 - tk1*-83.9266561); 
      }
      else
      {
        cgspl[94] = -0.8422326490000001*tk1*(1-tklog) + -0.03948991485*tk2 + 
          8.409574583333334e-06*tk3 + -1.407783733333333e-09*tk4 + 
          1.186009935e-13*tk5 + (-1.45971538e+04 - tk1*31.9096189); 
      }
    }
    // Species XC7H13-Z
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[95] = 20.6194069*tk1*(1-tklog) + -0.0149139895*tk2 + 
          1.707394733333334e-06*tk3 + -1.328974391666666e-10*tk4 + 
          4.634790189999999e-15*tk5 + (-4.13779765e+03 - 
          tk1*-85.47591009999999); 
      }
      else
      {
        cgspl[95] = -1.64635315*tk1*(1-tklog) + -0.04022549745*tk2 + 
          9.081524216666667e-06*tk3 + -1.61583975e-09*tk4 + 1.44041191e-13*tk5 + 
          (3.80138821e+03 - tk1*34.7109409); 
      }
    }
    // Species YC7H13-Y2
    {
      if (tk1 > 1.389e+03)
      {
        cgspl[96] = 19.6153468*tk1*(1-tklog) + -0.01535370525*tk2 + 
          1.758841133333334e-06*tk3 + -1.369549658333333e-10*tk4 + 
          4.777498639999999e-15*tk5 + (-6.29650914e+03 - tk1*-80.9254914); 
      }
      else
      {
        cgspl[96] = -0.306783292*tk1*(1-tklog) + -0.03603455725*tk2 + 
          7.003898600000001e-06*tk3 + -1.0004174e-09*tk4 + 
          6.630260699999998e-14*tk5 + (1.21727364e+03 - tk1*28.0339431); 
      }
    }
    // Species YC7H13O-Y2
    {
      if (tk1 > 1.391e+03)
      {
        cgspl[97] = 24.489557*tk1*(1-tklog) + -0.0136249639*tk2 + 
          1.51146668e-06*tk3 + -1.152898975e-10*tk4 + 3.966495684999999e-15*tk5 
          + (-1.87690928e+04 - tk1*-102.028362); 
      }
      else
      {
        cgspl[97] = -0.727101657*tk1*(1-tklog) + -0.0443568192*tk2 + 
          1.110806368333333e-05*tk3 + -2.167884383333333e-09*tk4 + 
          2.064706284999999e-13*tk5 + (-1.03333355e+04 - tk1*32.3860863); 
      }
    }
    // Species YC7H15O2
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[98] = 26.3777368*tk1*(1-tklog) + -0.0170497967*tk2 + 
          1.9619758e-06*tk3 + -1.532348333333333e-10*tk4 + 
          5.356660249999999e-15*tk5 + (-3.39133584e+04 - tk1*-111.594572); 
      }
      else
      {
        cgspl[98] = -0.176194227*tk1*(1-tklog) + -0.0486283817*tk2 + 
          1.184726306666667e-05*tk3 + -2.337391383333333e-09*tk4 + 
          2.334389515e-13*tk5 + (-2.46466146e+04 - tk1*30.8813371); 
      }
    }
    // Species ACC6H10
    {
      if (tk1 > 1.395e+03)
      {
        cgspl[99] = 17.1179863*tk1*(1-tklog) + -0.01144488185*tk2 + 
          1.302817396666667e-06*tk3 + -1.010111041666666e-10*tk4 + 
          3.512993704999999e-15*tk5 + (-3.26888037e+03 - tk1*-66.6240298); 
      }
      else
      {
        cgspl[99] = -0.762523956*tk1*(1-tklog) + -0.0338082029*tk2 + 
          8.694858716666668e-06*tk3 + -1.816558908333333e-09*tk4 + 
          1.89530755e-13*tk5 + (2.72970447e+03 - tk1*28.4975422); 
      }
    }
    // Species ACC6H9-A
    {
      if (tk1 > 1.398e+03)
      {
        cgspl[100] = 17.5340261*tk1*(1-tklog) + -0.0102380972*tk2 + 
          1.166393333333334e-06*tk3 + -9.048562499999998e-11*tk4 + 
          3.148240374999999e-15*tk5 + (1.49755134e+04 - tk1*-69.03166589999999); 
      }
      else
      {
        cgspl[100] = -1.7084984*tk1*(1-tklog) + -0.0361216337*tk2 + 
          1.03691751e-05*tk3 + -2.363327158333333e-09*tk4 + 2.61435174e-13*tk5 + 
          (2.10506573e+04 - tk1*32.0184824); 
      }
    }
    // Species ACC6H9-D
    {
      if (tk1 > 1.398e+03)
      {
        cgspl[101] = 17.5340261*tk1*(1-tklog) + -0.0102380972*tk2 + 
          1.166393333333334e-06*tk3 + -9.048562499999998e-11*tk4 + 
          3.148240374999999e-15*tk5 + (1.49755134e+04 - tk1*-69.03166589999999); 
      }
      else
      {
        cgspl[101] = -1.7084984*tk1*(1-tklog) + -0.0361216337*tk2 + 
          1.03691751e-05*tk3 + -2.363327158333333e-09*tk4 + 2.61435174e-13*tk5 + 
          (2.10506573e+04 - tk1*32.0184824); 
      }
    }
    // Species NEOC5H11
    {
      if (tk1 > 1.449e+03)
      {
        cgspl[102] = 20.3101659*tk1*(1-tklog) + -0.01073233485*tk2 + 
          1.286876271666667e-06*tk3 + -1.033189016666666e-10*tk4 + 
          3.68203782e-15*tk5 + (-6.45138607e+03 - tk1*-90.1108572); 
      }
      else
      {
        cgspl[102] = 3.39222761*tk1*(1-tklog) + -0.01770345445*tk2 + 
          -2.000334666666667e-06*tk3 + 1.950938624999999e-09*tk4 + 
          -3.399979685e-13*tk5 + (1.81088321e+03 - tk1*9.54432517); 
      }
    }
    // Species NEOC5H11O2
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[103] = 20.5483317*tk1*(1-tklog) + -0.0123152325*tk2 + 
          1.41819798e-06*tk3 + -1.1082427e-10*tk4 + 3.875658404999999e-15*tk5 + 
          (-2.32426335e+04 - tk1*-83.2803718); 
      }
      else
      {
        cgspl[103] = -0.391775516*tk1*(1-tklog) + -0.0376150014*tk2 + 
          9.429792416666668e-06*tk3 + -1.8893738e-09*tk4 + 1.89222685e-13*tk5 + 
          (-1.60564398e+04 - tk1*28.7221556); 
      }
    }
    // Species NEOC5H10OOH
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[104] = 22.5938496*tk1*(1-tklog) + -0.01132677795*tk2 + 
          1.311740655e-06*tk3 + -1.029053841666667e-10*tk4 + 3.608747355e-15*tk5 
          + (-1.60940953e+04 - tk1*-91.3887696); 
      }
      else
      {
        cgspl[104] = -0.605220938*tk1*(1-tklog) + -0.0397956288*tk2 + 
          1.039609521666667e-05*tk3 + -2.115783016666666e-09*tk4 + 
          2.11290552e-13*tk5 + (-8.28955104e+03 - tk1*32.2696177); 
      }
    }
    // Species TC4H9CHO
    {
      if (tk1 > 1.396e+03)
      {
        cgspl[105] = 18.4056359*tk1*(1-tklog) + -0.01076412115*tk2 + 
          1.23109255e-06*tk3 + -9.578499916666663e-11*tk4 + 
          3.340018724999999e-15*tk5 + (-3.84092887e+04 - tk1*-74.9886911); 
      }
      else
      {
        cgspl[105] = -1.75588233*tk1*(1-tklog) + -0.0344370608*tk2 + 
          8.286017850000002e-06*tk3 + -1.524347083333333e-09*tk4 + 
          1.36028467e-13*tk5 + (-3.14740629e+04 - tk1*33.133448); 
      }
    }
    // Species TC4H9CO
    {
      if (tk1 > 1.397e+03)
      {
        cgspl[106] = 17.9864364*tk1*(1-tklog) + -9.7007182e-03*tk2 + 
          1.111193116666667e-06*tk3 + -8.654970499999999e-11*tk4 + 
          3.020322859999999e-15*tk5 + (-1.94463168e+04 - tk1*-71.0669249); 
      }
      else
      {
        cgspl[106] = -0.977833363*tk1*(1-tklog) + -0.0320286742*tk2 + 
          7.773481133333334e-06*tk3 + -1.43452235e-09*tk4 + 1.278406525e-13*tk5 
          + (-1.29463546e+04 - tk1*30.5727393); 
      }
    }
    // Species IC8H18
    {
      if (tk1 > 1.396e+03)
      {
        cgspl[107] = 27.137359*tk1*(1-tklog) + -0.0189502445*tk2 + 
          2.1572893e-06*tk3 + -1.6730031e-10*tk4 + 5.820028999999999e-15*tk5 + 
          (-4.07958177e+04 - tk1*-123.277495); 
      }
      else
      {
        cgspl[107] = -4.20868893*tk1*(1-tklog) + -0.0557202905*tk2 + 
          1.31891097e-05*tk3 + -2.436718683333333e-09*tk4 + 
          2.218715954999999e-13*tk5 + (-2.99446875e+04 - tk1*44.9521701); 
      }
    }
    // Species AC8H17
    {
      if (tk1 > 1.396e+03)
      {
        cgspl[108] = 26.7069782*tk1*(1-tklog) + -0.01788303005*tk2 + 
          2.036311733333334e-06*tk3 + -1.579471916666666e-10*tk4 + 
          5.495392149999998e-15*tk5 + (-1.57229692e+04 - tk1*-117.001113); 
      }
      else
      {
        cgspl[108] = -3.41944741*tk1*(1-tklog) + -0.0534015945*tk2 + 
          1.275685938333333e-05*tk3 + -2.37784735e-09*tk4 + 
          2.182393244999999e-13*tk5 + (-5.33514196e+03 - tk1*44.5471727); 
      }
    }
    // Species BC8H17
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[109] = 26.4569179*tk1*(1-tklog) + -0.0177710376*tk2 + 
          2.008683066666667e-06*tk3 + -1.550744641666666e-10*tk4 + 
          5.378594699999999e-15*tk5 + (-1.70392791e+04 - tk1*-116.245511); 
      }
      else
      {
        cgspl[109] = -3.09104262*tk1*(1-tklog) + -0.051159448*tk2 + 
          1.141431455e-05*tk3 + -1.9181995e-09*tk4 + 1.5350654e-13*tk5 + 
          (-6.62829069e+03 - tk1*43.1173932); 
      }
    }
    // Species CC8H17
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[110] = 25.1497158*tk1*(1-tklog) + -0.01855484225*tk2 + 
          2.114241383333334e-06*tk3 + -1.640611908333333e-10*tk4 + 
          5.709694699999999e-15*tk5 + (-1.8276179e+04 - tk1*-109.056834); 
      }
      else
      {
        cgspl[110] = -0.0973159697*tk1*(1-tklog) + -0.0446326862*tk2 + 
          8.547896900000002e-06*tk3 + -1.1470044e-09*tk4 + 
          6.389419799999998e-14*tk5 + (-8.811473019999999e+03 - tk1*28.9791898); 
      }
    }
    // Species DC8H17
    {
      if (tk1 > 1.396e+03)
      {
        cgspl[111] = 26.7069782*tk1*(1-tklog) + -0.01788303005*tk2 + 
          2.036311733333334e-06*tk3 + -1.579471916666666e-10*tk4 + 
          5.495392149999998e-15*tk5 + (-1.61255862e+04 - tk1*-118.098245); 
      }
      else
      {
        cgspl[111] = -3.41944741*tk1*(1-tklog) + -0.0534015945*tk2 + 
          1.275685938333333e-05*tk3 + -2.37784735e-09*tk4 + 
          2.182393244999999e-13*tk5 + (-5.73775897e+03 - tk1*43.4500414); 
      }
    }
    // Species IC8H16
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[112] = 25.6756746*tk1*(1-tklog) + -0.0170900999*tk2 + 
          1.933382533333333e-06*tk3 + -1.49329565e-10*tk4 + 
          5.180650099999999e-15*tk5 + (-2.62458324e+04 - tk1*-113.928273); 
      }
      else
      {
        cgspl[112] = -2.79610447*tk1*(1-tklog) + -0.050418086*tk2 + 
          1.187084418333333e-05*tk3 + -2.1721652e-09*tk4 + 
          1.950159069999999e-13*tk5 + (-1.64002496e+04 - tk1*38.8854068); 
      }
    }
    // Species JC8H16
    {
      if (tk1 > 1.397e+03)
      {
        cgspl[113] = 26.0101527*tk1*(1-tklog) + -0.01698516215*tk2 + 
          1.923707516666667e-06*tk3 + -1.486856783333333e-10*tk4 + 
          5.160609999999999e-15*tk5 + (-2.65174535e+04 - tk1*-115.359195); 
      }
      else
      {
        cgspl[113] = -3.31862122*tk1*(1-tklog) + -0.0521576525*tk2 + 
          1.274605021666667e-05*tk3 + -2.434743491666666e-09*tk4 + 
          2.284569309999999e-13*tk5 + (-1.6544825e+04 - tk1*41.4548253); 
      }
    }
    // Species AC8H17O2
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[114] = 30.2815958*tk1*(1-tklog) + -0.0189364536*tk2 + 
          2.17447305e-06*tk3 + -1.696035108333333e-10*tk4 + 
          5.923563349999999e-15*tk5 + (-3.48032462e+04 - tk1*-131.647036); 
      }
      else
      {
        cgspl[114] = -1.78614072*tk1*(1-tklog) + -0.0573774515*tk2 + 
          1.417674531666667e-05*tk3 + -2.783674458333333e-09*tk4 + 
          2.72586614e-13*tk5 + (-2.37742056e+04 - tk1*40.0275514); 
      }
    }
    // Species BC8H17O2
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[115] = 30.9351615*tk1*(1-tklog) + -0.0187051282*tk2 + 
          2.151182833333334e-06*tk3 + -1.679540416666666e-10*tk4 + 
          5.869945549999999e-15*tk5 + (-3.77457753e+04 - tk1*-136.7308); 
      }
      else
      {
        cgspl[115] = -3.07002356*tk1*(1-tklog) + -0.061320219*tk2 + 
          1.62033794e-05*tk3 + -3.409396875e-09*tk4 + 3.545512734999999e-13*tk5 
          + (-2.64014308e+04 - tk1*44.0345691); 
      }
    }
    // Species CC8H17O2
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[116] = 30.9721695*tk1*(1-tklog) + -0.0186683041*tk2 + 
          2.145402983333333e-06*tk3 + -1.674228866666666e-10*tk4 + 
          5.849453599999999e-15*tk5 + (-3.79648855e+04 - tk1*-138.456446); 
      }
      else
      {
        cgspl[116] = -1.80143766*tk1*(1-tklog) + -0.0590152555*tk2 + 
          1.517718276666667e-05*tk3 + -3.112336341666666e-09*tk4 + 
          3.169996535e-13*tk5 + (-2.68892049e+04 - tk1*36.2716425); 
      }
    }
    // Species DC8H17O2
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[117] = 30.2815958*tk1*(1-tklog) + -0.0189364536*tk2 + 
          2.17447305e-06*tk3 + -1.696035108333333e-10*tk4 + 
          5.923563349999999e-15*tk5 + (-3.48032462e+04 - tk1*-132.744167); 
      }
      else
      {
        cgspl[117] = -1.78614072*tk1*(1-tklog) + -0.0573774515*tk2 + 
          1.417674531666667e-05*tk3 + -2.783674458333333e-09*tk4 + 
          2.72586614e-13*tk5 + (-2.37742056e+04 - tk1*38.93042); 
      }
    }
    // Species CC8H17O2H
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[118] = 33.4492663*tk1*(1-tklog) + -0.01873804295*tk2 + 
          2.158254816666667e-06*tk3 + -1.686939316666666e-10*tk4 + 
          5.900675999999999e-15*tk5 + (-5.58859437e+04 - tk1*-152.839441); 
      }
      else
      {
        cgspl[118] = -2.84505394*tk1*(1-tklog) + -0.063600014*tk2 + 
          1.661708068333334e-05*tk3 + -3.409283933333333e-09*tk4 + 
          3.440749254999999e-13*tk5 + (-4.37253252e+04 - tk1*40.4161993); 
      }
    }
    // Species CC8H17O
    {
      if (tk1 > 1.398e+03)
      {
        cgspl[119] = 29.4459429*tk1*(1-tklog) + -0.01797082155*tk2 + 
          2.0447894e-06*tk3 + -1.585279941666666e-10*tk4 + 
          5.513783749999999e-15*tk5 + (-3.65164717e+04 - tk1*-132.7047); 
      }
      else
      {
        cgspl[119] = -3.83187637*tk1*(1-tklog) + -0.059290777*tk2 + 
          1.534816246666667e-05*tk3 + -3.119293608333333e-09*tk4 + 
          3.108617094999999e-13*tk5 + (-2.54590182e+04 - tk1*44.2631132); 
      }
    }
    // Species AC8H16OOH-A
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[120] = 32.2733666*tk1*(1-tklog) + -0.0180017957*tk2 + 
          2.076410716666667e-06*tk3 + -1.6245188e-10*tk4 + 
          5.686120949999998e-15*tk5 + (-2.80443391e+04 - tk1*-140.571052); 
      }
      else
      {
        cgspl[120] = -1.84900727*tk1*(1-tklog) + -0.059235188*tk2 + 
          1.499348545e-05*tk3 + -2.969030616666666e-09*tk4 + 
          2.898715754999999e-13*tk5 + (-1.64331259e+04 - tk1*41.7717295); 
      }
    }
    // Species AC8H16OOH-B
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[121] = 32.5510943*tk1*(1-tklog) + -0.0173993366*tk2 + 
          1.9740765e-06*tk3 + -1.52846565e-10*tk4 + 5.313162749999998e-15*tk5 + 
          (-2.9511989e+04 - tk1*-141.56826); 
      }
      else
      {
        cgspl[121] = -2.0691273*tk1*(1-tklog) + -0.0580710695*tk2 + 
          1.406711211666667e-05*tk3 + -2.589078324999999e-09*tk4 + 
          2.299401515e-13*tk5 + (-1.76383455e+04 - tk1*44.0292848); 
      }
    }
    // Species AC8H16OOH-C
    {
      if (tk1 > 1.389e+03)
      {
        cgspl[122] = 30.7262781*tk1*(1-tklog) + -0.0186639326*tk2 + 
          2.1528514e-06*tk3 + -1.684307683333333e-10*tk4 + 
          5.895257149999998e-15*tk5 + (-2.98447203e+04 - tk1*-131.584073); 
      }
      else
      {
        cgspl[122] = 1.37822561*tk1*(1-tklog) + -0.050681773*tk2 + 
          1.089171655e-05*tk3 + -1.770723458333333e-09*tk4 + 1.397263245e-13*tk5 
          + (-1.91404617e+04 - tk1*27.738376); 
      }
    }
    // Species BC8H16OOH-A
    {
      if (tk1 > 1.395e+03)
      {
        cgspl[123] = 32.9649795*tk1*(1-tklog) + -0.0177260763*tk2 + 
          2.046011616666667e-06*tk3 + -1.601463349999999e-10*tk4 + 
          5.607183449999999e-15*tk5 + (-3.05966248e+04 - tk1*-144.760131); 
      }
      else
      {
        cgspl[123] = -3.22207871*tk1*(1-tklog) + -0.0633178175*tk2 + 
          1.707111816666667e-05*tk3 + -3.604647616666666e-09*tk4 + 
          3.724950484999999e-13*tk5 + (-1.86411004e+04 - tk1*47.3182493); 
      }
    }
    // Species BC8H16OOH-D
    {
      if (tk1 > 1.395e+03)
      {
        cgspl[124] = 32.9649795*tk1*(1-tklog) + -0.0177260763*tk2 + 
          2.046011616666667e-06*tk3 + -1.601463349999999e-10*tk4 + 
          5.607183449999999e-15*tk5 + (-3.05966248e+04 - tk1*-145.857263); 
      }
      else
      {
        cgspl[124] = -3.22207871*tk1*(1-tklog) + -0.0633178175*tk2 + 
          1.707111816666667e-05*tk3 + -3.604647616666666e-09*tk4 + 
          3.724950484999999e-13*tk5 + (-1.86411004e+04 - tk1*46.2211179); 
      }
    }
    // Species CC8H16OOH-A
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[125] = 33.0252762*tk1*(1-tklog) + -0.01767209265*tk2 + 
          2.037728583333334e-06*tk3 + -1.593924558333333e-10*tk4 + 
          5.578250849999999e-15*tk5 + (-3.08189254e+04 - tk1*-146.607038); 
      }
      else
      {
        cgspl[125] = -2.0575127*tk1*(1-tklog) + -0.061282901*tk2 + 
          1.61840654e-05*tk3 + -3.350385116666666e-09*tk4 + 
          3.405014754999999e-13*tk5 + (-1.91154529e+04 - tk1*40.0197465); 
      }
    }
    // Species DC8H16OOH-C
    {
      if (tk1 > 1.389e+03)
      {
        cgspl[126] = 30.7262781*tk1*(1-tklog) + -0.0186639326*tk2 + 
          2.1528514e-06*tk3 + -1.684307683333333e-10*tk4 + 
          5.895257149999998e-15*tk5 + (-2.94421033e+04 - tk1*-131.98669); 
      }
      else
      {
        cgspl[126] = 1.37822561*tk1*(1-tklog) + -0.050681773*tk2 + 
          1.089171655e-05*tk3 + -1.770723458333333e-09*tk4 + 1.397263245e-13*tk5 
          + (-1.87378447e+04 - tk1*27.3357589); 
      }
    }
    // Species DC8H16OOH-B
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[127] = 32.5510943*tk1*(1-tklog) + -0.0173993366*tk2 + 
          1.9740765e-06*tk3 + -1.52846565e-10*tk4 + 5.313162749999998e-15*tk5 + 
          (-2.9109372e+04 - tk1*-142.665392); 
      }
      else
      {
        cgspl[127] = -2.0691273*tk1*(1-tklog) + -0.0580710695*tk2 + 
          1.406711211666667e-05*tk3 + -2.589078324999999e-09*tk4 + 
          2.299401515e-13*tk5 + (-1.72357285e+04 - tk1*42.9321534); 
      }
    }
    // Species IC8ETERAB
    {
      if (tk1 > 1.403e+03)
      {
        cgspl[128] = 27.6798014*tk1*(1-tklog) + -0.0179661503*tk2 + 
          2.05121095e-06*tk3 + -1.593910066666666e-10*tk4 + 
          5.552712699999998e-15*tk5 + (-4.05912134e+04 - tk1*-125.295695); 
      }
      else
      {
        cgspl[128] = -7.80049041*tk1*(1-tklog) + -0.0617904385*tk2 + 
          1.597702548333334e-05*tk3 + -3.196210516666666e-09*tk4 + 
          3.108807139999999e-13*tk5 + (-2.88197649e+04 - tk1*63.4400053); 
      }
    }
    // Species IC8ETERAC
    {
      if (tk1 > 1.401e+03)
      {
        cgspl[129] = 27.4247596*tk1*(1-tklog) + -0.01819610855*tk2 + 
          2.071525383333334e-06*tk3 + -1.606794025e-10*tk4 + 
          5.590922749999999e-15*tk5 + (-5.27913842e+04 - tk1*-127.246044); 
      }
      else
      {
        cgspl[129] = -9.591872840000001*tk1*(1-tklog) + -0.06411823849999999*tk2 
          + 1.672108066666667e-05*tk3 + -3.365707966666666e-09*tk4 + 
          3.286668625e-13*tk5 + (-4.05657072e+04 - tk1*69.4955343); 
      }
    }
    // Species IC8ETERBD
    {
      if (tk1 > 1.403e+03)
      {
        cgspl[130] = 27.6798014*tk1*(1-tklog) + -0.0179661503*tk2 + 
          2.05121095e-06*tk3 + -1.593910066666666e-10*tk4 + 
          5.552712699999998e-15*tk5 + (-4.05912134e+04 - tk1*-126.392826); 
      }
      else
      {
        cgspl[130] = -7.80049041*tk1*(1-tklog) + -0.0617904385*tk2 + 
          1.597702548333334e-05*tk3 + -3.196210516666666e-09*tk4 + 
          3.108807139999999e-13*tk5 + (-2.88197649e+04 - tk1*62.3428739); 
      }
    }
    // Species AC8H16OOH-BO2
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[131] = 36.8372436*tk1*(1-tklog) + -0.0185370962*tk2 + 
          2.148300533333334e-06*tk3 + -1.686081249999999e-10*tk4 + 
          5.914572799999999e-15*tk5 + (-4.93845688e+04 - tk1*-161.072392); 
      }
      else
      {
        cgspl[131] = -1.29316549*tk1*(1-tklog) + -0.0663812575*tk2 + 
          1.790060466666667e-05*tk3 + -3.788707399999999e-09*tk4 + 
          3.934884399999999e-13*tk5 + (-3.67109267e+04 - tk1*41.5232002); 
      }
    }
    // Species BC8H16OOH-AO2
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[132] = 36.8372436*tk1*(1-tklog) + -0.0185370962*tk2 + 
          2.148300533333334e-06*tk3 + -1.686081249999999e-10*tk4 + 
          5.914572799999999e-15*tk5 + (-4.93845688e+04 - tk1*-161.072392); 
      }
      else
      {
        cgspl[132] = -1.29316549*tk1*(1-tklog) + -0.0663812575*tk2 + 
          1.790060466666667e-05*tk3 + -3.788707399999999e-09*tk4 + 
          3.934884399999999e-13*tk5 + (-3.67109267e+04 - tk1*41.5232002); 
      }
    }
    // Species BC8H16OOH-DO2
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[133] = 36.8372436*tk1*(1-tklog) + -0.0185370962*tk2 + 
          2.148300533333334e-06*tk3 + -1.686081249999999e-10*tk4 + 
          5.914572799999999e-15*tk5 + (-4.93845688e+04 - tk1*-162.169524); 
      }
      else
      {
        cgspl[133] = -1.29316549*tk1*(1-tklog) + -0.0663812575*tk2 + 
          1.790060466666667e-05*tk3 + -3.788707399999999e-09*tk4 + 
          3.934884399999999e-13*tk5 + (-3.67109267e+04 - tk1*40.4260689); 
      }
    }
    // Species CC8H16OOH-AO2
    {
      if (tk1 > 1.392e+03)
      {
        cgspl[134] = 36.8128611*tk1*(1-tklog) + -0.01856514165*tk2 + 
          2.152771533333334e-06*tk3 + -1.690195625e-10*tk4 + 
          5.930417249999999e-15*tk5 + (-4.99876829e+04 - tk1*-162.468861); 
      }
      else
      {
        cgspl[134] = -0.684583036*tk1*(1-tklog) + -0.0657139565*tk2 + 
          1.776473116666667e-05*tk3 + -3.787265658333332e-09*tk4 + 
          3.970653955e-13*tk5 + (-3.7509542e+04 - tk1*36.7518862); 
      }
    }
    // Species DC8H16OOH-BO2
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[135] = 36.8372436*tk1*(1-tklog) + -0.0185370962*tk2 + 
          2.148300533333334e-06*tk3 + -1.686081249999999e-10*tk4 + 
          5.914572799999999e-15*tk5 + (-4.93845688e+04 - tk1*-162.169524); 
      }
      else
      {
        cgspl[135] = -1.29316549*tk1*(1-tklog) + -0.0663812575*tk2 + 
          1.790060466666667e-05*tk3 + -3.788707399999999e-09*tk4 + 
          3.934884399999999e-13*tk5 + (-3.67109267e+04 - tk1*40.4260689); 
      }
    }
    // Species IC8KETAB
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[136] = 34.7134*tk1*(1-tklog) + -0.01709726125*tk2 + 
          1.982545316666667e-06*tk3 + -1.556712466666667e-10*tk4 + 
          5.462793899999999e-15*tk5 + (-6.57745821e+04 - tk1*-153.03586); 
      }
      else
      {
        cgspl[136] = -2.37786754*tk1*(1-tklog) + -0.062499637*tk2 + 
          1.635080563333333e-05*tk3 + -3.296444024999999e-09*tk4 + 
          3.240487264999999e-13*tk5 + (-5.33254847e+04 - tk1*44.6667297); 
      }
    }
    // Species IC8KETBA
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[137] = 33.6880261*tk1*(1-tklog) + -0.01757654075*tk2 + 
          2.0408272e-06*tk3 + -1.603826908333333e-10*tk4 + 
          5.631296499999999e-15*tk5 + (-6.575011900000001e+04 - 
          tk1*-147.550024); 
      }
      else
      {
        cgspl[137] = -3.65722567*tk1*(1-tklog) + -0.0635798385*tk2 + 
          1.678877033333334e-05*tk3 + -3.444732858333333e-09*tk4 + 
          3.461932874999999e-13*tk5 + (-5.32214668e+04 - tk1*51.3886594); 
      }
    }
    // Species IC8KETBD
    {
      if (tk1 > 1.393e+03)
      {
        cgspl[138] = 33.6880261*tk1*(1-tklog) + -0.01757654075*tk2 + 
          2.0408272e-06*tk3 + -1.603826908333333e-10*tk4 + 
          5.631296499999999e-15*tk5 + (-6.575011900000001e+04 - 
          tk1*-147.957674); 
      }
      else
      {
        cgspl[138] = -3.65722567*tk1*(1-tklog) + -0.0635798385*tk2 + 
          1.678877033333334e-05*tk3 + -3.444732858333333e-09*tk4 + 
          3.461932874999999e-13*tk5 + (-5.32214668e+04 - tk1*50.9810097); 
      }
    }
    // Species IC8KETDB
    {
      if (tk1 > 1.395e+03)
      {
        cgspl[139] = 34.7542216*tk1*(1-tklog) + -0.01705678565*tk2 + 
          1.976160216666667e-06*tk3 + -1.550818116666666e-10*tk4 + 
          5.439971849999999e-15*tk5 + (-6.61213882e+04 - tk1*-153.639919); 
      }
      else
      {
        cgspl[139] = -2.64616572*tk1*(1-tklog) + -0.0631770405*tk2 + 
          1.669284683333333e-05*tk3 + -3.398669783333332e-09*tk4 + 
          3.369093945e-13*tk5 + (-5.36421698e+04 - tk1*45.4596528); 
      }
    }
    // Species IC3H7COC3H6-T
    {
      if (tk1 > 1.39e+03)
      {
        cgspl[140] = 23.0231691*tk1*(1-tklog) + -0.0148896768*tk2 + 
          1.715612216666667e-06*tk3 + -1.341186541666666e-10*tk4 + 
          4.691652899999998e-15*tk5 + (-3.05668554e+04 - 
          tk1*-95.57684999999999); 
      }
      else
      {
        cgspl[140] = -0.6604792350000001*tk1*(1-tklog) + -0.0419215085*tk2 + 
          9.655815733333336e-06*tk3 + -1.75197845e-09*tk4 + 1.594801155e-13*tk5 
          + (-2.21318078e+04 - tk1*32.2077356); 
      }
    }
    // Species TC4H9COC2H4S
    {
      if (tk1 > 1.394e+03)
      {
        cgspl[141] = 24.4311775*tk1*(1-tklog) + -0.01428023795*tk2 + 
          1.644985111666667e-06*tk3 + -1.285896308333333e-10*tk4 + 
          4.498308854999999e-15*tk5 + (-3.0261203e+04 - tk1*-104.211609); 
      }
      else
      {
        cgspl[141] = -2.86500441*tk1*(1-tklog) + -0.0472793701*tk2 + 
          1.19730419e-05*tk3 + -2.367465908333333e-09*tk4 + 2.30441994e-13*tk5 + 
          (-2.09863386e+04 - tk1*41.6255786); 
      }
    }
    // Species N2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[142] = 2.92664*tk1*(1-tklog) + -7.439885e-04*tk2 + 
          9.474601666666666e-08*tk3 + -8.414199999999998e-12*tk4 + 
          3.376675499999999e-16*tk5 + (-922.7977 - tk1*5.980528); 
      }
      else
      {
        cgspl[142] = 3.298677*tk1*(1-tklog) + -7.0412e-04*tk2 + 6.60537e-07*tk3 
          + -4.7012625e-10*tk4 + 1.2224275e-13*tk5 + (-1.0209e+03 - 
          tk1*3.950372); 
      }
    }
  }
  
  double mole_frac[143];
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
    mole_frac[35] = mass_frac[35] * recip_molecular_masses[35];
    mole_frac[35] = (mole_frac[35] > 1e-200) ? mole_frac[35] : 1e-200;
    mole_frac[35] *= sumyow;
    mole_frac[36] = mass_frac[36] * recip_molecular_masses[36];
    mole_frac[36] = (mole_frac[36] > 1e-200) ? mole_frac[36] : 1e-200;
    mole_frac[36] *= sumyow;
    mole_frac[37] = mass_frac[37] * recip_molecular_masses[37];
    mole_frac[37] = (mole_frac[37] > 1e-200) ? mole_frac[37] : 1e-200;
    mole_frac[37] *= sumyow;
    mole_frac[38] = mass_frac[38] * recip_molecular_masses[38];
    mole_frac[38] = (mole_frac[38] > 1e-200) ? mole_frac[38] : 1e-200;
    mole_frac[38] *= sumyow;
    mole_frac[39] = mass_frac[39] * recip_molecular_masses[39];
    mole_frac[39] = (mole_frac[39] > 1e-200) ? mole_frac[39] : 1e-200;
    mole_frac[39] *= sumyow;
    mole_frac[40] = mass_frac[40] * recip_molecular_masses[40];
    mole_frac[40] = (mole_frac[40] > 1e-200) ? mole_frac[40] : 1e-200;
    mole_frac[40] *= sumyow;
    mole_frac[41] = mass_frac[41] * recip_molecular_masses[41];
    mole_frac[41] = (mole_frac[41] > 1e-200) ? mole_frac[41] : 1e-200;
    mole_frac[41] *= sumyow;
    mole_frac[42] = mass_frac[42] * recip_molecular_masses[42];
    mole_frac[42] = (mole_frac[42] > 1e-200) ? mole_frac[42] : 1e-200;
    mole_frac[42] *= sumyow;
    mole_frac[43] = mass_frac[43] * recip_molecular_masses[43];
    mole_frac[43] = (mole_frac[43] > 1e-200) ? mole_frac[43] : 1e-200;
    mole_frac[43] *= sumyow;
    mole_frac[44] = mass_frac[44] * recip_molecular_masses[44];
    mole_frac[44] = (mole_frac[44] > 1e-200) ? mole_frac[44] : 1e-200;
    mole_frac[44] *= sumyow;
    mole_frac[45] = mass_frac[45] * recip_molecular_masses[45];
    mole_frac[45] = (mole_frac[45] > 1e-200) ? mole_frac[45] : 1e-200;
    mole_frac[45] *= sumyow;
    mole_frac[46] = mass_frac[46] * recip_molecular_masses[46];
    mole_frac[46] = (mole_frac[46] > 1e-200) ? mole_frac[46] : 1e-200;
    mole_frac[46] *= sumyow;
    mole_frac[47] = mass_frac[47] * recip_molecular_masses[47];
    mole_frac[47] = (mole_frac[47] > 1e-200) ? mole_frac[47] : 1e-200;
    mole_frac[47] *= sumyow;
    mole_frac[48] = mass_frac[48] * recip_molecular_masses[48];
    mole_frac[48] = (mole_frac[48] > 1e-200) ? mole_frac[48] : 1e-200;
    mole_frac[48] *= sumyow;
    mole_frac[49] = mass_frac[49] * recip_molecular_masses[49];
    mole_frac[49] = (mole_frac[49] > 1e-200) ? mole_frac[49] : 1e-200;
    mole_frac[49] *= sumyow;
    mole_frac[50] = mass_frac[50] * recip_molecular_masses[50];
    mole_frac[50] = (mole_frac[50] > 1e-200) ? mole_frac[50] : 1e-200;
    mole_frac[50] *= sumyow;
    mole_frac[51] = mass_frac[51] * recip_molecular_masses[51];
    mole_frac[51] = (mole_frac[51] > 1e-200) ? mole_frac[51] : 1e-200;
    mole_frac[51] *= sumyow;
    mole_frac[52] = mass_frac[52] * recip_molecular_masses[52];
    mole_frac[52] = (mole_frac[52] > 1e-200) ? mole_frac[52] : 1e-200;
    mole_frac[52] *= sumyow;
    mole_frac[53] = mass_frac[53] * recip_molecular_masses[53];
    mole_frac[53] = (mole_frac[53] > 1e-200) ? mole_frac[53] : 1e-200;
    mole_frac[53] *= sumyow;
    mole_frac[54] = mass_frac[54] * recip_molecular_masses[54];
    mole_frac[54] = (mole_frac[54] > 1e-200) ? mole_frac[54] : 1e-200;
    mole_frac[54] *= sumyow;
    mole_frac[55] = mass_frac[55] * recip_molecular_masses[55];
    mole_frac[55] = (mole_frac[55] > 1e-200) ? mole_frac[55] : 1e-200;
    mole_frac[55] *= sumyow;
    mole_frac[56] = mass_frac[56] * recip_molecular_masses[56];
    mole_frac[56] = (mole_frac[56] > 1e-200) ? mole_frac[56] : 1e-200;
    mole_frac[56] *= sumyow;
    mole_frac[57] = mass_frac[57] * recip_molecular_masses[57];
    mole_frac[57] = (mole_frac[57] > 1e-200) ? mole_frac[57] : 1e-200;
    mole_frac[57] *= sumyow;
    mole_frac[58] = mass_frac[58] * recip_molecular_masses[58];
    mole_frac[58] = (mole_frac[58] > 1e-200) ? mole_frac[58] : 1e-200;
    mole_frac[58] *= sumyow;
    mole_frac[59] = mass_frac[59] * recip_molecular_masses[59];
    mole_frac[59] = (mole_frac[59] > 1e-200) ? mole_frac[59] : 1e-200;
    mole_frac[59] *= sumyow;
    mole_frac[60] = mass_frac[60] * recip_molecular_masses[60];
    mole_frac[60] = (mole_frac[60] > 1e-200) ? mole_frac[60] : 1e-200;
    mole_frac[60] *= sumyow;
    mole_frac[61] = mass_frac[61] * recip_molecular_masses[61];
    mole_frac[61] = (mole_frac[61] > 1e-200) ? mole_frac[61] : 1e-200;
    mole_frac[61] *= sumyow;
    mole_frac[62] = mass_frac[62] * recip_molecular_masses[62];
    mole_frac[62] = (mole_frac[62] > 1e-200) ? mole_frac[62] : 1e-200;
    mole_frac[62] *= sumyow;
    mole_frac[63] = mass_frac[63] * recip_molecular_masses[63];
    mole_frac[63] = (mole_frac[63] > 1e-200) ? mole_frac[63] : 1e-200;
    mole_frac[63] *= sumyow;
    mole_frac[64] = mass_frac[64] * recip_molecular_masses[64];
    mole_frac[64] = (mole_frac[64] > 1e-200) ? mole_frac[64] : 1e-200;
    mole_frac[64] *= sumyow;
    mole_frac[65] = mass_frac[65] * recip_molecular_masses[65];
    mole_frac[65] = (mole_frac[65] > 1e-200) ? mole_frac[65] : 1e-200;
    mole_frac[65] *= sumyow;
    mole_frac[66] = mass_frac[66] * recip_molecular_masses[66];
    mole_frac[66] = (mole_frac[66] > 1e-200) ? mole_frac[66] : 1e-200;
    mole_frac[66] *= sumyow;
    mole_frac[67] = mass_frac[67] * recip_molecular_masses[67];
    mole_frac[67] = (mole_frac[67] > 1e-200) ? mole_frac[67] : 1e-200;
    mole_frac[67] *= sumyow;
    mole_frac[68] = mass_frac[68] * recip_molecular_masses[68];
    mole_frac[68] = (mole_frac[68] > 1e-200) ? mole_frac[68] : 1e-200;
    mole_frac[68] *= sumyow;
    mole_frac[69] = mass_frac[69] * recip_molecular_masses[69];
    mole_frac[69] = (mole_frac[69] > 1e-200) ? mole_frac[69] : 1e-200;
    mole_frac[69] *= sumyow;
    mole_frac[70] = mass_frac[70] * recip_molecular_masses[70];
    mole_frac[70] = (mole_frac[70] > 1e-200) ? mole_frac[70] : 1e-200;
    mole_frac[70] *= sumyow;
    mole_frac[71] = mass_frac[71] * recip_molecular_masses[71];
    mole_frac[71] = (mole_frac[71] > 1e-200) ? mole_frac[71] : 1e-200;
    mole_frac[71] *= sumyow;
    mole_frac[72] = mass_frac[72] * recip_molecular_masses[72];
    mole_frac[72] = (mole_frac[72] > 1e-200) ? mole_frac[72] : 1e-200;
    mole_frac[72] *= sumyow;
    mole_frac[73] = mass_frac[73] * recip_molecular_masses[73];
    mole_frac[73] = (mole_frac[73] > 1e-200) ? mole_frac[73] : 1e-200;
    mole_frac[73] *= sumyow;
    mole_frac[74] = mass_frac[74] * recip_molecular_masses[74];
    mole_frac[74] = (mole_frac[74] > 1e-200) ? mole_frac[74] : 1e-200;
    mole_frac[74] *= sumyow;
    mole_frac[75] = mass_frac[75] * recip_molecular_masses[75];
    mole_frac[75] = (mole_frac[75] > 1e-200) ? mole_frac[75] : 1e-200;
    mole_frac[75] *= sumyow;
    mole_frac[76] = mass_frac[76] * recip_molecular_masses[76];
    mole_frac[76] = (mole_frac[76] > 1e-200) ? mole_frac[76] : 1e-200;
    mole_frac[76] *= sumyow;
    mole_frac[77] = mass_frac[77] * recip_molecular_masses[77];
    mole_frac[77] = (mole_frac[77] > 1e-200) ? mole_frac[77] : 1e-200;
    mole_frac[77] *= sumyow;
    mole_frac[78] = mass_frac[78] * recip_molecular_masses[78];
    mole_frac[78] = (mole_frac[78] > 1e-200) ? mole_frac[78] : 1e-200;
    mole_frac[78] *= sumyow;
    mole_frac[79] = mass_frac[79] * recip_molecular_masses[79];
    mole_frac[79] = (mole_frac[79] > 1e-200) ? mole_frac[79] : 1e-200;
    mole_frac[79] *= sumyow;
    mole_frac[80] = mass_frac[80] * recip_molecular_masses[80];
    mole_frac[80] = (mole_frac[80] > 1e-200) ? mole_frac[80] : 1e-200;
    mole_frac[80] *= sumyow;
    mole_frac[81] = mass_frac[81] * recip_molecular_masses[81];
    mole_frac[81] = (mole_frac[81] > 1e-200) ? mole_frac[81] : 1e-200;
    mole_frac[81] *= sumyow;
    mole_frac[82] = mass_frac[82] * recip_molecular_masses[82];
    mole_frac[82] = (mole_frac[82] > 1e-200) ? mole_frac[82] : 1e-200;
    mole_frac[82] *= sumyow;
    mole_frac[83] = mass_frac[83] * recip_molecular_masses[83];
    mole_frac[83] = (mole_frac[83] > 1e-200) ? mole_frac[83] : 1e-200;
    mole_frac[83] *= sumyow;
    mole_frac[84] = mass_frac[84] * recip_molecular_masses[84];
    mole_frac[84] = (mole_frac[84] > 1e-200) ? mole_frac[84] : 1e-200;
    mole_frac[84] *= sumyow;
    mole_frac[85] = mass_frac[85] * recip_molecular_masses[85];
    mole_frac[85] = (mole_frac[85] > 1e-200) ? mole_frac[85] : 1e-200;
    mole_frac[85] *= sumyow;
    mole_frac[86] = mass_frac[86] * recip_molecular_masses[86];
    mole_frac[86] = (mole_frac[86] > 1e-200) ? mole_frac[86] : 1e-200;
    mole_frac[86] *= sumyow;
    mole_frac[87] = mass_frac[87] * recip_molecular_masses[87];
    mole_frac[87] = (mole_frac[87] > 1e-200) ? mole_frac[87] : 1e-200;
    mole_frac[87] *= sumyow;
    mole_frac[88] = mass_frac[88] * recip_molecular_masses[88];
    mole_frac[88] = (mole_frac[88] > 1e-200) ? mole_frac[88] : 1e-200;
    mole_frac[88] *= sumyow;
    mole_frac[89] = mass_frac[89] * recip_molecular_masses[89];
    mole_frac[89] = (mole_frac[89] > 1e-200) ? mole_frac[89] : 1e-200;
    mole_frac[89] *= sumyow;
    mole_frac[90] = mass_frac[90] * recip_molecular_masses[90];
    mole_frac[90] = (mole_frac[90] > 1e-200) ? mole_frac[90] : 1e-200;
    mole_frac[90] *= sumyow;
    mole_frac[91] = mass_frac[91] * recip_molecular_masses[91];
    mole_frac[91] = (mole_frac[91] > 1e-200) ? mole_frac[91] : 1e-200;
    mole_frac[91] *= sumyow;
    mole_frac[92] = mass_frac[92] * recip_molecular_masses[92];
    mole_frac[92] = (mole_frac[92] > 1e-200) ? mole_frac[92] : 1e-200;
    mole_frac[92] *= sumyow;
    mole_frac[93] = mass_frac[93] * recip_molecular_masses[93];
    mole_frac[93] = (mole_frac[93] > 1e-200) ? mole_frac[93] : 1e-200;
    mole_frac[93] *= sumyow;
    mole_frac[94] = mass_frac[94] * recip_molecular_masses[94];
    mole_frac[94] = (mole_frac[94] > 1e-200) ? mole_frac[94] : 1e-200;
    mole_frac[94] *= sumyow;
    mole_frac[95] = mass_frac[95] * recip_molecular_masses[95];
    mole_frac[95] = (mole_frac[95] > 1e-200) ? mole_frac[95] : 1e-200;
    mole_frac[95] *= sumyow;
    mole_frac[96] = mass_frac[96] * recip_molecular_masses[96];
    mole_frac[96] = (mole_frac[96] > 1e-200) ? mole_frac[96] : 1e-200;
    mole_frac[96] *= sumyow;
    mole_frac[97] = mass_frac[97] * recip_molecular_masses[97];
    mole_frac[97] = (mole_frac[97] > 1e-200) ? mole_frac[97] : 1e-200;
    mole_frac[97] *= sumyow;
    mole_frac[98] = mass_frac[98] * recip_molecular_masses[98];
    mole_frac[98] = (mole_frac[98] > 1e-200) ? mole_frac[98] : 1e-200;
    mole_frac[98] *= sumyow;
    mole_frac[99] = mass_frac[99] * recip_molecular_masses[99];
    mole_frac[99] = (mole_frac[99] > 1e-200) ? mole_frac[99] : 1e-200;
    mole_frac[99] *= sumyow;
    mole_frac[100] = mass_frac[100] * recip_molecular_masses[100];
    mole_frac[100] = (mole_frac[100] > 1e-200) ? mole_frac[100] : 1e-200;
    mole_frac[100] *= sumyow;
    mole_frac[101] = mass_frac[101] * recip_molecular_masses[101];
    mole_frac[101] = (mole_frac[101] > 1e-200) ? mole_frac[101] : 1e-200;
    mole_frac[101] *= sumyow;
    mole_frac[102] = mass_frac[102] * recip_molecular_masses[102];
    mole_frac[102] = (mole_frac[102] > 1e-200) ? mole_frac[102] : 1e-200;
    mole_frac[102] *= sumyow;
    mole_frac[103] = mass_frac[103] * recip_molecular_masses[103];
    mole_frac[103] = (mole_frac[103] > 1e-200) ? mole_frac[103] : 1e-200;
    mole_frac[103] *= sumyow;
    mole_frac[104] = mass_frac[104] * recip_molecular_masses[104];
    mole_frac[104] = (mole_frac[104] > 1e-200) ? mole_frac[104] : 1e-200;
    mole_frac[104] *= sumyow;
    mole_frac[105] = mass_frac[105] * recip_molecular_masses[105];
    mole_frac[105] = (mole_frac[105] > 1e-200) ? mole_frac[105] : 1e-200;
    mole_frac[105] *= sumyow;
    mole_frac[106] = mass_frac[106] * recip_molecular_masses[106];
    mole_frac[106] = (mole_frac[106] > 1e-200) ? mole_frac[106] : 1e-200;
    mole_frac[106] *= sumyow;
    mole_frac[107] = mass_frac[107] * recip_molecular_masses[107];
    mole_frac[107] = (mole_frac[107] > 1e-200) ? mole_frac[107] : 1e-200;
    mole_frac[107] *= sumyow;
    mole_frac[108] = mass_frac[108] * recip_molecular_masses[108];
    mole_frac[108] = (mole_frac[108] > 1e-200) ? mole_frac[108] : 1e-200;
    mole_frac[108] *= sumyow;
    mole_frac[109] = mass_frac[109] * recip_molecular_masses[109];
    mole_frac[109] = (mole_frac[109] > 1e-200) ? mole_frac[109] : 1e-200;
    mole_frac[109] *= sumyow;
    mole_frac[110] = mass_frac[110] * recip_molecular_masses[110];
    mole_frac[110] = (mole_frac[110] > 1e-200) ? mole_frac[110] : 1e-200;
    mole_frac[110] *= sumyow;
    mole_frac[111] = mass_frac[111] * recip_molecular_masses[111];
    mole_frac[111] = (mole_frac[111] > 1e-200) ? mole_frac[111] : 1e-200;
    mole_frac[111] *= sumyow;
    mole_frac[112] = mass_frac[112] * recip_molecular_masses[112];
    mole_frac[112] = (mole_frac[112] > 1e-200) ? mole_frac[112] : 1e-200;
    mole_frac[112] *= sumyow;
    mole_frac[113] = mass_frac[113] * recip_molecular_masses[113];
    mole_frac[113] = (mole_frac[113] > 1e-200) ? mole_frac[113] : 1e-200;
    mole_frac[113] *= sumyow;
    mole_frac[114] = mass_frac[114] * recip_molecular_masses[114];
    mole_frac[114] = (mole_frac[114] > 1e-200) ? mole_frac[114] : 1e-200;
    mole_frac[114] *= sumyow;
    mole_frac[115] = mass_frac[115] * recip_molecular_masses[115];
    mole_frac[115] = (mole_frac[115] > 1e-200) ? mole_frac[115] : 1e-200;
    mole_frac[115] *= sumyow;
    mole_frac[116] = mass_frac[116] * recip_molecular_masses[116];
    mole_frac[116] = (mole_frac[116] > 1e-200) ? mole_frac[116] : 1e-200;
    mole_frac[116] *= sumyow;
    mole_frac[117] = mass_frac[117] * recip_molecular_masses[117];
    mole_frac[117] = (mole_frac[117] > 1e-200) ? mole_frac[117] : 1e-200;
    mole_frac[117] *= sumyow;
    mole_frac[118] = mass_frac[118] * recip_molecular_masses[118];
    mole_frac[118] = (mole_frac[118] > 1e-200) ? mole_frac[118] : 1e-200;
    mole_frac[118] *= sumyow;
    mole_frac[119] = mass_frac[119] * recip_molecular_masses[119];
    mole_frac[119] = (mole_frac[119] > 1e-200) ? mole_frac[119] : 1e-200;
    mole_frac[119] *= sumyow;
    mole_frac[120] = mass_frac[120] * recip_molecular_masses[120];
    mole_frac[120] = (mole_frac[120] > 1e-200) ? mole_frac[120] : 1e-200;
    mole_frac[120] *= sumyow;
    mole_frac[121] = mass_frac[121] * recip_molecular_masses[121];
    mole_frac[121] = (mole_frac[121] > 1e-200) ? mole_frac[121] : 1e-200;
    mole_frac[121] *= sumyow;
    mole_frac[122] = mass_frac[122] * recip_molecular_masses[122];
    mole_frac[122] = (mole_frac[122] > 1e-200) ? mole_frac[122] : 1e-200;
    mole_frac[122] *= sumyow;
    mole_frac[123] = mass_frac[123] * recip_molecular_masses[123];
    mole_frac[123] = (mole_frac[123] > 1e-200) ? mole_frac[123] : 1e-200;
    mole_frac[123] *= sumyow;
    mole_frac[124] = mass_frac[124] * recip_molecular_masses[124];
    mole_frac[124] = (mole_frac[124] > 1e-200) ? mole_frac[124] : 1e-200;
    mole_frac[124] *= sumyow;
    mole_frac[125] = mass_frac[125] * recip_molecular_masses[125];
    mole_frac[125] = (mole_frac[125] > 1e-200) ? mole_frac[125] : 1e-200;
    mole_frac[125] *= sumyow;
    mole_frac[126] = mass_frac[126] * recip_molecular_masses[126];
    mole_frac[126] = (mole_frac[126] > 1e-200) ? mole_frac[126] : 1e-200;
    mole_frac[126] *= sumyow;
    mole_frac[127] = mass_frac[127] * recip_molecular_masses[127];
    mole_frac[127] = (mole_frac[127] > 1e-200) ? mole_frac[127] : 1e-200;
    mole_frac[127] *= sumyow;
    mole_frac[128] = mass_frac[128] * recip_molecular_masses[128];
    mole_frac[128] = (mole_frac[128] > 1e-200) ? mole_frac[128] : 1e-200;
    mole_frac[128] *= sumyow;
    mole_frac[129] = mass_frac[129] * recip_molecular_masses[129];
    mole_frac[129] = (mole_frac[129] > 1e-200) ? mole_frac[129] : 1e-200;
    mole_frac[129] *= sumyow;
    mole_frac[130] = mass_frac[130] * recip_molecular_masses[130];
    mole_frac[130] = (mole_frac[130] > 1e-200) ? mole_frac[130] : 1e-200;
    mole_frac[130] *= sumyow;
    mole_frac[131] = mass_frac[131] * recip_molecular_masses[131];
    mole_frac[131] = (mole_frac[131] > 1e-200) ? mole_frac[131] : 1e-200;
    mole_frac[131] *= sumyow;
    mole_frac[132] = mass_frac[132] * recip_molecular_masses[132];
    mole_frac[132] = (mole_frac[132] > 1e-200) ? mole_frac[132] : 1e-200;
    mole_frac[132] *= sumyow;
    mole_frac[133] = mass_frac[133] * recip_molecular_masses[133];
    mole_frac[133] = (mole_frac[133] > 1e-200) ? mole_frac[133] : 1e-200;
    mole_frac[133] *= sumyow;
    mole_frac[134] = mass_frac[134] * recip_molecular_masses[134];
    mole_frac[134] = (mole_frac[134] > 1e-200) ? mole_frac[134] : 1e-200;
    mole_frac[134] *= sumyow;
    mole_frac[135] = mass_frac[135] * recip_molecular_masses[135];
    mole_frac[135] = (mole_frac[135] > 1e-200) ? mole_frac[135] : 1e-200;
    mole_frac[135] *= sumyow;
    mole_frac[136] = mass_frac[136] * recip_molecular_masses[136];
    mole_frac[136] = (mole_frac[136] > 1e-200) ? mole_frac[136] : 1e-200;
    mole_frac[136] *= sumyow;
    mole_frac[137] = mass_frac[137] * recip_molecular_masses[137];
    mole_frac[137] = (mole_frac[137] > 1e-200) ? mole_frac[137] : 1e-200;
    mole_frac[137] *= sumyow;
    mole_frac[138] = mass_frac[138] * recip_molecular_masses[138];
    mole_frac[138] = (mole_frac[138] > 1e-200) ? mole_frac[138] : 1e-200;
    mole_frac[138] *= sumyow;
    mole_frac[139] = mass_frac[139] * recip_molecular_masses[139];
    mole_frac[139] = (mole_frac[139] > 1e-200) ? mole_frac[139] : 1e-200;
    mole_frac[139] *= sumyow;
    mole_frac[140] = mass_frac[140] * recip_molecular_masses[140];
    mole_frac[140] = (mole_frac[140] > 1e-200) ? mole_frac[140] : 1e-200;
    mole_frac[140] *= sumyow;
    mole_frac[141] = mass_frac[141] * recip_molecular_masses[141];
    mole_frac[141] = (mole_frac[141] > 1e-200) ? mole_frac[141] : 1e-200;
    mole_frac[141] *= sumyow;
    mole_frac[142] = mass_frac[142] * recip_molecular_masses[142];
    mole_frac[142] = (mole_frac[142] > 1e-200) ? mole_frac[142] : 1e-200;
    mole_frac[142] *= sumyow;
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
    ctot += mole_frac[35];
    ctot += mole_frac[36];
    ctot += mole_frac[37];
    ctot += mole_frac[38];
    ctot += mole_frac[39];
    ctot += mole_frac[40];
    ctot += mole_frac[41];
    ctot += mole_frac[42];
    ctot += mole_frac[43];
    ctot += mole_frac[44];
    ctot += mole_frac[45];
    ctot += mole_frac[46];
    ctot += mole_frac[47];
    ctot += mole_frac[48];
    ctot += mole_frac[49];
    ctot += mole_frac[50];
    ctot += mole_frac[51];
    ctot += mole_frac[52];
    ctot += mole_frac[53];
    ctot += mole_frac[54];
    ctot += mole_frac[55];
    ctot += mole_frac[56];
    ctot += mole_frac[57];
    ctot += mole_frac[58];
    ctot += mole_frac[59];
    ctot += mole_frac[60];
    ctot += mole_frac[61];
    ctot += mole_frac[62];
    ctot += mole_frac[63];
    ctot += mole_frac[64];
    ctot += mole_frac[65];
    ctot += mole_frac[66];
    ctot += mole_frac[67];
    ctot += mole_frac[68];
    ctot += mole_frac[69];
    ctot += mole_frac[70];
    ctot += mole_frac[71];
    ctot += mole_frac[72];
    ctot += mole_frac[73];
    ctot += mole_frac[74];
    ctot += mole_frac[75];
    ctot += mole_frac[76];
    ctot += mole_frac[77];
    ctot += mole_frac[78];
    ctot += mole_frac[79];
    ctot += mole_frac[80];
    ctot += mole_frac[81];
    ctot += mole_frac[82];
    ctot += mole_frac[83];
    ctot += mole_frac[84];
    ctot += mole_frac[85];
    ctot += mole_frac[86];
    ctot += mole_frac[87];
    ctot += mole_frac[88];
    ctot += mole_frac[89];
    ctot += mole_frac[90];
    ctot += mole_frac[91];
    ctot += mole_frac[92];
    ctot += mole_frac[93];
    ctot += mole_frac[94];
    ctot += mole_frac[95];
    ctot += mole_frac[96];
    ctot += mole_frac[97];
    ctot += mole_frac[98];
    ctot += mole_frac[99];
    ctot += mole_frac[100];
    ctot += mole_frac[101];
    ctot += mole_frac[102];
    ctot += mole_frac[103];
    ctot += mole_frac[104];
    ctot += mole_frac[105];
    ctot += mole_frac[106];
    ctot += mole_frac[107];
    ctot += mole_frac[108];
    ctot += mole_frac[109];
    ctot += mole_frac[110];
    ctot += mole_frac[111];
    ctot += mole_frac[112];
    ctot += mole_frac[113];
    ctot += mole_frac[114];
    ctot += mole_frac[115];
    ctot += mole_frac[116];
    ctot += mole_frac[117];
    ctot += mole_frac[118];
    ctot += mole_frac[119];
    ctot += mole_frac[120];
    ctot += mole_frac[121];
    ctot += mole_frac[122];
    ctot += mole_frac[123];
    ctot += mole_frac[124];
    ctot += mole_frac[125];
    ctot += mole_frac[126];
    ctot += mole_frac[127];
    ctot += mole_frac[128];
    ctot += mole_frac[129];
    ctot += mole_frac[130];
    ctot += mole_frac[131];
    ctot += mole_frac[132];
    ctot += mole_frac[133];
    ctot += mole_frac[134];
    ctot += mole_frac[135];
    ctot += mole_frac[136];
    ctot += mole_frac[137];
    ctot += mole_frac[138];
    ctot += mole_frac[139];
    ctot += mole_frac[140];
    ctot += mole_frac[141];
    ctot += mole_frac[142];
    thbctemp[0] = ctot;
    thbctemp[1] = ctot + 1.5*mole_frac[1] + 11.0*mole_frac[5] + 
      0.8999999999999999*mole_frac[8] + 2.8*mole_frac[9]; 
    thbctemp[2] = ctot + 1.5*mole_frac[1] + 11.0*mole_frac[5] + 
      0.8999999999999999*mole_frac[8] + 2.8*mole_frac[9] + mole_frac[19] + 
      2.0*mole_frac[23]; 
    thbctemp[3] = ctot + 1.5*mole_frac[1] + 11.0*mole_frac[5] + 0.5*mole_frac[8] 
      + mole_frac[9] + mole_frac[19] + 2.0*mole_frac[23]; 
    thbctemp[4] = ctot - 0.27*mole_frac[1] + 11.0*mole_frac[5] + mole_frac[19] + 
      2.0*mole_frac[23]; 
    thbctemp[5] = ctot + 0.3*mole_frac[1] + 13.0*mole_frac[5] + 
      0.8999999999999999*mole_frac[8] + 2.8*mole_frac[9] + mole_frac[19] + 
      2.0*mole_frac[23]; 
    thbctemp[6] = ctot + mole_frac[1] + 5.0*mole_frac[3] + 5.0*mole_frac[5] + 
      0.5*mole_frac[8] + 2.5*mole_frac[9] + mole_frac[19] + 2.0*mole_frac[23]; 
    thbctemp[7] = ctot + mole_frac[1] + 11.0*mole_frac[5] + 0.5*mole_frac[8] + 
      mole_frac[9] + mole_frac[19] + 2.0*mole_frac[23]; 
    thbctemp[8] = ctot + mole_frac[1] + 5.0*mole_frac[5] + 0.5*mole_frac[8] + 
      mole_frac[9] + mole_frac[19] + 2.0*mole_frac[23]; 
  }
  
  double rr_f[643];
  double rr_r[643];
  //   0)  H + O2 <=> O + OH
  {
    double forward = 3.547e+15 * exp(-0.406*vlntemp - 1.66e+04*ortc);
    double reverse = 1.027e+13 * exp(-0.015*vlntemp + 133.0*ortc);
    rr_f[0] = forward * mole_frac[0] * mole_frac[3];
    rr_r[0] = reverse * mole_frac[2] * mole_frac[4];
  }
  //   1)  H2 + O <=> H + OH
  {
    double forward = 5.08e+04 * exp(2.67*vlntemp - 6.292e+03*ortc);
    double reverse = 2.637e+04 * exp(2.651*vlntemp - 4.88e+03*ortc);
    rr_f[1] = forward * mole_frac[1] * mole_frac[2];
    rr_r[1] = reverse * mole_frac[0] * mole_frac[4];
  }
  //   2)  H2 + OH <=> H + H2O
  {
    double forward = 2.16e+08 * exp(1.51*vlntemp - 3.43e+03*ortc);
    double reverse = 2.29e+09 * exp(1.404*vlntemp - 1.832e+04*ortc);
    rr_f[2] = forward * mole_frac[1] * mole_frac[4];
    rr_r[2] = reverse * mole_frac[0] * mole_frac[5];
  }
  //   3)  O + H2O <=> 2 OH
  {
    double forward = 2.97e+06 * exp(2.02*vlntemp - 1.34e+04*ortc);
    double reverse = 1.454e+05 * exp(2.107*vlntemp + 2.904e+03*ortc);
    rr_f[3] = forward * mole_frac[2] * mole_frac[5];
    rr_r[3] = reverse * mole_frac[4] * mole_frac[4];
  }
  //   4)  H2 + M <=> 2 H + M
  {
    double forward = 4.577e+19 * exp(-1.4*vlntemp - 1.044e+05*ortc);
    double reverse = 1.145e+20 * exp(-1.676*vlntemp - 820.0*ortc);
    rr_f[4] = forward * mole_frac[1];
    rr_r[4] = reverse * mole_frac[0] * mole_frac[0];
    rr_f[4] *= thbctemp[1];
    rr_r[4] *= thbctemp[1];
  }
  //   5)  O2 + M <=> 2 O + M
  {
    double forward = 4.42e+17 * exp(-0.634*vlntemp - 1.189e+05*ortc);
    double reverse = 6.165e+15 * exp(-0.5 * vlntemp);
    rr_f[5] = forward * mole_frac[3];
    rr_r[5] = reverse * mole_frac[2] * mole_frac[2];
    rr_f[5] *= thbctemp[2];
    rr_r[5] *= thbctemp[2];
  }
  //   6)  OH + M <=> H + O + M
  {
    double forward = 9.78e+17 * exp(-0.743*vlntemp - 1.021e+05*ortc);
    double reverse = 4.714e+18 * otc;
    rr_f[6] = forward * mole_frac[4];
    rr_r[6] = reverse * mole_frac[0] * mole_frac[2];
    rr_f[6] *= thbctemp[3];
    rr_r[6] *= thbctemp[3];
  }
  //   7)  H2O + M <=> H + OH + M
  {
    double forward = 1.907e+23 * exp(-1.83*vlntemp - 1.185e+05*ortc);
    double reverse = 4.5e+22 * otc * otc;
    rr_f[7] = forward * mole_frac[5];
    rr_r[7] = reverse * mole_frac[0] * mole_frac[4];
    rr_f[7] *= thbctemp[4];
    rr_r[7] *= thbctemp[4];
  }
  //   8)  H + O2 (+M) <=> HO2 (+M)
  {
    double rr_k0 = 3.482e+16 * exp(-0.411*vlntemp + 1.115e+03*ortc);
    double rr_kinf = 1.475e+12 * exp(0.6 * vlntemp);
    double pr = rr_k0 / rr_kinf * thbctemp[5];
    double fcent = log10(MAX(0.5 * exp(-9.999999999999999e+29 * temperature) + 
      0.5 * exp(-9.999999999999999e-31 * temperature) + exp(-1.0e+10 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] - cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[8] = forward * mole_frac[0] * mole_frac[3];
    rr_r[8] = reverse * mole_frac[6];
  }
  //   9)  H + HO2 <=> H2 + O2
  {
    double forward = 1.66e+13 * exp(-823.0*ortc);
    double reverse = 3.166e+12 * exp(0.348*vlntemp - 5.551e+04*ortc);
    rr_f[9] = forward * mole_frac[0] * mole_frac[6];
    rr_r[9] = reverse * mole_frac[1] * mole_frac[3];
  }
  //  10)  H + HO2 <=> 2 OH
  {
    double forward = 7.079e+13 * exp(-295.0*ortc);
    double reverse = 2.028e+10 * exp(0.72*vlntemp - 3.684e+04*ortc);
    rr_f[10] = forward * mole_frac[0] * mole_frac[6];
    rr_r[10] = reverse * mole_frac[4] * mole_frac[4];
  }
  //  11)  O + HO2 <=> O2 + OH
  {
    double forward = 3.25e+13;
    double reverse = 3.217e+12 * exp(0.329*vlntemp - 5.328e+04*ortc);
    rr_f[11] = forward * mole_frac[2] * mole_frac[6];
    rr_r[11] = reverse * mole_frac[3] * mole_frac[4];
  }
  //  12)  OH + HO2 <=> O2 + H2O
  {
    double forward = 1.973e+10 * exp(0.962*vlntemp + 328.4*ortc);
    double reverse = 3.989e+10 * exp(1.204*vlntemp - 6.925e+04*ortc);
    rr_f[12] = forward * mole_frac[4] * mole_frac[6];
    rr_r[12] = reverse * mole_frac[3] * mole_frac[5];
  }
  //  13)  O2 + H2O2 <=> 2 HO2
  {
    double forward = 1.136e+16 * exp(-0.347*vlntemp - 4.973e+04*ortc);
    double reverse = 1.03e+14 * exp(-1.104e+04*ortc);
    rr_f[13] = forward * mole_frac[3] * mole_frac[7];
    rr_r[13] = reverse * mole_frac[6] * mole_frac[6];
  }
  //  14)  O2 + H2O2 <=> 2 HO2
  {
    double forward = 2.141e+13 * exp(-0.347*vlntemp - 3.728e+04*ortc);
    double reverse = 1.94e+11 * exp(1.409e+03*ortc);
    rr_f[14] = forward * mole_frac[3] * mole_frac[7];
    rr_r[14] = reverse * mole_frac[6] * mole_frac[6];
  }
  //  15)  H2O2 (+M) <=> 2 OH (+M)
  {
    double rr_k0 = 1.202e+17 * exp(-4.55e+04*ortc);
    double rr_kinf = 2.951e+14 * exp(-4.843e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[2];
    double fcent = log10(MAX(0.5 * exp(-9.999999999999999e+29 * temperature) + 
      0.5 * exp(-9.999999999999999e-31 * temperature) + exp(-1.0e+10 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = 2.0 * cgspl[4] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[15] = forward * mole_frac[7];
    rr_r[15] = reverse * mole_frac[4] * mole_frac[4];
  }
  //  16)  H + H2O2 <=> OH + H2O
  {
    double forward = 2.41e+13 * exp(-3.97e+03*ortc);
    double reverse = 1.265e+08 * exp(1.31*vlntemp - 7.141e+04*ortc);
    rr_f[16] = forward * mole_frac[0] * mole_frac[7];
    rr_r[16] = reverse * mole_frac[4] * mole_frac[5];
  }
  //  17)  H + H2O2 <=> H2 + HO2
  {
    double forward = 2.15e+10 * temperature * exp(-6.0e+03*ortc);
    double reverse = 3.716e+07 * exp(1.695*vlntemp - 2.2e+04*ortc);
    rr_f[17] = forward * mole_frac[0] * mole_frac[7];
    rr_r[17] = reverse * mole_frac[1] * mole_frac[6];
  }
  //  18)  O + H2O2 <=> OH + HO2
  {
    double forward = 9.55e+06 * temperature * temperature * exp(-3.97e+03*ortc);
    double reverse = 8.568e+03 * exp(2.676*vlntemp - 1.856e+04*ortc);
    rr_f[18] = forward * mole_frac[2] * mole_frac[7];
    rr_r[18] = reverse * mole_frac[4] * mole_frac[6];
  }
  //  19)  OH + H2O2 <=> H2O + HO2
  {
    double forward = 2.0e+12 * exp(-427.2*ortc);
    double reverse = 3.665e+10 * exp(0.589*vlntemp - 3.132e+04*ortc);
    rr_f[19] = forward * mole_frac[4] * mole_frac[7];
    rr_r[19] = reverse * mole_frac[5] * mole_frac[6];
  }
  //  20)  OH + H2O2 <=> H2O + HO2
  {
    double forward = 1.7e+18 * exp(-2.941e+04*ortc);
    double reverse = 3.115e+16 * exp(0.589*vlntemp - 6.03e+04*ortc);
    rr_f[20] = forward * mole_frac[4] * mole_frac[7];
    rr_r[20] = reverse * mole_frac[5] * mole_frac[6];
  }
  //  21)  O + CO (+M) <=> CO2 (+M)
  {
    double rr_k0 = 1.35e+24 * exp(-2.788*vlntemp - 4.191e+03*ortc);
    double rr_kinf = 1.8e+10 * exp(-2.384e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[6];
    double forward = rr_kinf * pr/(1.0 + pr);
    double xik = -cgspl[2] - cgspl[8] + cgspl[9];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[21] = forward * mole_frac[2] * mole_frac[8];
    rr_r[21] = reverse * mole_frac[9];
  }
  //  22)  O2 + CO <=> O + CO2
  {
    double forward = 1.05e+12 * exp(-4.254e+04*ortc);
    double reverse = 8.035e+15 * exp(-0.8*vlntemp - 5.123e+04*ortc);
    rr_f[22] = forward * mole_frac[3] * mole_frac[8];
    rr_r[22] = reverse * mole_frac[2] * mole_frac[9];
  }
  //  23)  OH + CO <=> H + CO2
  {
    double forward = 2.23e+05 * exp(1.89*vlntemp + 1.158e+03*ortc);
    double reverse = 5.896e+11 * exp(0.699*vlntemp - 2.426e+04*ortc);
    rr_f[23] = forward * mole_frac[4] * mole_frac[8];
    rr_r[23] = reverse * mole_frac[0] * mole_frac[9];
  }
  //  24)  HO2 + CO <=> OH + CO2
  {
    double forward = 3.01e+13 * exp(-2.3e+04*ortc);
    double reverse = 2.28e+16 * exp(-0.47*vlntemp - 8.497e+04*ortc);
    rr_f[24] = forward * mole_frac[6] * mole_frac[8];
    rr_r[24] = reverse * mole_frac[4] * mole_frac[9];
  }
  //  25)  HCO + M <=> H + CO + M
  {
    double forward = 4.75e+11 * exp(0.66*vlntemp - 1.487e+04*ortc);
    double reverse = 3.582e+10 * exp(1.041*vlntemp + 457.3*ortc);
    rr_f[25] = forward * mole_frac[11];
    rr_r[25] = reverse * mole_frac[0] * mole_frac[8];
    rr_f[25] *= thbctemp[7];
    rr_r[25] *= thbctemp[7];
  }
  //  26)  O2 + HCO <=> HO2 + CO
  {
    double forward = 7.58e+12 * exp(-410.0*ortc);
    double reverse = 1.198e+12 * exp(0.309*vlntemp - 3.395e+04*ortc);
    rr_f[26] = forward * mole_frac[3] * mole_frac[11];
    rr_r[26] = reverse * mole_frac[6] * mole_frac[8];
  }
  //  27)  H + HCO <=> H2 + CO
  {
    double forward = 7.34e+13;
    double reverse = 2.212e+12 * exp(0.656*vlntemp - 8.823e+04*ortc);
    rr_f[27] = forward * mole_frac[0] * mole_frac[11];
    rr_r[27] = reverse * mole_frac[1] * mole_frac[8];
  }
  //  28)  O + HCO <=> OH + CO
  {
    double forward = 3.02e+13;
    double reverse = 4.725e+11 * exp(0.638*vlntemp - 8.682e+04*ortc);
    rr_f[28] = forward * mole_frac[2] * mole_frac[11];
    rr_r[28] = reverse * mole_frac[4] * mole_frac[8];
  }
  //  29)  O + HCO <=> H + CO2
  {
    double forward = 3.0e+13;
    double reverse = 1.241e+18 * exp(-0.553*vlntemp - 1.122e+05*ortc);
    rr_f[29] = forward * mole_frac[2] * mole_frac[11];
    rr_r[29] = reverse * mole_frac[0] * mole_frac[9];
  }
  //  30)  OH + HCO <=> H2O + CO
  {
    double forward = 1.02e+14;
    double reverse = 3.259e+13 * exp(0.551*vlntemp - 1.031e+05*ortc);
    rr_f[30] = forward * mole_frac[4] * mole_frac[11];
    rr_r[30] = reverse * mole_frac[5] * mole_frac[8];
  }
  //  31)  HCO + CH3 <=> CO + CH4
  {
    double forward = 2.65e+13;
    double reverse = 7.286e+14 * exp(0.211*vlntemp - 8.977e+04*ortc);
    rr_f[31] = forward * mole_frac[11] * mole_frac[20];
    rr_r[31] = reverse * mole_frac[8] * mole_frac[19];
  }
  //  32)  HO2 + HCO <=> O2 + CH2O
  {
    double forward = 2.499e+14 * exp(-0.061*vlntemp - 1.392e+04*ortc);
    double reverse = 8.07e+15 * exp(-5.342e+04*ortc);
    rr_f[32] = forward * mole_frac[6] * mole_frac[11];
    rr_r[32] = reverse * mole_frac[3] * mole_frac[10];
  }
  //  33)  HO2 + HCO <=> H + OH + CO2
  {
    double forward = 3.0e+13;
    double reverse = 0.0;
    rr_f[33] = forward * mole_frac[6] * mole_frac[11];
    rr_r[33] = reverse * mole_frac[0] * mole_frac[4] * mole_frac[9];
  }
  //  34)  O2CHO <=> O2 + HCO
  {
    double forward = 9.959e+15 * exp(-1.126*vlntemp - 4.1e+04*ortc);
    double reverse = 1.2e+11 * exp(1.1e+03*ortc);
    rr_f[34] = forward * mole_frac[13];
    rr_r[34] = reverse * mole_frac[3] * mole_frac[11];
  }
  //  35)  CH2O + O2CHO <=> HCO + HO2CHO
  {
    double forward = 1.99e+12 * exp(-1.166e+04*ortc);
    double reverse = 3.908e+14 * exp(-0.909*vlntemp - 1.181e+04*ortc);
    rr_f[35] = forward * mole_frac[10] * mole_frac[13];
    rr_r[35] = reverse * mole_frac[11] * mole_frac[12];
  }
  //  36)  HO2CHO <=> OH + OCHO
  {
    double forward = 5.01e+14 * exp(-4.015e+04*ortc);
    double reverse = 3.856e+08 * exp(1.532*vlntemp + 6.372e+03*ortc);
    rr_f[36] = forward * mole_frac[12];
    rr_r[36] = reverse * mole_frac[4] * mole_frac[14];
  }
  //  37)  OCHO + M <=> H + CO2 + M
  {
    double forward = 5.318e+14 * exp(-0.353*vlntemp - 1.758e+04*ortc);
    double reverse = 7.5e+13 * exp(-2.9e+04*ortc);
    rr_f[37] = forward * mole_frac[14];
    rr_r[37] = reverse * mole_frac[0] * mole_frac[9];
    rr_f[37] *= thbctemp[0];
    rr_r[37] *= thbctemp[0];
  }
  //  38)  CO + CH2O <=> 2 HCO
  {
    double forward = 9.186e+13 * exp(0.37*vlntemp - 7.304e+04*ortc);
    double reverse = 1.8e+13;
    rr_f[38] = forward * mole_frac[8] * mole_frac[10];
    rr_r[38] = reverse * mole_frac[11] * mole_frac[11];
  }
  //  39)  2 HCO <=> H2 + 2 CO
  {
    double forward = 3.0e+12;
    double reverse = 0.0;
    rr_f[39] = forward * mole_frac[11] * mole_frac[11];
    rr_r[39] = reverse * mole_frac[1] * mole_frac[8] * mole_frac[8];
  }
  //  40)  H + HCO (+M) <=> CH2O (+M)
  {
    double rr_k0 = 1.35e+24 * exp(-2.57*vlntemp - 1.425e+03*ortc);
    double rr_kinf = 1.09e+12 * exp(0.48*vlntemp + 260.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.2176 * exp(-3.690036900369004e-03 * temperature) 
      + 0.7824 * exp(-3.629764065335753e-04 * temperature) + exp(-6.57e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] + cgspl[10] - cgspl[11];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[40] = forward * mole_frac[0] * mole_frac[11];
    rr_r[40] = reverse * mole_frac[10];
  }
  //  41)  H2 + CO (+M) <=> CH2O (+M)
  {
    double rr_k0 = 5.07e+27 * exp(-3.42*vlntemp - 8.4348e+04*ortc);
    double rr_kinf = 4.3e+07 * exp(1.5*vlntemp - 7.96e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.06799999999999995 * exp(-5.076142131979695e-03 * 
      temperature) + 0.9320000000000001 * exp(-6.493506493506494e-04 * 
      temperature) + exp(-1.03e+04 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[8] + cgspl[10];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[41] = forward * mole_frac[1] * mole_frac[8];
    rr_r[41] = reverse * mole_frac[10];
  }
  //  42)  OH + CH2O <=> H2O + HCO
  {
    double forward = 7.82e+07 * exp(1.63*vlntemp + 1.055e+03*ortc);
    double reverse = 4.896e+06 * exp(1.811*vlntemp - 2.903e+04*ortc);
    rr_f[42] = forward * mole_frac[4] * mole_frac[10];
    rr_r[42] = reverse * mole_frac[5] * mole_frac[11];
  }
  //  43)  H + CH2O <=> H2 + HCO
  {
    double forward = 5.74e+07 * exp(1.9*vlntemp - 2.74e+03*ortc);
    double reverse = 3.39e+05 * exp(2.187*vlntemp - 1.793e+04*ortc);
    rr_f[43] = forward * mole_frac[0] * mole_frac[10];
    rr_r[43] = reverse * mole_frac[1] * mole_frac[11];
  }
  //  44)  O + CH2O <=> OH + HCO
  {
    double forward = 6.26e+09 * exp(1.15*vlntemp - 2.26e+03*ortc);
    double reverse = 1.919e+07 * exp(1.418*vlntemp - 1.604e+04*ortc);
    rr_f[44] = forward * mole_frac[2] * mole_frac[10];
    rr_r[44] = reverse * mole_frac[4] * mole_frac[11];
  }
  //  45)  CH2O + CH3 <=> HCO + CH4
  {
    double forward = 38.3 * exp(3.36*vlntemp - 4.312e+03*ortc);
    double reverse = 206.3 * exp(3.201*vlntemp - 2.104e+04*ortc);
    rr_f[45] = forward * mole_frac[10] * mole_frac[20];
    rr_r[45] = reverse * mole_frac[11] * mole_frac[19];
  }
  //  46)  HO2 + CH2O <=> H2O2 + HCO
  {
    double forward = 7.1e-03 * exp(4.517*vlntemp - 6.58e+03*ortc);
    double reverse = 0.02426 * exp(4.108*vlntemp - 5.769e+03*ortc);
    rr_f[46] = forward * mole_frac[6] * mole_frac[10];
    rr_r[46] = reverse * mole_frac[7] * mole_frac[11];
  }
  //  47)  CH3O (+M) <=> H + CH2O (+M)
  {
    double rr_k0 = 1.867e+25 * exp(-3.0*vlntemp - 2.4307e+04*ortc);
    double rr_kinf = 6.8e+13 * exp(-2.617e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.09999999999999998 * exp(-4.0e-04 * temperature) + 
      0.9 * exp(-7.692307692307692e-04 * temperature) + exp(-1.0e+99 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = cgspl[0] + cgspl[10] - cgspl[16];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[47] = forward * mole_frac[16];
    rr_r[47] = reverse * mole_frac[0] * mole_frac[10];
  }
  //  48)  O2 + CH3O <=> HO2 + CH2O
  {
    double forward = 4.38e-19 * exp(9.5*vlntemp + 5.501e+03*ortc);
    double reverse = 1.416e-20 * exp(9.816000000000001*vlntemp - 
      2.108e+04*ortc); 
    rr_f[48] = forward * mole_frac[3] * mole_frac[16];
    rr_r[48] = reverse * mole_frac[6] * mole_frac[10];
  }
  //  49)  CH3O + CH3 <=> CH2O + CH4
  {
    double forward = 1.2e+13;
    double reverse = 6.749e+13 * exp(0.218*vlntemp - 8.281e+04*ortc);
    rr_f[49] = forward * mole_frac[16] * mole_frac[20];
    rr_r[49] = reverse * mole_frac[10] * mole_frac[19];
  }
  //  50)  H + CH3O <=> H2 + CH2O
  {
    double forward = 2.0e+13;
    double reverse = 1.233e+11 * exp(0.664*vlntemp - 8.127e+04*ortc);
    rr_f[50] = forward * mole_frac[0] * mole_frac[16];
    rr_r[50] = reverse * mole_frac[1] * mole_frac[10];
  }
  //  51)  HO2 + CH3O <=> H2O2 + CH2O
  {
    double forward = 3.01e+11;
    double reverse = 1.074e+12 * exp(-0.031*vlntemp - 6.527e+04*ortc);
    rr_f[51] = forward * mole_frac[6] * mole_frac[16];
    rr_r[51] = reverse * mole_frac[7] * mole_frac[10];
  }
  //  52)  H + CH2O (+M) <=> CH2OH (+M)
  {
    double rr_k0 = 1.27e+32 * exp(-4.82*vlntemp - 6.53e+03*ortc);
    double rr_kinf = 5.4e+11 * exp(0.454*vlntemp - 3.6e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.2813 * exp(-9.708737864077669e-03 * temperature) 
      + 0.7187 * exp(-7.74593338497289e-04 * temperature) + exp(-4.16e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] - cgspl[10] + cgspl[15];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[52] = forward * mole_frac[0] * mole_frac[10];
    rr_r[52] = reverse * mole_frac[15];
  }
  //  53)  O2 + CH2OH <=> HO2 + CH2O
  {
    double forward = 1.51e+15 * otc;
    double reverse = 1.975e+14 * exp(-0.58*vlntemp - 2.006e+04*ortc);
    rr_f[53] = forward * mole_frac[3] * mole_frac[15];
    rr_r[53] = reverse * mole_frac[6] * mole_frac[10];
  }
  //  54)  O2 + CH2OH <=> HO2 + CH2O
  {
    double forward = 2.41e+14 * exp(-5.017e+03*ortc);
    double reverse = 3.152e+13 * exp(0.42*vlntemp - 2.508e+04*ortc);
    rr_f[54] = forward * mole_frac[3] * mole_frac[15];
    rr_r[54] = reverse * mole_frac[6] * mole_frac[10];
  }
  //  55)  H + CH2OH <=> H2 + CH2O
  {
    double forward = 6.0e+12;
    double reverse = 1.497e+11 * exp(0.768*vlntemp - 7.475e+04*ortc);
    rr_f[55] = forward * mole_frac[0] * mole_frac[15];
    rr_r[55] = reverse * mole_frac[1] * mole_frac[10];
  }
  //  56)  HO2 + CH2OH <=> H2O2 + CH2O
  {
    double forward = 1.2e+13;
    double reverse = 1.732e+14 * exp(0.073*vlntemp - 5.875e+04*ortc);
    rr_f[56] = forward * mole_frac[6] * mole_frac[15];
    rr_r[56] = reverse * mole_frac[7] * mole_frac[10];
  }
  //  57)  HCO + CH2OH <=> 2 CH2O
  {
    double forward = 1.8e+14;
    double reverse = 7.602e+14 * exp(0.481*vlntemp - 5.956e+04*ortc);
    rr_f[57] = forward * mole_frac[11] * mole_frac[15];
    rr_r[57] = reverse * mole_frac[10] * mole_frac[10];
  }
  //  58)  OH + CH2OH <=> H2O + CH2O
  {
    double forward = 2.4e+13;
    double reverse = 6.347e+12 * exp(0.662*vlntemp - 8.964e+04*ortc);
    rr_f[58] = forward * mole_frac[4] * mole_frac[15];
    rr_r[58] = reverse * mole_frac[5] * mole_frac[10];
  }
  //  59)  O + CH2OH <=> OH + CH2O
  {
    double forward = 4.2e+13;
    double reverse = 5.438e+11 * exp(0.749*vlntemp - 7.334e+04*ortc);
    rr_f[59] = forward * mole_frac[2] * mole_frac[15];
    rr_r[59] = reverse * mole_frac[4] * mole_frac[10];
  }
  //  60)  H + CH3 (+M) <=> CH4 (+M)
  {
    double rr_k0 = 2.477e+33 * exp(-4.76*vlntemp - 2.444e+03*ortc);
    double rr_kinf = 1.27e+16 * exp(-0.6*vlntemp - 383.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.217 * exp(-0.01351351351351351 * temperature) + 
      0.783 * exp(-3.401360544217687e-04 * temperature) + exp(-6.96e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] + cgspl[19] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[60] = forward * mole_frac[0] * mole_frac[20];
    rr_r[60] = reverse * mole_frac[19];
  }
  //  61)  H + CH4 <=> H2 + CH3
  {
    double forward = 6.14e+05 * exp(2.5*vlntemp - 9.587e+03*ortc);
    double reverse = 673.0 * exp(2.946*vlntemp - 8.047e+03*ortc);
    rr_f[61] = forward * mole_frac[0] * mole_frac[19];
    rr_r[61] = reverse * mole_frac[1] * mole_frac[20];
  }
  //  62)  OH + CH4 <=> H2O + CH3
  {
    double forward = 5.83e+04 * exp(2.6*vlntemp - 2.19e+03*ortc);
    double reverse = 677.6 * exp(2.94*vlntemp - 1.554e+04*ortc);
    rr_f[62] = forward * mole_frac[4] * mole_frac[19];
    rr_r[62] = reverse * mole_frac[5] * mole_frac[20];
  }
  //  63)  O + CH4 <=> OH + CH3
  {
    double forward = 1.02e+09 * exp(1.5*vlntemp - 8.6e+03*ortc);
    double reverse = 5.804e+05 * exp(1.927*vlntemp - 5.648e+03*ortc);
    rr_f[63] = forward * mole_frac[2] * mole_frac[19];
    rr_r[63] = reverse * mole_frac[4] * mole_frac[20];
  }
  //  64)  HO2 + CH4 <=> H2O2 + CH3
  {
    double forward = 11.3 * exp(3.74*vlntemp - 2.101e+04*ortc);
    double reverse = 7.166 * exp(3.491*vlntemp - 3.468e+03*ortc);
    rr_f[64] = forward * mole_frac[6] * mole_frac[19];
    rr_r[64] = reverse * mole_frac[7] * mole_frac[20];
  }
  //  65)  CH4 + CH2 <=> 2 CH3
  {
    double forward = 2.46e+06 * temperature * temperature * exp(-8.27e+03*ortc);
    double reverse = 1.736e+06 * exp(1.868*vlntemp - 1.298e+04*ortc);
    rr_f[65] = forward * mole_frac[19] * mole_frac[21];
    rr_r[65] = reverse * mole_frac[20] * mole_frac[20];
  }
  //  66)  OH + CH3 <=> H2 + CH2O
  {
    double forward = 8.0e+09 * exp(0.5*vlntemp + 1.755e+03*ortc);
    double reverse = 1.066e+12 * exp(0.322*vlntemp - 6.821e+04*ortc);
    rr_f[66] = forward * mole_frac[4] * mole_frac[20];
    rr_r[66] = reverse * mole_frac[1] * mole_frac[10];
  }
  //  67)  OH + CH3 <=> H2O + CH2(S)
  {
    double forward = 4.508e+17 * exp(-1.34*vlntemp - 1.417e+03*ortc);
    double reverse = 1.654e+16 * exp(-0.855*vlntemp - 1.039e+03*ortc);
    rr_f[67] = forward * mole_frac[4] * mole_frac[20];
    rr_r[67] = reverse * mole_frac[5] * mole_frac[22];
  }
  //  68)  OH + CH3 <=> H + CH3O
  {
    double forward = 6.943e+07 * exp(1.343*vlntemp - 1.12e+04*ortc);
    double reverse = 1.5e+12 * exp(0.5*vlntemp + 110.0*ortc);
    rr_f[68] = forward * mole_frac[4] * mole_frac[20];
    rr_r[68] = reverse * mole_frac[0] * mole_frac[16];
  }
  //  69)  OH + CH3 <=> H + CH2OH
  {
    double forward = 3.09e+07 * exp(1.596*vlntemp - 4.506e+03*ortc);
    double reverse = 1.65e+11 * exp(0.65*vlntemp + 284.0*ortc);
    rr_f[69] = forward * mole_frac[4] * mole_frac[20];
    rr_r[69] = reverse * mole_frac[0] * mole_frac[15];
  }
  //  70)  OH + CH3 <=> H2O + CH2
  {
    double forward = 5.6e+07 * exp(1.6*vlntemp - 5.42e+03*ortc);
    double reverse = 9.224e+05 * exp(2.072*vlntemp - 1.406e+04*ortc);
    rr_f[70] = forward * mole_frac[4] * mole_frac[20];
    rr_r[70] = reverse * mole_frac[5] * mole_frac[21];
  }
  //  71)  HO2 + CH3 <=> OH + CH3O
  {
    double forward = 1.0e+12 * exp(0.269*vlntemp + 687.5*ortc);
    double reverse = 6.19e+12 * exp(0.147*vlntemp - 2.455e+04*ortc);
    rr_f[71] = forward * mole_frac[6] * mole_frac[20];
    rr_r[71] = reverse * mole_frac[4] * mole_frac[16];
  }
  //  72)  HO2 + CH3 <=> O2 + CH4
  {
    double forward = 1.16e+05 * exp(2.23*vlntemp + 3.022e+03*ortc);
    double reverse = 2.018e+07 * exp(2.132*vlntemp - 5.321e+04*ortc);
    rr_f[72] = forward * mole_frac[6] * mole_frac[20];
    rr_r[72] = reverse * mole_frac[3] * mole_frac[19];
  }
  //  73)  O + CH3 <=> H + CH2O
  {
    double forward = 5.54e+13 * exp(0.05*vlntemp + 136.0*ortc);
    double reverse = 3.83e+15 * exp(-0.147*vlntemp - 6.841e+04*ortc);
    rr_f[73] = forward * mole_frac[2] * mole_frac[20];
    rr_r[73] = reverse * mole_frac[0] * mole_frac[10];
  }
  //  74)  O2 + CH3 <=> O + CH3O
  {
    double forward = 7.546e+12 * exp(-2.832e+04*ortc);
    double reverse = 4.718e+14 * exp(-0.451*vlntemp - 288.0*ortc);
    rr_f[74] = forward * mole_frac[3] * mole_frac[20];
    rr_r[74] = reverse * mole_frac[2] * mole_frac[16];
  }
  //  75)  O2 + CH3 <=> OH + CH2O
  {
    double forward = 2.641 * exp(3.283*vlntemp - 8.105e+03*ortc);
    double reverse = 0.5285 * exp(3.477*vlntemp - 5.992e+04*ortc);
    rr_f[75] = forward * mole_frac[3] * mole_frac[20];
    rr_r[75] = reverse * mole_frac[4] * mole_frac[10];
  }
  //  76)  O2 + CH3 (+M) <=> CH3O2 (+M)
  {
    double rr_k0 = 6.85e+24 * exp(-3.0 * vlntemp);
    double rr_kinf = 7.812e+09 * exp(0.9 * vlntemp);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.4 * exp(-1.0e-03 * temperature) + 0.6 * 
      exp(-0.01428571428571429 * temperature) + exp(-1.7e+03 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[3] + cgspl[18] - cgspl[20];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[76] = forward * mole_frac[3] * mole_frac[20];
    rr_r[76] = reverse * mole_frac[18];
  }
  //  77)  CH2O + CH3O2 <=> HCO + CH3O2H
  {
    double forward = 1.99e+12 * exp(-1.166e+04*ortc);
    double reverse = 1.323e+14 * exp(-0.853*vlntemp - 9.259e+03*ortc);
    rr_f[77] = forward * mole_frac[10] * mole_frac[18];
    rr_r[77] = reverse * mole_frac[11] * mole_frac[17];
  }
  //  78)  CH3O2 + CH4 <=> CH3O2H + CH3
  {
    double forward = 1.81e+11 * exp(-1.848e+04*ortc);
    double reverse = 2.233e+12 * exp(-0.694*vlntemp + 655.0*ortc);
    rr_f[78] = forward * mole_frac[18] * mole_frac[19];
    rr_r[78] = reverse * mole_frac[17] * mole_frac[20];
  }
  //  79)  CH3O2 + CH3 <=> 2 CH3O
  {
    double forward = 5.08e+12 * exp(1.411e+03*ortc);
    double reverse = 1.967e+12 * exp(0.176*vlntemp - 2.807e+04*ortc);
    rr_f[79] = forward * mole_frac[18] * mole_frac[20];
    rr_r[79] = reverse * mole_frac[16] * mole_frac[16];
  }
  //  80)  HO2 + CH3O2 <=> O2 + CH3O2H
  {
    double forward = 2.47e+11 * exp(1.57e+03*ortc);
    double reverse = 5.302e+14 * exp(-0.792*vlntemp - 3.552e+04*ortc);
    rr_f[80] = forward * mole_frac[6] * mole_frac[18];
    rr_r[80] = reverse * mole_frac[3] * mole_frac[17];
  }
  //  81)  2 CH3O2 <=> O2 + 2 CH3O
  {
    double forward = 1.4e+16 * exp(-1.61*vlntemp - 1.86e+03*ortc);
    double reverse = 0.0;
    rr_f[81] = forward * mole_frac[18] * mole_frac[18];
    rr_r[81] = reverse * mole_frac[3] * mole_frac[16] * mole_frac[16];
  }
  //  82)  H + CH3O2 <=> OH + CH3O
  {
    double forward = 9.6e+13;
    double reverse = 1.72e+09 * exp(1.019*vlntemp - 4.078e+04*ortc);
    rr_f[82] = forward * mole_frac[0] * mole_frac[18];
    rr_r[82] = reverse * mole_frac[4] * mole_frac[16];
  }
  //  83)  O + CH3O2 <=> O2 + CH3O
  {
    double forward = 3.6e+13;
    double reverse = 2.229e+11 * exp(0.628*vlntemp - 5.752e+04*ortc);
    rr_f[83] = forward * mole_frac[2] * mole_frac[18];
    rr_r[83] = reverse * mole_frac[3] * mole_frac[16];
  }
  //  84)  CH3O2H <=> OH + CH3O
  {
    double forward = 6.31e+14 * exp(-4.23e+04*ortc);
    double reverse = 2.514e+06 * exp(1.883*vlntemp + 2.875e+03*ortc);
    rr_f[84] = forward * mole_frac[17];
    rr_r[84] = reverse * mole_frac[4] * mole_frac[16];
  }
  //  85)  CH2(S) <=> CH2
  {
    double forward = 1.0e+13;
    double reverse = 4.488e+12 * exp(-0.013*vlntemp - 9.02e+03*ortc);
    rr_f[85] = forward * mole_frac[22];
    rr_r[85] = reverse * mole_frac[21];
  }
  //  86)  CH4 + CH2(S) <=> 2 CH3
  {
    double forward = 1.6e+13 * exp(570.0*ortc);
    double reverse = 5.067e+12 * exp(-0.145*vlntemp - 1.316e+04*ortc);
    rr_f[86] = forward * mole_frac[19] * mole_frac[22];
    rr_r[86] = reverse * mole_frac[20] * mole_frac[20];
  }
  //  87)  O2 + CH2(S) <=> H + OH + CO
  {
    double forward = 7.0e+13;
    double reverse = 0.0;
    rr_f[87] = forward * mole_frac[3] * mole_frac[22];
    rr_r[87] = reverse * mole_frac[0] * mole_frac[4] * mole_frac[8];
  }
  //  88)  H2 + CH2(S) <=> H + CH3
  {
    double forward = 7.0e+13;
    double reverse = 2.022e+16 * exp(-0.591*vlntemp - 1.527e+04*ortc);
    rr_f[88] = forward * mole_frac[1] * mole_frac[22];
    rr_r[88] = reverse * mole_frac[0] * mole_frac[20];
  }
  //  89)  H + CH2(S) <=> H + CH2
  {
    double forward = 3.0e+13;
    double reverse = 1.346e+13 * exp(-0.013*vlntemp - 9.02e+03*ortc);
    rr_f[89] = forward * mole_frac[0] * mole_frac[22];
    rr_r[89] = reverse * mole_frac[0] * mole_frac[21];
  }
  //  90)  O + CH2(S) <=> 2 H + CO
  {
    double forward = 3.0e+13;
    double reverse = 0.0;
    rr_f[90] = forward * mole_frac[2] * mole_frac[22];
    rr_r[90] = reverse * mole_frac[0] * mole_frac[0] * mole_frac[8];
  }
  //  91)  OH + CH2(S) <=> H + CH2O
  {
    double forward = 3.0e+13;
    double reverse = 1.154e+18 * exp(-0.77*vlntemp - 8.523e+04*ortc);
    rr_f[91] = forward * mole_frac[4] * mole_frac[22];
    rr_r[91] = reverse * mole_frac[0] * mole_frac[10];
  }
  //  92)  CO2 + CH2(S) <=> CO + CH2O
  {
    double forward = 3.0e+12;
    double reverse = 4.366e+10 * exp(0.421*vlntemp - 5.981e+04*ortc);
    rr_f[92] = forward * mole_frac[9] * mole_frac[22];
    rr_r[92] = reverse * mole_frac[8] * mole_frac[10];
  }
  //  93)  H + CH2 (+M) <=> CH3 (+M)
  {
    double rr_k0 = 3.2e+27 * exp(-3.14*vlntemp - 1.23e+03*ortc);
    double rr_kinf = 2.5e+16 * exp(-0.8 * vlntemp);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.32 * exp(-0.01282051282051282 * temperature) + 
      0.68 * exp(-5.012531328320802e-04 * temperature) + exp(-5.59e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] + cgspl[20] - cgspl[21];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[93] = forward * mole_frac[0] * mole_frac[21];
    rr_r[93] = reverse * mole_frac[20];
  }
  //  94)  O2 + CH2 <=> O + CH2O
  {
    double forward = 2.4e+12 * exp(-1.5e+03*ortc);
    double reverse = 5.955e+14 * exp(-0.365*vlntemp - 6.098e+04*ortc);
    rr_f[94] = forward * mole_frac[3] * mole_frac[21];
    rr_r[94] = reverse * mole_frac[2] * mole_frac[10];
  }
  //  95)  O2 + CH2 <=> 2 H + CO2
  {
    double forward = 5.8e+12 * exp(-1.5e+03*ortc);
    double reverse = 0.0;
    rr_f[95] = forward * mole_frac[3] * mole_frac[21];
    rr_r[95] = reverse * mole_frac[0] * mole_frac[0] * mole_frac[9];
  }
  //  96)  O2 + CH2 <=> H + OH + CO
  {
    double forward = 5.0e+12 * exp(-1.5e+03*ortc);
    double reverse = 0.0;
    rr_f[96] = forward * mole_frac[3] * mole_frac[21];
    rr_r[96] = reverse * mole_frac[0] * mole_frac[4] * mole_frac[8];
  }
  //  97)  O + CH2 <=> 2 H + CO
  {
    double forward = 5.0e+13;
    double reverse = 0.0;
    rr_f[97] = forward * mole_frac[2] * mole_frac[21];
    rr_r[97] = reverse * mole_frac[0] * mole_frac[0] * mole_frac[8];
  }
  //  98)  2 CH3 (+M) <=> C2H6 (+M)
  {
    double rr_k0 = 1.135e+36 * exp(-5.246*vlntemp - 1.705e+03*ortc);
    double rr_kinf = 9.214e+16 * exp(-1.17*vlntemp - 635.8*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.595 * exp(-8.928571428571428e-04 * temperature) + 
      0.405 * exp(-0.01436781609195402 * temperature) + exp(-1.0e+10 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -2.0 * cgspl[20] + cgspl[23];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[98] = forward * mole_frac[20] * mole_frac[20];
    rr_r[98] = reverse * mole_frac[23];
  }
  //  99)  H + C2H5 (+M) <=> C2H6 (+M)
  {
    double rr_k0 = 1.99e+41 * exp(-7.08*vlntemp - 6.685e+03*ortc);
    double rr_kinf = 5.21e+17 * exp(-0.99*vlntemp - 1.58e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.158 * exp(-8.0e-03 * temperature) + 0.842 * 
      exp(-4.506534474988734e-04 * temperature) + exp(-6.882e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] + cgspl[23] - cgspl[24];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[99] = forward * mole_frac[0] * mole_frac[24];
    rr_r[99] = reverse * mole_frac[23];
  }
  // 100)  H + C2H6 <=> H2 + C2H5
  {
    double forward = 1.15e+08 * exp(1.9*vlntemp - 7.53e+03*ortc);
    double reverse = 1.062e+04 * exp(2.582*vlntemp - 9.76e+03*ortc);
    rr_f[100] = forward * mole_frac[0] * mole_frac[23];
    rr_r[100] = reverse * mole_frac[1] * mole_frac[24];
  }
  // 101)  O + C2H6 <=> OH + C2H5
  {
    double forward = 3.55e+06 * exp(2.4*vlntemp - 5.83e+03*ortc);
    double reverse = 170.2 * exp(3.063*vlntemp - 6.648e+03*ortc);
    rr_f[101] = forward * mole_frac[2] * mole_frac[23];
    rr_r[101] = reverse * mole_frac[4] * mole_frac[24];
  }
  // 102)  OH + C2H6 <=> H2O + C2H5
  {
    double forward = 1.48e+07 * exp(1.9*vlntemp - 950.0*ortc);
    double reverse = 1.45e+04 * exp(2.476*vlntemp - 1.807e+04*ortc);
    rr_f[102] = forward * mole_frac[4] * mole_frac[23];
    rr_r[102] = reverse * mole_frac[5] * mole_frac[24];
  }
  // 103)  O2 + C2H6 <=> HO2 + C2H5
  {
    double forward = 6.03e+13 * exp(-5.187e+04*ortc);
    double reverse = 2.921e+10 * exp(0.334*vlntemp + 593.0*ortc);
    rr_f[103] = forward * mole_frac[3] * mole_frac[23];
    rr_r[103] = reverse * mole_frac[6] * mole_frac[24];
  }
  // 104)  CH3 + C2H6 <=> CH4 + C2H5
  {
    double forward = 1.51e-07 * exp(6.0*vlntemp - 6.047e+03*ortc);
    double reverse = 1.273e-08 * exp(6.236*vlntemp - 9.817e+03*ortc);
    rr_f[104] = forward * mole_frac[20] * mole_frac[23];
    rr_r[104] = reverse * mole_frac[19] * mole_frac[24];
  }
  // 105)  HO2 + C2H6 <=> H2O2 + C2H5
  {
    double forward = 34.6 * exp(3.61*vlntemp - 1.692e+04*ortc);
    double reverse = 1.849 * exp(3.597*vlntemp - 3.151e+03*ortc);
    rr_f[105] = forward * mole_frac[6] * mole_frac[23];
    rr_r[105] = reverse * mole_frac[7] * mole_frac[24];
  }
  // 106)  CH3O2 + C2H6 <=> CH3O2H + C2H5
  {
    double forward = 19.4 * exp(3.64*vlntemp - 1.71e+04*ortc);
    double reverse = 20.17 * exp(3.182*vlntemp - 1.734e+03*ortc);
    rr_f[106] = forward * mole_frac[18] * mole_frac[23];
    rr_r[106] = reverse * mole_frac[17] * mole_frac[24];
  }
  // 107)  CH2(S) + C2H6 <=> CH3 + C2H5
  {
    double forward = 1.2e+14;
    double reverse = 3.203e+12 * exp(0.091*vlntemp - 1.75e+04*ortc);
    rr_f[107] = forward * mole_frac[22] * mole_frac[23];
    rr_r[107] = reverse * mole_frac[20] * mole_frac[24];
  }
  // 108)  H + C2H4 (+M) <=> C2H5 (+M)
  {
    double rr_k0 = 1.2e+42 * exp(-7.62*vlntemp - 6.97e+03*ortc);
    double rr_kinf = 1.081e+12 * exp(0.454*vlntemp - 1.822e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.02500000000000002 * exp(-4.761904761904762e-03 * 
      temperature) + 0.975 * exp(-1.016260162601626e-03 * temperature) + 
      exp(-4.374e+03 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] + cgspl[24] - cgspl[25];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[108] = forward * mole_frac[0] * mole_frac[25];
    rr_r[108] = reverse * mole_frac[24];
  }
  // 109)  H2 + CH3O2 <=> H + CH3O2H
  {
    double forward = 1.5e+14 * exp(-2.603e+04*ortc);
    double reverse = 1.688e+18 * exp(-1.14*vlntemp - 8.434e+03*ortc);
    rr_f[109] = forward * mole_frac[1] * mole_frac[18];
    rr_r[109] = reverse * mole_frac[0] * mole_frac[17];
  }
  // 110)  C2H5 + C2H3 <=> 2 C2H4
  {
    double forward = 6.859e+11 * exp(0.11*vlntemp + 4.3e+03*ortc);
    double reverse = 4.82e+14 * exp(-7.153e+04*ortc);
    rr_f[110] = forward * mole_frac[24] * mole_frac[26];
    rr_r[110] = reverse * mole_frac[25] * mole_frac[25];
  }
  // 111)  CH3 + C2H5 <=> CH4 + C2H4
  {
    double forward = 1.18e+04 * exp(2.45*vlntemp + 2.921e+03*ortc);
    double reverse = 2.39e+06 * exp(2.4*vlntemp - 6.669e+04*ortc);
    rr_f[111] = forward * mole_frac[20] * mole_frac[24];
    rr_r[111] = reverse * mole_frac[19] * mole_frac[25];
  }
  // 112)  H + C2H5 <=> 2 CH3
  {
    double forward = 9.69e+13 * exp(-220.0*ortc);
    double reverse = 2.029e+09 * exp(1.028*vlntemp - 1.051e+04*ortc);
    rr_f[112] = forward * mole_frac[0] * mole_frac[24];
    rr_r[112] = reverse * mole_frac[20] * mole_frac[20];
  }
  // 113)  H + C2H5 <=> H2 + C2H4
  {
    double forward = 2.0e+12;
    double reverse = 4.44e+11 * exp(0.396*vlntemp - 6.807e+04*ortc);
    rr_f[113] = forward * mole_frac[0] * mole_frac[24];
    rr_r[113] = reverse * mole_frac[1] * mole_frac[25];
  }
  // 114)  O + C2H5 <=> H + CH3CHO
  {
    double forward = 1.1e+14;
    double reverse = 1.033e+17 * exp(-0.5*vlntemp - 7.742e+04*ortc);
    rr_f[114] = forward * mole_frac[2] * mole_frac[24];
    rr_r[114] = reverse * mole_frac[0] * mole_frac[28];
  }
  // 115)  HO2 + C2H5 <=> OH + C2H5O
  {
    double forward = 1.1e+13;
    double reverse = 9.68e+15 * exp(-0.723*vlntemp - 2.765e+04*ortc);
    rr_f[115] = forward * mole_frac[6] * mole_frac[24];
    rr_r[115] = reverse * mole_frac[4] * mole_frac[32];
  }
  // 116)  CH3O2 + C2H5 <=> CH3O + C2H5O
  {
    double forward = 8.0e+12 * exp(1.0e+03*ortc);
    double reverse = 4.404e+14 * exp(-0.425*vlntemp - 3.089e+04*ortc);
    rr_f[116] = forward * mole_frac[18] * mole_frac[24];
    rr_r[116] = reverse * mole_frac[16] * mole_frac[32];
  }
  // 117)  O2 + C2H5O <=> HO2 + CH3CHO
  {
    double forward = 4.28e+10 * exp(-1.097e+03*ortc);
    double reverse = 1.322e+08 * exp(0.615*vlntemp - 3.413e+04*ortc);
    rr_f[117] = forward * mole_frac[3] * mole_frac[32];
    rr_r[117] = reverse * mole_frac[6] * mole_frac[28];
  }
  // 118)  C2H5O <=> CH2O + CH3
  {
    double forward = 1.321e+20 * exp(-2.018*vlntemp - 2.075e+04*ortc);
    double reverse = 3.0e+11 * exp(-6.336e+03*ortc);
    rr_f[118] = forward * mole_frac[32];
    rr_r[118] = reverse * mole_frac[10] * mole_frac[20];
  }
  // 119)  C2H5O <=> H + CH3CHO
  {
    double forward = 5.428e+15 * exp(-0.6870000000000001*vlntemp - 
      2.223e+04*ortc); 
    double reverse = 8.0e+12 * exp(-6.4e+03*ortc);
    rr_f[119] = forward * mole_frac[32];
    rr_r[119] = reverse * mole_frac[0] * mole_frac[28];
  }
  // 120)  C2H5O2 <=> O2 + C2H5
  {
    double forward = 1.312e+62 * exp(-14.784*vlntemp - 4.918e+04*ortc);
    double reverse = 2.876e+56 * exp(-13.82*vlntemp - 1.462e+04*ortc);
    rr_f[120] = forward * mole_frac[33];
    rr_r[120] = reverse * mole_frac[3] * mole_frac[24];
  }
  // 121)  O2 + C2H5 <=> HO2 + C2H4
  {
    double forward = 7.561e+14 * exp(-1.01*vlntemp - 4.749e+03*ortc);
    double reverse = 8.802e+14 * exp(-0.962*vlntemp - 1.813e+04*ortc);
    rr_f[121] = forward * mole_frac[3] * mole_frac[24];
    rr_r[121] = reverse * mole_frac[6] * mole_frac[25];
  }
  // 122)  O2 + C2H5 <=> HO2 + C2H4
  {
    double forward = 0.4 * exp(3.88*vlntemp - 1.362e+04*ortc);
    double reverse = 0.4656 * exp(3.928*vlntemp - 2.7e+04*ortc);
    rr_f[122] = forward * mole_frac[3] * mole_frac[24];
    rr_r[122] = reverse * mole_frac[6] * mole_frac[25];
  }
  // 123)  O2 + C2H5 <=> OH + CH3CHO
  {
    double forward = 826.5 * exp(2.41*vlntemp - 5.285e+03*ortc);
    double reverse = 2.247e+03 * exp(2.301*vlntemp - 6.597e+04*ortc);
    rr_f[123] = forward * mole_frac[3] * mole_frac[24];
    rr_r[123] = reverse * mole_frac[4] * mole_frac[28];
  }
  // 124)  C2H5O2 <=> OH + CH3CHO
  {
    double forward = 2.52e+41 * exp(-10.2*vlntemp - 4.371e+04*ortc);
    double reverse = 1.502e+36 * exp(-9.345000000000001*vlntemp - 
      6.984e+04*ortc); 
    rr_f[124] = forward * mole_frac[33];
    rr_r[124] = reverse * mole_frac[4] * mole_frac[28];
  }
  // 125)  C2H5O2 <=> HO2 + C2H4
  {
    double forward = 1.815e+38 * exp(-8.449999999999999*vlntemp - 
      3.789e+04*ortc); 
    double reverse = 4.632e+32 * exp(-7.438*vlntemp - 1.67e+04*ortc);
    rr_f[125] = forward * mole_frac[33];
    rr_r[125] = reverse * mole_frac[6] * mole_frac[25];
  }
  // 126)  C2H3O1-2 <=> CH3CO
  {
    double forward = 8.5e+14 * exp(-1.4e+04*ortc);
    double reverse = 1.002e+14 * exp(0.041*vlntemp - 4.871e+04*ortc);
    rr_f[126] = forward * mole_frac[34];
    rr_r[126] = reverse * mole_frac[29];
  }
  // 127)  CH3CHO <=> HCO + CH3
  {
    double forward = 7.687e+20 * exp(-1.342*vlntemp - 8.695e+04*ortc);
    double reverse = 1.75e+13;
    rr_f[127] = forward * mole_frac[28];
    rr_r[127] = reverse * mole_frac[11] * mole_frac[20];
  }
  // 128)  H + CH3CHO <=> H2 + CH3CO
  {
    double forward = 2.37e+13 * exp(-3.642e+03*ortc);
    double reverse = 1.639e+10 * exp(0.633*vlntemp - 1.76e+04*ortc);
    rr_f[128] = forward * mole_frac[0] * mole_frac[28];
    rr_r[128] = reverse * mole_frac[1] * mole_frac[29];
  }
  // 129)  O + CH3CHO <=> OH + CH3CO
  {
    double forward = 5.94e+12 * exp(-1.868e+03*ortc);
    double reverse = 2.133e+09 * exp(0.614*vlntemp - 1.441e+04*ortc);
    rr_f[129] = forward * mole_frac[2] * mole_frac[28];
    rr_r[129] = reverse * mole_frac[4] * mole_frac[29];
  }
  // 130)  OH + CH3CHO <=> H2O + CH3CO
  {
    double forward = 3.37e+12 * exp(619.0*ortc);
    double reverse = 2.472e+10 * exp(0.527*vlntemp - 2.823e+04*ortc);
    rr_f[130] = forward * mole_frac[4] * mole_frac[28];
    rr_r[130] = reverse * mole_frac[5] * mole_frac[29];
  }
  // 131)  O2 + CH3CHO <=> HO2 + CH3CO
  {
    double forward = 3.01e+13 * exp(-3.915e+04*ortc);
    double reverse = 1.092e+11 * exp(0.285*vlntemp + 1.588e+03*ortc);
    rr_f[131] = forward * mole_frac[3] * mole_frac[28];
    rr_r[131] = reverse * mole_frac[6] * mole_frac[29];
  }
  // 132)  CH3 + CH3CHO <=> CH4 + CH3CO
  {
    double forward = 7.08e-04 * exp(4.58*vlntemp - 1.966e+03*ortc);
    double reverse = 4.468e-04 * exp(4.767*vlntemp - 1.746e+04*ortc);
    rr_f[132] = forward * mole_frac[20] * mole_frac[28];
    rr_r[132] = reverse * mole_frac[19] * mole_frac[29];
  }
  // 133)  HO2 + CH3CHO <=> H2O2 + CH3CO
  {
    double forward = 3.01e+12 * exp(-1.192e+04*ortc);
    double reverse = 1.205e+12 * exp(-0.062*vlntemp - 9.877e+03*ortc);
    rr_f[133] = forward * mole_frac[6] * mole_frac[28];
    rr_r[133] = reverse * mole_frac[7] * mole_frac[29];
  }
  // 134)  CH3O2 + CH3CHO <=> CH3O2H + CH3CO
  {
    double forward = 3.01e+12 * exp(-1.192e+04*ortc);
    double reverse = 2.344e+13 * exp(-0.507*vlntemp - 8.282e+03*ortc);
    rr_f[134] = forward * mole_frac[18] * mole_frac[28];
    rr_r[134] = reverse * mole_frac[17] * mole_frac[29];
  }
  // 135)  CH3CO (+M) <=> CO + CH3 (+M)
  {
    double rr_k0 = 1.2e+15 * exp(-1.2518e+04*ortc);
    double rr_kinf = 3.0e+12 * exp(-1.672e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double forward = rr_kinf * pr/(1.0 + pr);
    double xik = cgspl[8] + cgspl[20] - cgspl[29];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[135] = forward * mole_frac[29];
    rr_r[135] = reverse * mole_frac[8] * mole_frac[20];
  }
  // 136)  H + CH3CO <=> H2 + CH2CO
  {
    double forward = 2.0e+13;
    double reverse = 1.037e+13 * exp(0.201*vlntemp - 6.056e+04*ortc);
    rr_f[136] = forward * mole_frac[0] * mole_frac[29];
    rr_r[136] = reverse * mole_frac[1] * mole_frac[30];
  }
  // 137)  O + CH3CO <=> OH + CH2CO
  {
    double forward = 2.0e+13;
    double reverse = 5.381e+12 * exp(0.182*vlntemp - 5.914e+04*ortc);
    rr_f[137] = forward * mole_frac[2] * mole_frac[29];
    rr_r[137] = reverse * mole_frac[4] * mole_frac[30];
  }
  // 138)  CH3 + CH3CO <=> CH4 + CH2CO
  {
    double forward = 5.0e+13;
    double reverse = 2.364e+16 * exp(-0.245*vlntemp - 6.21e+04*ortc);
    rr_f[138] = forward * mole_frac[20] * mole_frac[29];
    rr_r[138] = reverse * mole_frac[19] * mole_frac[30];
  }
  // 139)  CO + CH2 (+M) <=> CH2CO (+M)
  {
    double rr_k0 = 2.69e+33 * exp(-5.11*vlntemp - 7.095e+03*ortc);
    double rr_kinf = 8.1e+11;
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.4093 * exp(-3.636363636363636e-03 * temperature) 
      + 0.5907 * exp(-8.156606851549756e-04 * temperature) + exp(-5.185e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[8] - cgspl[21] + cgspl[30];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[139] = forward * mole_frac[8] * mole_frac[21];
    rr_r[139] = reverse * mole_frac[30];
  }
  // 140)  H + CH2CO <=> CO + CH3
  {
    double forward = 1.1e+13 * exp(-3.4e+03*ortc);
    double reverse = 2.4e+12 * exp(-4.02e+04*ortc);
    rr_f[140] = forward * mole_frac[0] * mole_frac[30];
    rr_r[140] = reverse * mole_frac[8] * mole_frac[20];
  }
  // 141)  H + CH2CO <=> H2 + HCCO
  {
    double forward = 2.0e+14 * exp(-8.0e+03*ortc);
    double reverse = 1.434e+11 * exp(0.47*vlntemp - 4.52e+03*ortc);
    rr_f[141] = forward * mole_frac[0] * mole_frac[30];
    rr_r[141] = reverse * mole_frac[1] * mole_frac[31];
  }
  // 142)  O + CH2CO <=> CO2 + CH2
  {
    double forward = 1.75e+12 * exp(-1.35e+03*ortc);
    double reverse = 2.854e+09 * exp(0.8090000000000001*vlntemp - 
      4.944e+04*ortc); 
    rr_f[142] = forward * mole_frac[2] * mole_frac[30];
    rr_r[142] = reverse * mole_frac[9] * mole_frac[21];
  }
  // 143)  O + CH2CO <=> OH + HCCO
  {
    double forward = 1.0e+13 * exp(-8.0e+03*ortc);
    double reverse = 3.723e+09 * exp(0.452*vlntemp - 3.108e+03*ortc);
    rr_f[143] = forward * mole_frac[2] * mole_frac[30];
    rr_r[143] = reverse * mole_frac[4] * mole_frac[31];
  }
  // 144)  OH + CH2CO <=> H2O + HCCO
  {
    double forward = 1.0e+13 * exp(-2.0e+03*ortc);
    double reverse = 7.604e+10 * exp(0.365*vlntemp - 1.341e+04*ortc);
    rr_f[144] = forward * mole_frac[4] * mole_frac[30];
    rr_r[144] = reverse * mole_frac[5] * mole_frac[31];
  }
  // 145)  OH + CH2CO <=> CO + CH2OH
  {
    double forward = 2.0e+12 * exp(1.01e+03*ortc);
    double reverse = 8.17e+09 * exp(0.494*vlntemp - 2.453e+04*ortc);
    rr_f[145] = forward * mole_frac[4] * mole_frac[30];
    rr_r[145] = reverse * mole_frac[8] * mole_frac[15];
  }
  // 146)  CH2(S) + CH2CO <=> CO + C2H4
  {
    double forward = 1.6e+14;
    double reverse = 3.75e+14 * exp(0.217*vlntemp - 1.034e+05*ortc);
    rr_f[146] = forward * mole_frac[22] * mole_frac[30];
    rr_r[146] = reverse * mole_frac[8] * mole_frac[25];
  }
  // 147)  OH + HCCO <=> H2 + 2 CO
  {
    double forward = 1.0e+14;
    double reverse = 0.0;
    rr_f[147] = forward * mole_frac[4] * mole_frac[31];
    rr_r[147] = reverse * mole_frac[1] * mole_frac[8] * mole_frac[8];
  }
  // 148)  H + HCCO <=> CO + CH2(S)
  {
    double forward = 1.1e+13;
    double reverse = 4.061e+07 * exp(1.561*vlntemp - 1.854e+04*ortc);
    rr_f[148] = forward * mole_frac[0] * mole_frac[31];
    rr_r[148] = reverse * mole_frac[8] * mole_frac[22];
  }
  // 149)  O + HCCO <=> H + 2 CO
  {
    double forward = 8.0e+13;
    double reverse = 0.0;
    rr_f[149] = forward * mole_frac[2] * mole_frac[31];
    rr_r[149] = reverse * mole_frac[0] * mole_frac[8] * mole_frac[8];
  }
  // 150)  O2 + HCCO <=> OH + 2 CO
  {
    double forward = 4.2e+10 * exp(-850.0*ortc);
    double reverse = 0.0;
    rr_f[150] = forward * mole_frac[3] * mole_frac[31];
    rr_r[150] = reverse * mole_frac[4] * mole_frac[8] * mole_frac[8];
  }
  // 151)  H + C2H3 (+M) <=> C2H4 (+M)
  {
    double rr_k0 = 1.4e+30 * exp(-3.86*vlntemp - 3.32e+03*ortc);
    double rr_kinf = 1.36e+14 * exp(0.173*vlntemp - 660.0*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.218 * exp(-4.819277108433735e-03 * temperature) + 
      0.782 * exp(-3.755163349605708e-04 * temperature) + exp(-6.095e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] + cgspl[25] - cgspl[26];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[151] = forward * mole_frac[0] * mole_frac[26];
    rr_r[151] = reverse * mole_frac[25];
  }
  // 152)  C2H4 (+M) <=> H2 + C2H2 (+M)
  {
    double rr_k0 = 1.58e+51 * exp(-9.300000000000001*vlntemp - 9.78e+04*ortc);
    double rr_kinf = 8.0e+12 * exp(0.44*vlntemp - 8.877e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.265 * exp(-5.555555555555556e-03 * temperature) + 
      0.735 * exp(-9.66183574879227e-04 * temperature) + exp(-5.417e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = cgspl[1] - cgspl[25] + cgspl[27];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[152] = forward * mole_frac[25];
    rr_r[152] = reverse * mole_frac[1] * mole_frac[27];
  }
  // 153)  H + C2H4 <=> H2 + C2H3
  {
    double forward = 5.07e+07 * exp(1.93*vlntemp - 1.295e+04*ortc);
    double reverse = 1.602e+04 * exp(2.436*vlntemp - 5.19e+03*ortc);
    rr_f[153] = forward * mole_frac[0] * mole_frac[25];
    rr_r[153] = reverse * mole_frac[1] * mole_frac[26];
  }
  // 154)  O + C2H4 <=> HCO + CH3
  {
    double forward = 8.564e+06 * exp(1.88*vlntemp - 183.0*ortc);
    double reverse = 329.7 * exp(2.602*vlntemp - 2.614e+04*ortc);
    rr_f[154] = forward * mole_frac[2] * mole_frac[25];
    rr_r[154] = reverse * mole_frac[11] * mole_frac[20];
  }
  // 155)  OH + C2H4 <=> H2O + C2H3
  {
    double forward = 1.8e+06 * temperature * temperature * exp(-2.5e+03*ortc);
    double reverse = 6.029e+03 * exp(2.4*vlntemp - 9.632e+03*ortc);
    rr_f[155] = forward * mole_frac[4] * mole_frac[25];
    rr_r[155] = reverse * mole_frac[5] * mole_frac[26];
  }
  // 156)  CH3 + C2H4 <=> CH4 + C2H3
  {
    double forward = 6.62 * exp(3.7*vlntemp - 9.5e+03*ortc);
    double reverse = 1.908 * exp(3.76*vlntemp - 3.28e+03*ortc);
    rr_f[156] = forward * mole_frac[20] * mole_frac[25];
    rr_r[156] = reverse * mole_frac[19] * mole_frac[26];
  }
  // 157)  O2 + C2H4 <=> HO2 + C2H3
  {
    double forward = 4.0e+13 * exp(-5.82e+04*ortc);
    double reverse = 6.626e+10 * exp(0.158*vlntemp + 4.249e+03*ortc);
    rr_f[157] = forward * mole_frac[3] * mole_frac[25];
    rr_r[157] = reverse * mole_frac[6] * mole_frac[26];
  }
  // 158)  CH3O2 + C2H4 <=> CH3O2H + C2H3
  {
    double forward = 2.23e+12 * exp(-1.719e+04*ortc);
    double reverse = 7.929e+12 * exp(-0.634*vlntemp + 8.167e+03*ortc);
    rr_f[158] = forward * mole_frac[18] * mole_frac[25];
    rr_r[158] = reverse * mole_frac[17] * mole_frac[26];
  }
  // 159)  CH3 + CH2(S) <=> H + C2H4
  {
    double forward = 2.0e+13;
    double reverse = 6.128e+19 * exp(-1.223*vlntemp - 7.305e+04*ortc);
    rr_f[159] = forward * mole_frac[20] * mole_frac[22];
    rr_r[159] = reverse * mole_frac[0] * mole_frac[25];
  }
  // 160)  H + C2H2 (+M) <=> C2H3 (+M)
  {
    double rr_k0 = 3.8e+40 * exp(-7.27*vlntemp - 7.22e+03*ortc);
    double rr_kinf = 5.6e+12 * exp(-2.4e+03*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[8];
    double fcent = log10(MAX(0.249 * exp(-0.01015228426395939 * temperature) + 
      0.751 * exp(-7.680491551459293e-04 * temperature) + exp(-4.167e+03 * 
      otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[0] + cgspl[26] - cgspl[27];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[160] = forward * mole_frac[0] * mole_frac[27];
    rr_r[160] = reverse * mole_frac[26];
  }
  // 161)  O2 + C2H3 <=> HO2 + C2H2
  {
    double forward = 2.12e-06 * exp(6.0*vlntemp - 9.484e+03*ortc);
    double reverse = 1.087e-05 * exp(5.905*vlntemp - 2.403e+04*ortc);
    rr_f[161] = forward * mole_frac[3] * mole_frac[26];
    rr_r[161] = reverse * mole_frac[6] * mole_frac[27];
  }
  // 162)  O2 + C2H3 <=> CH2O + HCO
  {
    double forward = 8.5e+28 * exp(-5.312*vlntemp - 6.5e+03*ortc);
    double reverse = 3.994e+27 * exp(-4.883*vlntemp - 9.345e+04*ortc);
    rr_f[162] = forward * mole_frac[3] * mole_frac[26];
    rr_r[162] = reverse * mole_frac[10] * mole_frac[11];
  }
  // 163)  CH3 + C2H3 <=> CH4 + C2H2
  {
    double forward = 3.92e+11;
    double reverse = 3.497e+14 * exp(-0.193*vlntemp - 7.078e+04*ortc);
    rr_f[163] = forward * mole_frac[20] * mole_frac[26];
    rr_r[163] = reverse * mole_frac[19] * mole_frac[27];
  }
  // 164)  H + C2H3 <=> H2 + C2H2
  {
    double forward = 9.64e+13;
    double reverse = 9.427e+13 * exp(0.253*vlntemp - 6.924e+04*ortc);
    rr_f[164] = forward * mole_frac[0] * mole_frac[26];
    rr_r[164] = reverse * mole_frac[1] * mole_frac[27];
  }
  // 165)  OH + C2H3 <=> H2O + C2H2
  {
    double forward = 5.0e+12;
    double reverse = 5.184e+13 * exp(0.147*vlntemp - 8.413e+04*ortc);
    rr_f[165] = forward * mole_frac[4] * mole_frac[26];
    rr_r[165] = reverse * mole_frac[5] * mole_frac[27];
  }
  // 166)  O2 + C2H2 <=> OH + HCCO
  {
    double forward = 2.0e+08 * exp(1.5*vlntemp - 3.01e+04*ortc);
    double reverse = 2.039e+06 * exp(1.541*vlntemp - 3.227e+04*ortc);
    rr_f[166] = forward * mole_frac[3] * mole_frac[27];
    rr_r[166] = reverse * mole_frac[4] * mole_frac[31];
  }
  // 167)  O + C2H2 <=> CO + CH2
  {
    double forward = 6.94e+06 * temperature * temperature * exp(-1.9e+03*ortc);
    double reverse = 40.5 * exp(3.198*vlntemp - 4.836e+04*ortc);
    rr_f[167] = forward * mole_frac[2] * mole_frac[27];
    rr_r[167] = reverse * mole_frac[8] * mole_frac[21];
  }
  // 168)  O + C2H2 <=> H + HCCO
  {
    double forward = 1.35e+07 * temperature * temperature * exp(-1.9e+03*ortc);
    double reverse = 4.755e+07 * exp(1.65*vlntemp - 2.08e+04*ortc);
    rr_f[168] = forward * mole_frac[2] * mole_frac[27];
    rr_r[168] = reverse * mole_frac[0] * mole_frac[31];
  }
  // 169)  OH + C2H2 <=> H + CH2CO
  {
    double forward = 3.236e+13 * exp(-1.2e+04*ortc);
    double reverse = 3.061e+17 * exp(-0.802*vlntemp - 3.579e+04*ortc);
    rr_f[169] = forward * mole_frac[4] * mole_frac[27];
    rr_r[169] = reverse * mole_frac[0] * mole_frac[30];
  }
  // 170)  OH + C2H2 <=> CO + CH3
  {
    double forward = 4.83e-04 * exp(4.0*vlntemp + 2.0e+03*ortc);
    double reverse = 3.495e-06 * exp(4.638*vlntemp - 5.212e+04*ortc);
    rr_f[170] = forward * mole_frac[4] * mole_frac[27];
    rr_r[170] = reverse * mole_frac[8] * mole_frac[20];
  }
  // 171)  CH3COCH3 (+M) <=> CH3 + CH3CO (+M)
  {
    double rr_k0 = 7.013e+89 * exp(-20.38*vlntemp - 1.0715e+05*ortc);
    double rr_kinf = 7.108e+21 * exp(-1.57*vlntemp - 8.468e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.137 * exp(-1.0e-10 * temperature) + 0.863 * 
      exp(-2.401536983669549e-03 * temperature) + exp(-3.29e+09 * otc),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = cgspl[20] + cgspl[29] - cgspl[35];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[171] = forward * mole_frac[35];
    rr_r[171] = reverse * mole_frac[20] * mole_frac[29];
  }
  // 172)  OH + CH3COCH3 <=> H2O + CH3COCH2
  {
    double forward = 1.25e+05 * exp(2.483*vlntemp - 445.0*ortc);
    double reverse = 8.62e+04 * exp(2.322*vlntemp - 2.471e+04*ortc);
    rr_f[172] = forward * mole_frac[4] * mole_frac[35];
    rr_r[172] = reverse * mole_frac[5] * mole_frac[36];
  }
  // 173)  H + CH3COCH3 <=> H2 + CH3COCH2
  {
    double forward = 9.8e+05 * exp(2.43*vlntemp - 5.16e+03*ortc);
    double reverse = 6.374e+04 * exp(2.375*vlntemp - 1.453e+04*ortc);
    rr_f[173] = forward * mole_frac[0] * mole_frac[35];
    rr_r[173] = reverse * mole_frac[1] * mole_frac[36];
  }
  // 174)  O + CH3COCH3 <=> OH + CH3COCH2
  {
    double forward = 5.13e+11 * exp(0.211*vlntemp - 4.89e+03*ortc);
    double reverse = 1.732e+10 * exp(0.137*vlntemp - 1.285e+04*ortc);
    rr_f[174] = forward * mole_frac[2] * mole_frac[35];
    rr_r[174] = reverse * mole_frac[4] * mole_frac[36];
  }
  // 175)  CH3 + CH3COCH3 <=> CH4 + CH3COCH2
  {
    double forward = 3.96e+11 * exp(-9.784e+03*ortc);
    double reverse = 2.35e+13 * exp(-0.501*vlntemp - 2.069e+04*ortc);
    rr_f[175] = forward * mole_frac[20] * mole_frac[35];
    rr_r[175] = reverse * mole_frac[19] * mole_frac[36];
  }
  // 176)  O2 + CH3COCH3 <=> HO2 + CH3COCH2
  {
    double forward = 6.03e+13 * exp(-4.85e+04*ortc);
    double reverse = 2.057e+13 * exp(-0.403*vlntemp - 3.181e+03*ortc);
    rr_f[176] = forward * mole_frac[3] * mole_frac[35];
    rr_r[176] = reverse * mole_frac[6] * mole_frac[36];
  }
  // 177)  HO2 + CH3COCH3 <=> H2O2 + CH3COCH2
  {
    double forward = 1.7e+13 * exp(-2.046e+04*ortc);
    double reverse = 6.397e+14 * exp(-0.75*vlntemp - 1.383e+04*ortc);
    rr_f[177] = forward * mole_frac[6] * mole_frac[35];
    rr_r[177] = reverse * mole_frac[7] * mole_frac[36];
  }
  // 178)  CH3O2 + CH3COCH3 <=> CH3O2H + CH3COCH2
  {
    double forward = 1.7e+13 * exp(-2.046e+04*ortc);
    double reverse = 1.245e+16 * exp(-1.195*vlntemp - 1.223e+04*ortc);
    rr_f[178] = forward * mole_frac[18] * mole_frac[35];
    rr_r[178] = reverse * mole_frac[17] * mole_frac[36];
  }
  // 179)  CH3COCH2 <=> CH3 + CH2CO
  {
    double forward = 1.0e+14 * exp(-3.1e+04*ortc);
    double reverse = 1.0e+11 * exp(-6.0e+03*ortc);
    rr_f[179] = forward * mole_frac[36];
    rr_r[179] = reverse * mole_frac[20] * mole_frac[30];
  }
  // 180)  C2H3CHO <=> HCO + C2H3
  {
    double forward = 2.003e+24 * exp(-2.135*vlntemp - 1.034e+05*ortc);
    double reverse = 1.81e+13;
    rr_f[180] = forward * mole_frac[37];
    rr_r[180] = reverse * mole_frac[11] * mole_frac[26];
  }
  // 181)  H + C2H3CHO <=> H2 + C2H3CO
  {
    double forward = 1.34e+13 * exp(-3.3e+03*ortc);
    double reverse = 3.311e+10 * exp(0.613*vlntemp - 2.268e+04*ortc);
    rr_f[181] = forward * mole_frac[0] * mole_frac[37];
    rr_r[181] = reverse * mole_frac[1] * mole_frac[38];
  }
  // 182)  O + C2H3CHO <=> OH + C2H3CO
  {
    double forward = 5.94e+12 * exp(-1.868e+03*ortc);
    double reverse = 7.618e+09 * exp(0.594*vlntemp - 1.984e+04*ortc);
    rr_f[182] = forward * mole_frac[2] * mole_frac[37];
    rr_r[182] = reverse * mole_frac[4] * mole_frac[38];
  }
  // 183)  OH + C2H3CHO <=> H2O + C2H3CO
  {
    double forward = 9.24e+06 * exp(1.5*vlntemp + 962.0*ortc);
    double reverse = 2.42e+05 * exp(2.007*vlntemp - 3.331e+04*ortc);
    rr_f[183] = forward * mole_frac[4] * mole_frac[37];
    rr_r[183] = reverse * mole_frac[5] * mole_frac[38];
  }
  // 184)  O2 + C2H3CHO <=> HO2 + C2H3CO
  {
    double forward = 1.005e+13 * exp(-4.07e+04*ortc);
    double reverse = 1.302e+11 * exp(0.265*vlntemp - 5.391e+03*ortc);
    rr_f[184] = forward * mole_frac[3] * mole_frac[37];
    rr_r[184] = reverse * mole_frac[6] * mole_frac[38];
  }
  // 185)  HO2 + C2H3CHO <=> H2O2 + C2H3CO
  {
    double forward = 3.01e+12 * exp(-1.192e+04*ortc);
    double reverse = 4.303e+12 * exp(-0.082*vlntemp - 1.53e+04*ortc);
    rr_f[185] = forward * mole_frac[6] * mole_frac[37];
    rr_r[185] = reverse * mole_frac[7] * mole_frac[38];
  }
  // 186)  CH3 + C2H3CHO <=> CH4 + C2H3CO
  {
    double forward = 2.608e+06 * exp(1.78*vlntemp - 5.911e+03*ortc);
    double reverse = 5.878e+06 * exp(1.947*vlntemp - 2.683e+04*ortc);
    rr_f[186] = forward * mole_frac[20] * mole_frac[37];
    rr_r[186] = reverse * mole_frac[19] * mole_frac[38];
  }
  // 187)  C2H3 + C2H3CHO <=> C2H4 + C2H3CO
  {
    double forward = 1.74e+12 * exp(-8.44e+03*ortc);
    double reverse = 1.0e+13 * exp(-2.8e+04*ortc);
    rr_f[187] = forward * mole_frac[26] * mole_frac[37];
    rr_r[187] = reverse * mole_frac[25] * mole_frac[38];
  }
  // 188)  CH3O2 + C2H3CHO <=> CH3O2H + C2H3CO
  {
    double forward = 3.01e+12 * exp(-1.192e+04*ortc);
    double reverse = 8.371e+13 * exp(-0.527*vlntemp - 1.371e+04*ortc);
    rr_f[188] = forward * mole_frac[18] * mole_frac[37];
    rr_r[188] = reverse * mole_frac[17] * mole_frac[38];
  }
  // 189)  C2H3CO <=> CO + C2H3
  {
    double forward = 1.37e+21 * exp(-2.179*vlntemp - 3.941e+04*ortc);
    double reverse = 1.51e+11 * exp(-4.81e+03*ortc);
    rr_f[189] = forward * mole_frac[38];
    rr_r[189] = reverse * mole_frac[8] * mole_frac[26];
  }
  // 190)  C2H5CO <=> CO + C2H5
  {
    double forward = 2.46e+23 * exp(-3.208*vlntemp - 1.755e+04*ortc);
    double reverse = 1.51e+11 * exp(-4.81e+03*ortc);
    rr_f[190] = forward * mole_frac[39];
    rr_r[190] = reverse * mole_frac[8] * mole_frac[24];
  }
  // 191)  IC3H7 <=> H + C3H6
  {
    double forward = 6.919e+13 * exp(-0.025*vlntemp - 3.769e+04*ortc);
    double reverse = 2.64e+13 * exp(-2.16e+03*ortc);
    rr_f[191] = forward * mole_frac[40];
    rr_r[191] = reverse * mole_frac[0] * mole_frac[41];
  }
  // 192)  H + IC3H7 <=> CH3 + C2H5
  {
    double forward = 2.0e+13;
    double reverse = 4.344e+07 * exp(1.176*vlntemp - 8.62e+03*ortc);
    rr_f[192] = forward * mole_frac[0] * mole_frac[40];
    rr_r[192] = reverse * mole_frac[20] * mole_frac[24];
  }
  // 193)  O2 + IC3H7 <=> HO2 + C3H6
  {
    double forward = 4.5e-19 * exp(-5.02e+03*ortc);
    double reverse = 2.0e-19 * exp(-1.75e+04*ortc);
    rr_f[193] = forward * mole_frac[3] * mole_frac[40];
    rr_r[193] = reverse * mole_frac[6] * mole_frac[41];
  }
  // 194)  OH + IC3H7 <=> H2O + C3H6
  {
    double forward = 2.41e+13;
    double reverse = 2.985e+12 * exp(0.57*vlntemp - 8.382e+04*ortc);
    rr_f[194] = forward * mole_frac[4] * mole_frac[40];
    rr_r[194] = reverse * mole_frac[5] * mole_frac[41];
  }
  // 195)  O + IC3H7 <=> H + CH3COCH3
  {
    double forward = 4.818e+13;
    double reverse = 1.293e+16 * exp(-0.19*vlntemp - 7.938e+04*ortc);
    rr_f[195] = forward * mole_frac[2] * mole_frac[40];
    rr_r[195] = reverse * mole_frac[0] * mole_frac[35];
  }
  // 196)  O + IC3H7 <=> CH3 + CH3CHO
  {
    double forward = 4.818e+13;
    double reverse = 1.279e+11 * exp(0.8*vlntemp - 8.648e+04*ortc);
    rr_f[196] = forward * mole_frac[2] * mole_frac[40];
    rr_r[196] = reverse * mole_frac[20] * mole_frac[28];
  }
  // 197)  C3H6 <=> CH3 + C2H3
  {
    double forward = 2.73e+62 * exp(-13.28*vlntemp - 1.232e+05*ortc);
    double reverse = 6.822e+53 * exp(-11.779*vlntemp - 2.055e+04*ortc);
    rr_f[197] = forward * mole_frac[41];
    rr_r[197] = reverse * mole_frac[20] * mole_frac[26];
  }
  // 198)  C3H6 <=> H + C3H5-A
  {
    double forward = 2.01e+61 * exp(-13.26*vlntemp - 1.185e+05*ortc);
    double reverse = 2.041e+61 * exp(-13.52*vlntemp - 3.061e+04*ortc);
    rr_f[198] = forward * mole_frac[41];
    rr_r[198] = reverse * mole_frac[0] * mole_frac[42];
  }
  // 199)  C3H6 <=> H + C3H5-S
  {
    double forward = 7.709999999999999e+69 * exp(-16.09*vlntemp - 1.4e+05*ortc);
    double reverse = 2.551e+67 * exp(-15.867*vlntemp - 2.869e+04*ortc);
    rr_f[199] = forward * mole_frac[41];
    rr_r[199] = reverse * mole_frac[0] * mole_frac[43];
  }
  // 200)  C3H6 <=> H + C3H5-T
  {
    double forward = 5.62e+71 * exp(-16.58*vlntemp - 1.393e+05*ortc);
    double reverse = 4.26e+68 * exp(-16.164*vlntemp - 3.008e+04*ortc);
    rr_f[200] = forward * mole_frac[41];
    rr_r[200] = reverse * mole_frac[0] * mole_frac[44];
  }
  // 201)  O + C3H6 <=> HCO + C2H5
  {
    double forward = 1.58e+07 * exp(1.76*vlntemp + 1.216e+03*ortc);
    double reverse = 91.88 * exp(2.725*vlntemp - 2.311e+04*ortc);
    rr_f[201] = forward * mole_frac[2] * mole_frac[41];
    rr_r[201] = reverse * mole_frac[11] * mole_frac[24];
  }
  // 202)  O + C3H6 <=> H + CH3 + CH2CO
  {
    double forward = 2.5e+07 * exp(1.76*vlntemp - 76.0*ortc);
    double reverse = 0.0;
    rr_f[202] = forward * mole_frac[2] * mole_frac[41];
    rr_r[202] = reverse * mole_frac[0] * mole_frac[20] * mole_frac[30];
  }
  // 203)  O + C3H6 <=> 2 H + CH3CHCO
  {
    double forward = 2.5e+07 * exp(1.76*vlntemp - 76.0*ortc);
    double reverse = 0.0;
    rr_f[203] = forward * mole_frac[2] * mole_frac[41];
    rr_r[203] = reverse * mole_frac[0] * mole_frac[0] * mole_frac[54];
  }
  // 204)  O + C3H6 <=> OH + C3H5-A
  {
    double forward = 5.24e+11 * exp(0.7*vlntemp - 5.884e+03*ortc);
    double reverse = 1.104e+11 * exp(0.697*vlntemp - 2.015e+04*ortc);
    rr_f[204] = forward * mole_frac[2] * mole_frac[41];
    rr_r[204] = reverse * mole_frac[4] * mole_frac[42];
  }
  // 205)  O + C3H6 <=> OH + C3H5-S
  {
    double forward = 1.2e+11 * exp(0.7*vlntemp - 8.959e+03*ortc);
    double reverse = 8.239e+07 * exp(1.18*vlntemp + 207.0*ortc);
    rr_f[205] = forward * mole_frac[2] * mole_frac[41];
    rr_r[205] = reverse * mole_frac[4] * mole_frac[43];
  }
  // 206)  O + C3H6 <=> OH + C3H5-T
  {
    double forward = 6.03e+10 * exp(0.7*vlntemp - 7.632e+03*ortc);
    double reverse = 9.483e+06 * exp(1.373*vlntemp - 576.0*ortc);
    rr_f[206] = forward * mole_frac[2] * mole_frac[41];
    rr_r[206] = reverse * mole_frac[4] * mole_frac[44];
  }
  // 207)  OH + C3H6 <=> H2O + C3H5-A
  {
    double forward = 3.12e+06 * temperature * temperature * exp(298.0*ortc);
    double reverse = 1.343e+07 * exp(1.909*vlntemp - 3.027e+04*ortc);
    rr_f[207] = forward * mole_frac[4] * mole_frac[41];
    rr_r[207] = reverse * mole_frac[5] * mole_frac[42];
  }
  // 208)  OH + C3H6 <=> H2O + C3H5-S
  {
    double forward = 2.11e+06 * temperature * temperature * 
      exp(-2.778e+03*ortc); 
    double reverse = 2.959e+04 * exp(2.393*vlntemp - 9.916e+03*ortc);
    rr_f[208] = forward * mole_frac[4] * mole_frac[41];
    rr_r[208] = reverse * mole_frac[5] * mole_frac[43];
  }
  // 209)  OH + C3H6 <=> H2O + C3H5-T
  {
    double forward = 1.11e+06 * temperature * temperature * 
      exp(-1.451e+03*ortc); 
    double reverse = 3.565e+03 * exp(2.586*vlntemp - 1.07e+04*ortc);
    rr_f[209] = forward * mole_frac[4] * mole_frac[41];
    rr_r[209] = reverse * mole_frac[5] * mole_frac[44];
  }
  // 210)  HO2 + C3H6 <=> H2O2 + C3H5-A
  {
    double forward = 2.7e+04 * exp(2.5*vlntemp - 1.234e+04*ortc);
    double reverse = 6.341e+06 * exp(1.82*vlntemp - 1.201e+04*ortc);
    rr_f[210] = forward * mole_frac[6] * mole_frac[41];
    rr_r[210] = reverse * mole_frac[7] * mole_frac[42];
  }
  // 211)  HO2 + C3H6 <=> H2O2 + C3H5-S
  {
    double forward = 1.8e+04 * exp(2.5*vlntemp - 2.762e+04*ortc);
    double reverse = 1.377e+04 * exp(2.304*vlntemp - 3.864e+03*ortc);
    rr_f[211] = forward * mole_frac[6] * mole_frac[41];
    rr_r[211] = reverse * mole_frac[7] * mole_frac[43];
  }
  // 212)  HO2 + C3H6 <=> H2O2 + C3H5-T
  {
    double forward = 9.0e+03 * exp(2.5*vlntemp - 2.359e+04*ortc);
    double reverse = 1.577e+03 * exp(2.497*vlntemp - 1.941e+03*ortc);
    rr_f[212] = forward * mole_frac[6] * mole_frac[41];
    rr_r[212] = reverse * mole_frac[7] * mole_frac[44];
  }
  // 213)  H + C3H6 <=> H2 + C3H5-A
  {
    double forward = 1.73e+05 * exp(2.5*vlntemp - 2.492e+03*ortc);
    double reverse = 7.023e+04 * exp(2.515*vlntemp - 1.817e+04*ortc);
    rr_f[213] = forward * mole_frac[0] * mole_frac[41];
    rr_r[213] = reverse * mole_frac[1] * mole_frac[42];
  }
  // 214)  H + C3H6 <=> H2 + C3H5-S
  {
    double forward = 8.04e+05 * exp(2.5*vlntemp - 1.228e+04*ortc);
    double reverse = 1.063e+03 * exp(2.999*vlntemp - 4.526e+03*ortc);
    rr_f[214] = forward * mole_frac[0] * mole_frac[41];
    rr_r[214] = reverse * mole_frac[1] * mole_frac[43];
  }
  // 215)  H + C3H6 <=> H2 + C3H5-T
  {
    double forward = 4.05e+05 * exp(2.5*vlntemp - 9.794e+03*ortc);
    double reverse = 122.7 * exp(3.192*vlntemp - 4.15e+03*ortc);
    rr_f[215] = forward * mole_frac[0] * mole_frac[41];
    rr_r[215] = reverse * mole_frac[1] * mole_frac[44];
  }
  // 216)  H + C3H6 <=> CH3 + C2H4
  {
    double forward = 2.3e+13 * exp(-2.547e+03*ortc);
    double reverse = 7.272e+07 * exp(1.271*vlntemp - 1.12e+04*ortc);
    rr_f[216] = forward * mole_frac[0] * mole_frac[41];
    rr_r[216] = reverse * mole_frac[20] * mole_frac[25];
  }
  // 217)  O2 + C3H6 <=> HO2 + C3H5-A
  {
    double forward = 4.0e+12 * exp(-3.99e+04*ortc);
    double reverse = 8.514e+12 * exp(-0.333*vlntemp - 887.0*ortc);
    rr_f[217] = forward * mole_frac[3] * mole_frac[41];
    rr_r[217] = reverse * mole_frac[6] * mole_frac[42];
  }
  // 218)  O2 + C3H6 <=> HO2 + C3H5-S
  {
    double forward = 2.0e+12 * exp(-6.29e+04*ortc);
    double reverse = 1.387e+10 * exp(0.151*vlntemp - 459.0*ortc);
    rr_f[218] = forward * mole_frac[3] * mole_frac[41];
    rr_r[218] = reverse * mole_frac[6] * mole_frac[43];
  }
  // 219)  O2 + C3H6 <=> HO2 + C3H5-T
  {
    double forward = 1.4e+12 * exp(-6.07e+04*ortc);
    double reverse = 2.224e+09 * exp(0.344*vlntemp - 369.0*ortc);
    rr_f[219] = forward * mole_frac[3] * mole_frac[41];
    rr_r[219] = reverse * mole_frac[6] * mole_frac[44];
  }
  // 220)  CH3 + C3H6 <=> CH4 + C3H5-A
  {
    double forward = 2.21 * exp(3.5*vlntemp - 5.675e+03*ortc);
    double reverse = 818.4 * exp(3.07*vlntemp - 2.289e+04*ortc);
    rr_f[220] = forward * mole_frac[20] * mole_frac[41];
    rr_r[220] = reverse * mole_frac[19] * mole_frac[42];
  }
  // 221)  CH3 + C3H6 <=> CH4 + C3H5-S
  {
    double forward = 1.348 * exp(3.5*vlntemp - 1.285e+04*ortc);
    double reverse = 1.626 * exp(3.553*vlntemp - 6.635e+03*ortc);
    rr_f[221] = forward * mole_frac[20] * mole_frac[41];
    rr_r[221] = reverse * mole_frac[19] * mole_frac[43];
  }
  // 222)  CH3 + C3H6 <=> CH4 + C3H5-T
  {
    double forward = 0.84 * exp(3.5*vlntemp - 1.166e+04*ortc);
    double reverse = 0.2322 * exp(3.746*vlntemp - 7.552e+03*ortc);
    rr_f[222] = forward * mole_frac[20] * mole_frac[41];
    rr_r[222] = reverse * mole_frac[19] * mole_frac[44];
  }
  // 223)  C2H5 + C3H6 <=> C2H6 + C3H5-A
  {
    double forward = 1.0e+11 * exp(-9.8e+03*ortc);
    double reverse = 5.369e+05 * exp(1.33*vlntemp - 1.644e+04*ortc);
    rr_f[223] = forward * mole_frac[24] * mole_frac[41];
    rr_r[223] = reverse * mole_frac[23] * mole_frac[42];
  }
  // 224)  CH3O2 + C3H6 <=> CH3O2H + C3H5-A
  {
    double forward = 3.24e+11 * exp(-1.49e+04*ortc);
    double reverse = 2.0e+10 * exp(-1.5e+04*ortc);
    rr_f[224] = forward * mole_frac[18] * mole_frac[41];
    rr_r[224] = reverse * mole_frac[17] * mole_frac[42];
  }
  // 225)  C3H5-A <=> CH3 + C2H2
  {
    double forward = 2.397e+48 * exp(-9.9*vlntemp - 8.208e+04*ortc);
    double reverse = 2.61e+46 * exp(-9.82*vlntemp - 3.695e+04*ortc);
    rr_f[225] = forward * mole_frac[42];
    rr_r[225] = reverse * mole_frac[20] * mole_frac[27];
  }
  // 226)  C3H5-A <=> H + C3H4-A
  {
    double forward = 4.194e+13 * exp(0.216*vlntemp - 6.193e+04*ortc);
    double reverse = 2.4e+11 * exp(0.6899999999999999*vlntemp - 3.007e+03*ortc);
    rr_f[226] = forward * mole_frac[42];
    rr_r[226] = reverse * mole_frac[0] * mole_frac[46];
  }
  // 227)  HO2 + C3H5-A <=> OH + C3H5O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 1.605e+12 * exp(0.06*vlntemp - 1.166e+04*ortc);
    rr_f[227] = forward * mole_frac[6] * mole_frac[42];
    rr_r[227] = reverse * mole_frac[4] * mole_frac[49];
  }
  // 228)  CH3O2 + C3H5-A <=> CH3O + C3H5O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 1.99e+15 * exp(-0.74*vlntemp - 1.702e+04*ortc);
    rr_f[228] = forward * mole_frac[18] * mole_frac[42];
    rr_r[228] = reverse * mole_frac[16] * mole_frac[49];
  }
  // 229)  H + C3H5-A <=> H2 + C3H4-A
  {
    double forward = 1.232e+03 * exp(3.035*vlntemp - 2.582e+03*ortc);
    double reverse = 2.818 * exp(3.784*vlntemp - 4.722e+04*ortc);
    rr_f[229] = forward * mole_frac[0] * mole_frac[42];
    rr_r[229] = reverse * mole_frac[1] * mole_frac[46];
  }
  // 230)  CH3 + C3H5-A <=> CH4 + C3H4-A
  {
    double forward = 1.0e+11;
    double reverse = 4.921e+12 * exp(0.05*vlntemp - 4.778e+04*ortc);
    rr_f[230] = forward * mole_frac[20] * mole_frac[42];
    rr_r[230] = reverse * mole_frac[19] * mole_frac[46];
  }
  // 231)  C2H5 + C3H5-A <=> C2H6 + C3H4-A
  {
    double forward = 4.0e+11;
    double reverse = 1.802e+12 * exp(0.05*vlntemp - 4.033e+04*ortc);
    rr_f[231] = forward * mole_frac[24] * mole_frac[42];
    rr_r[231] = reverse * mole_frac[23] * mole_frac[46];
  }
  // 232)  C2H5 + C3H5-A <=> C2H4 + C3H6
  {
    double forward = 4.0e+11;
    double reverse = 6.937e+16 * exp(-1.33*vlntemp - 5.28e+04*ortc);
    rr_f[232] = forward * mole_frac[24] * mole_frac[42];
    rr_r[232] = reverse * mole_frac[25] * mole_frac[41];
  }
  // 233)  C2H3 + C3H5-A <=> C2H4 + C3H4-A
  {
    double forward = 1.0e+12;
    double reverse = 1.624e+13 * exp(0.05*vlntemp - 4.819e+04*ortc);
    rr_f[233] = forward * mole_frac[26] * mole_frac[42];
    rr_r[233] = reverse * mole_frac[25] * mole_frac[46];
  }
  // 234)  C3H6 + C3H4-A <=> 2 C3H5-A
  {
    double forward = 4.749e+08 * exp(0.734*vlntemp - 2.87e+04*ortc);
    double reverse = 8.43e+10 * exp(262.0*ortc);
    rr_f[234] = forward * mole_frac[41] * mole_frac[46];
    rr_r[234] = reverse * mole_frac[42] * mole_frac[42];
  }
  // 235)  O2 + C3H5-A <=> HO2 + C3H4-A
  {
    double forward = 2.18e+21 * exp(-2.85*vlntemp - 3.076e+04*ortc);
    double reverse = 2.614e+19 * exp(-2.449*vlntemp - 2.071e+04*ortc);
    rr_f[235] = forward * mole_frac[3] * mole_frac[42];
    rr_r[235] = reverse * mole_frac[6] * mole_frac[46];
  }
  // 236)  O2 + C3H5-A <=> OH + C2H3CHO
  {
    double forward = 2.47e+13 * exp(-0.44*vlntemp - 2.302e+04*ortc);
    double reverse = 1.989e+13 * exp(-0.609*vlntemp - 7.514e+04*ortc);
    rr_f[236] = forward * mole_frac[3] * mole_frac[42];
    rr_r[236] = reverse * mole_frac[4] * mole_frac[37];
  }
  // 237)  O2 + C3H5-A <=> OH + CH2O + C2H2
  {
    double forward = 9.72e+29 * exp(-5.71*vlntemp - 2.145e+04*ortc);
    double reverse = 0.0;
    rr_f[237] = forward * mole_frac[3] * mole_frac[42];
    rr_r[237] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[27];
  }
  // 238)  C3H5-S <=> CH3 + C2H2
  {
    double forward = 9.598e+39 * exp(-8.17*vlntemp - 4.203e+04*ortc);
    double reverse = 1.61e+40 * exp(-8.58*vlntemp - 2.033e+04*ortc);
    rr_f[238] = forward * mole_frac[43];
    rr_r[238] = reverse * mole_frac[20] * mole_frac[27];
  }
  // 239)  C3H5-S <=> H + C3H4-P
  {
    double forward = 4.187e+15 * exp(-0.79*vlntemp - 3.748e+04*ortc);
    double reverse = 5.8e+12 * exp(-3.1e+03*ortc);
    rr_f[239] = forward * mole_frac[43];
    rr_r[239] = reverse * mole_frac[0] * mole_frac[45];
  }
  // 240)  O2 + C3H5-S <=> HCO + CH3CHO
  {
    double forward = 4.335e+12;
    double reverse = 1.611e+17 * exp(-1.27*vlntemp - 9.653e+04*ortc);
    rr_f[240] = forward * mole_frac[3] * mole_frac[43];
    rr_r[240] = reverse * mole_frac[11] * mole_frac[28];
  }
  // 241)  H + C3H5-S <=> H2 + C3H4-A
  {
    double forward = 3.333e+12;
    double reverse = 7.977e+12 * exp(0.11*vlntemp - 6.886e+04*ortc);
    rr_f[241] = forward * mole_frac[0] * mole_frac[43];
    rr_r[241] = reverse * mole_frac[1] * mole_frac[46];
  }
  // 242)  CH3 + C3H5-S <=> CH4 + C3H4-A
  {
    double forward = 1.0e+11;
    double reverse = 6.253e+12 * exp(0.11*vlntemp - 6.934e+04*ortc);
    rr_f[242] = forward * mole_frac[20] * mole_frac[43];
    rr_r[242] = reverse * mole_frac[19] * mole_frac[46];
  }
  // 243)  C3H5-T <=> CH3 + C2H2
  {
    double forward = 2.163e+40 * exp(-8.31*vlntemp - 4.511e+04*ortc);
    double reverse = 1.61e+40 * exp(-8.58*vlntemp - 2.033e+04*ortc);
    rr_f[243] = forward * mole_frac[44];
    rr_r[243] = reverse * mole_frac[20] * mole_frac[27];
  }
  // 244)  C3H5-T <=> H + C3H4-A
  {
    double forward = 3.508e+14 * exp(-0.44*vlntemp - 4.089e+04*ortc);
    double reverse = 8.5e+12 * exp(-2.0e+03*ortc);
    rr_f[244] = forward * mole_frac[44];
    rr_r[244] = reverse * mole_frac[0] * mole_frac[46];
  }
  // 245)  C3H5-T <=> H + C3H4-P
  {
    double forward = 1.075e+15 * exp(-0.6*vlntemp - 3.849e+04*ortc);
    double reverse = 6.5e+12 * exp(-2.0e+03*ortc);
    rr_f[245] = forward * mole_frac[44];
    rr_r[245] = reverse * mole_frac[0] * mole_frac[45];
  }
  // 246)  O2 + C3H5-T <=> HO2 + C3H4-A
  {
    double forward = 1.89e+30 * exp(-5.59*vlntemp - 1.554e+04*ortc);
    double reverse = 3.037e+31 * exp(-5.865*vlntemp - 2.681e+04*ortc);
    rr_f[246] = forward * mole_frac[3] * mole_frac[44];
    rr_r[246] = reverse * mole_frac[6] * mole_frac[46];
  }
  // 247)  O2 + C3H5-T <=> O + CH3COCH2
  {
    double forward = 3.81e+17 * exp(-1.36*vlntemp - 5.58e+03*ortc);
    double reverse = 2.0e+11 * exp(-1.75e+04*ortc);
    rr_f[247] = forward * mole_frac[3] * mole_frac[44];
    rr_r[247] = reverse * mole_frac[2] * mole_frac[36];
  }
  // 248)  O2 + C3H5-T <=> CH2O + CH3CO
  {
    double forward = 3.71e+25 * exp(-3.96*vlntemp - 7.043e+03*ortc);
    double reverse = 1.872e+27 * exp(-4.43*vlntemp - 1.012e+05*ortc);
    rr_f[248] = forward * mole_frac[3] * mole_frac[44];
    rr_r[248] = reverse * mole_frac[10] * mole_frac[29];
  }
  // 249)  H + C3H5-T <=> H2 + C3H4-P
  {
    double forward = 3.333e+12;
    double reverse = 2.138e+16 * exp(-0.88*vlntemp - 7.105e+04*ortc);
    rr_f[249] = forward * mole_frac[0] * mole_frac[44];
    rr_r[249] = reverse * mole_frac[1] * mole_frac[45];
  }
  // 250)  CH3 + C3H5-T <=> CH4 + C3H4-P
  {
    double forward = 1.0e+11;
    double reverse = 1.676e+16 * exp(-0.88*vlntemp - 7.153e+04*ortc);
    rr_f[250] = forward * mole_frac[20] * mole_frac[44];
    rr_r[250] = reverse * mole_frac[19] * mole_frac[45];
  }
  // 251)  C3H4-A + M <=> H + C3H3 + M
  {
    double forward = 1.143e+17 * exp(-7.0e+04*ortc);
    double reverse = 1.798e+15 * exp(-0.38*vlntemp - 1.061e+04*ortc);
    rr_f[251] = forward * mole_frac[46];
    rr_r[251] = reverse * mole_frac[0] * mole_frac[47];
    rr_f[251] *= thbctemp[0];
    rr_r[251] *= thbctemp[0];
  }
  // 252)  C3H4-A <=> C3H4-P
  {
    double forward = 1.202e+15 * exp(-9.24e+04*ortc);
    double reverse = 3.222e+18 * exp(-0.99*vlntemp - 9.659e+04*ortc);
    rr_f[252] = forward * mole_frac[46];
    rr_r[252] = reverse * mole_frac[45];
  }
  // 253)  O2 + C3H4-A <=> HO2 + C3H3
  {
    double forward = 4.0e+13 * exp(-3.916e+04*ortc);
    double reverse = 3.17e+11 * exp(-0.08599999999999999*vlntemp - 311.0*ortc);
    rr_f[253] = forward * mole_frac[3] * mole_frac[46];
    rr_r[253] = reverse * mole_frac[6] * mole_frac[47];
  }
  // 254)  HO2 + C3H4-A <=> OH + CH2 + CH2CO
  {
    double forward = 4.0e+12 * exp(-1.9e+04*ortc);
    double reverse = 1.0;
    rr_f[254] = forward * mole_frac[6] * mole_frac[46];
    rr_r[254] = reverse * mole_frac[4] * mole_frac[21] * mole_frac[30];
  }
  // 255)  OH + C3H4-A <=> CH3 + CH2CO
  {
    double forward = 3.12e+12 * exp(397.0*ortc);
    double reverse = 1.806e+17 * exp(-1.38*vlntemp - 3.607e+04*ortc);
    rr_f[255] = forward * mole_frac[4] * mole_frac[46];
    rr_r[255] = reverse * mole_frac[20] * mole_frac[30];
  }
  // 256)  OH + C3H4-A <=> H2O + C3H3
  {
    double forward = 1.0e+07 * temperature * temperature * exp(-1.0e+03*ortc);
    double reverse = 1.602e+05 * exp(2.157*vlntemp - 3.173e+04*ortc);
    rr_f[256] = forward * mole_frac[4] * mole_frac[46];
    rr_r[256] = reverse * mole_frac[5] * mole_frac[47];
  }
  // 257)  O + C3H4-A <=> CO + C2H4
  {
    double forward = 7.8e+12 * exp(-1.6e+03*ortc);
    double reverse = 3.269e+08 * exp(1.252*vlntemp - 1.219e+05*ortc);
    rr_f[257] = forward * mole_frac[2] * mole_frac[46];
    rr_r[257] = reverse * mole_frac[8] * mole_frac[25];
  }
  // 258)  O + C3H4-A <=> CH2O + C2H2
  {
    double forward = 3.0e-03 * exp(4.61*vlntemp + 4.243e+03*ortc);
    double reverse = 232.0 * exp(3.23*vlntemp - 8.119e+04*ortc);
    rr_f[258] = forward * mole_frac[2] * mole_frac[46];
    rr_r[258] = reverse * mole_frac[10] * mole_frac[27];
  }
  // 259)  H + C3H4-A <=> H2 + C3H3
  {
    double forward = 2.0e+07 * temperature * temperature * exp(-5.0e+03*ortc);
    double reverse = 3.022e+04 * exp(2.262*vlntemp - 2.084e+04*ortc);
    rr_f[259] = forward * mole_frac[0] * mole_frac[46];
    rr_r[259] = reverse * mole_frac[1] * mole_frac[47];
  }
  // 260)  CH3 + C3H4-A <=> CH4 + C3H3
  {
    double forward = 0.0367 * exp(4.01*vlntemp - 6.83e+03*ortc);
    double reverse = 0.0506 * exp(3.826*vlntemp - 2.421e+04*ortc);
    rr_f[260] = forward * mole_frac[20] * mole_frac[46];
    rr_r[260] = reverse * mole_frac[19] * mole_frac[47];
  }
  // 261)  C3H5-A + C3H4-A <=> C3H6 + C3H3
  {
    double forward = 2.0e+11 * exp(-7.7e+03*ortc);
    double reverse = 2.644e+19 * exp(-2.71*vlntemp - 4.214e+04*ortc);
    rr_f[261] = forward * mole_frac[42] * mole_frac[46];
    rr_r[261] = reverse * mole_frac[41] * mole_frac[47];
  }
  // 262)  C3H4-P + M <=> H + C3H3 + M
  {
    double forward = 1.143e+17 * exp(-7.0e+04*ortc);
    double reverse = 6.708e+11 * exp(0.61*vlntemp - 6.42e+03*ortc);
    rr_f[262] = forward * mole_frac[45];
    rr_r[262] = reverse * mole_frac[0] * mole_frac[47];
    rr_f[262] *= thbctemp[0];
    rr_r[262] *= thbctemp[0];
  }
  // 263)  O2 + C3H4-P <=> OH + CH2 + HCCO
  {
    double forward = 1.0e+07 * exp(1.5*vlntemp - 3.01e+04*ortc);
    double reverse = 1.0;
    rr_f[263] = forward * mole_frac[3] * mole_frac[45];
    rr_r[263] = reverse * mole_frac[4] * mole_frac[21] * mole_frac[31];
  }
  // 264)  O2 + C3H4-P <=> HO2 + C3H3
  {
    double forward = 2.0e+13 * exp(-4.16e+04*ortc);
    double reverse = 6.371e+11 * exp(-0.208*vlntemp - 1.021e+03*ortc);
    rr_f[264] = forward * mole_frac[3] * mole_frac[45];
    rr_r[264] = reverse * mole_frac[6] * mole_frac[47];
  }
  // 265)  HO2 + C3H4-P <=> OH + CO + C2H4
  {
    double forward = 3.0e+12 * exp(-1.9e+04*ortc);
    double reverse = 1.0;
    rr_f[265] = forward * mole_frac[6] * mole_frac[45];
    rr_r[265] = reverse * mole_frac[4] * mole_frac[8] * mole_frac[25];
  }
  // 266)  OH + C3H4-P <=> H2O + C3H3
  {
    double forward = 1.0e+07 * temperature * temperature * exp(-1.0e+03*ortc);
    double reverse = 6.441e+05 * exp(2.034*vlntemp - 3.0e+04*ortc);
    rr_f[266] = forward * mole_frac[4] * mole_frac[45];
    rr_r[266] = reverse * mole_frac[5] * mole_frac[47];
  }
  // 267)  OH + C3H4-P <=> CH3 + CH2CO
  {
    double forward = 5.0e-04 * exp(4.5*vlntemp + 1.0e+03*ortc);
    double reverse = 0.01079 * exp(4.11*vlntemp - 3.128e+04*ortc);
    rr_f[267] = forward * mole_frac[4] * mole_frac[45];
    rr_r[267] = reverse * mole_frac[20] * mole_frac[30];
  }
  // 268)  O + C3H4-P <=> HCO + C2H3
  {
    double forward = 3.2e+12 * exp(-2.01e+03*ortc);
    double reverse = 2.548e+12 * exp(-0.39*vlntemp - 3.235e+04*ortc);
    rr_f[268] = forward * mole_frac[2] * mole_frac[45];
    rr_r[268] = reverse * mole_frac[11] * mole_frac[26];
  }
  // 269)  O + C3H4-P <=> CH3 + HCCO
  {
    double forward = 9.6e+08 * temperature;
    double reverse = 1.43e+04 * exp(1.793*vlntemp - 2.699e+04*ortc);
    rr_f[269] = forward * mole_frac[2] * mole_frac[45];
    rr_r[269] = reverse * mole_frac[20] * mole_frac[31];
  }
  // 270)  O + C3H4-P <=> H + CH2 + HCCO
  {
    double forward = 3.2e-19 * exp(-2.01e+03*ortc);
    double reverse = 1.0e-30;
    rr_f[270] = forward * mole_frac[2] * mole_frac[45];
    rr_r[270] = reverse * mole_frac[0] * mole_frac[21] * mole_frac[31];
  }
  // 271)  O + C3H4-P <=> OH + C3H3
  {
    double forward = 7.65e+08 * exp(1.5*vlntemp - 8.6e+03*ortc);
    double reverse = 2.177e+08 * exp(1.31*vlntemp - 2.247e+04*ortc);
    rr_f[271] = forward * mole_frac[2] * mole_frac[45];
    rr_r[271] = reverse * mole_frac[4] * mole_frac[47];
  }
  // 272)  H + C3H4-P <=> H2 + C3H3
  {
    double forward = 2.0e+07 * temperature * temperature * exp(-5.0e+03*ortc);
    double reverse = 1.215e+05 * exp(2.14*vlntemp - 1.911e+04*ortc);
    rr_f[272] = forward * mole_frac[0] * mole_frac[45];
    rr_r[272] = reverse * mole_frac[1] * mole_frac[47];
  }
  // 273)  CH3 + C3H4-P <=> CH4 + C3H3
  {
    double forward = 1.5 * exp(3.5*vlntemp - 5.6e+03*ortc);
    double reverse = 8.313000000000001 * exp(3.195*vlntemp - 2.125e+04*ortc);
    rr_f[273] = forward * mole_frac[20] * mole_frac[45];
    rr_r[273] = reverse * mole_frac[19] * mole_frac[47];
  }
  // 274)  C2H3 + C3H4-P <=> C2H4 + C3H3
  {
    double forward = 1.0e+12 * exp(-7.7e+03*ortc);
    double reverse = 9.541e+11 * exp(-0.39*vlntemp - 5.245e+04*ortc);
    rr_f[274] = forward * mole_frac[26] * mole_frac[45];
    rr_r[274] = reverse * mole_frac[25] * mole_frac[47];
  }
  // 275)  C3H5-A + C3H4-P <=> C3H6 + C3H3
  {
    double forward = 1.0e+12 * exp(-7.7e+03*ortc);
    double reverse = 4.931e+16 * exp(-1.73*vlntemp - 3.795e+04*ortc);
    rr_f[275] = forward * mole_frac[42] * mole_frac[45];
    rr_r[275] = reverse * mole_frac[41] * mole_frac[47];
  }
  // 276)  OH + C3H3 <=> H2O + C3H2
  {
    double forward = 1.0e+13;
    double reverse = 1.343e+15 * exp(-1.568e+04*ortc);
    rr_f[276] = forward * mole_frac[4] * mole_frac[47];
    rr_r[276] = reverse * mole_frac[5] * mole_frac[48];
  }
  // 277)  O2 + C3H3 <=> HCO + CH2CO
  {
    double forward = 3.01e+10 * exp(-2.87e+03*ortc);
    double reverse = 4.881e+11 * exp(-5.947e+04*ortc);
    rr_f[277] = forward * mole_frac[3] * mole_frac[47];
    rr_r[277] = reverse * mole_frac[11] * mole_frac[30];
  }
  // 278)  O2 + C3H2 <=> HCO + HCCO
  {
    double forward = 5.0e+13;
    double reverse = 2.326e+14 * exp(-0.214*vlntemp - 7.719e+04*ortc);
    rr_f[278] = forward * mole_frac[3] * mole_frac[48];
    rr_r[278] = reverse * mole_frac[11] * mole_frac[31];
  }
  // 279)  HO2 + C3H4-A <=> OH + CO + C2H4
  {
    double forward = 1.0e+12 * exp(-1.4e+04*ortc);
    double reverse = 1.0;
    rr_f[279] = forward * mole_frac[6] * mole_frac[46];
    rr_r[279] = reverse * mole_frac[4] * mole_frac[8] * mole_frac[25];
  }
  // 280)  HO2 + C3H4-A <=> H2O2 + C3H3
  {
    double forward = 3.0e+13 * exp(-1.4e+04*ortc);
    double reverse = 1.551e+16 * exp(-1.38*vlntemp - 4.4e+04*ortc);
    rr_f[280] = forward * mole_frac[6] * mole_frac[46];
    rr_r[280] = reverse * mole_frac[7] * mole_frac[47];
  }
  // 281)  CH3 + C2H2 <=> H + C3H4-P
  {
    double forward = 4.229e+08 * exp(1.143*vlntemp - 1.209e+04*ortc);
    double reverse = 1.0e+14 * exp(-4.0e+03*ortc);
    rr_f[281] = forward * mole_frac[20] * mole_frac[27];
    rr_r[281] = reverse * mole_frac[0] * mole_frac[45];
  }
  // 282)  CH3 + C2H2 <=> H + C3H4-A
  {
    double forward = 6.74e+19 * exp(-2.08*vlntemp - 3.159e+04*ortc);
    double reverse = 6.407e+25 * exp(-3.345*vlntemp - 2.177e+04*ortc);
    rr_f[282] = forward * mole_frac[20] * mole_frac[27];
    rr_r[282] = reverse * mole_frac[0] * mole_frac[46];
  }
  // 283)  H + C3H3 <=> H2 + C3H2
  {
    double forward = 5.0e+13;
    double reverse = 5.999e+07 * exp(1.365*vlntemp - 4.11e+03*ortc);
    rr_f[283] = forward * mole_frac[0] * mole_frac[47];
    rr_r[283] = reverse * mole_frac[1] * mole_frac[48];
  }
  // 284)  OH + C3H2 <=> HCO + C2H2
  {
    double forward = 5.0e+13;
    double reverse = 2.282e+16 * exp(-0.254*vlntemp - 7.502e+04*ortc);
    rr_f[284] = forward * mole_frac[4] * mole_frac[48];
    rr_r[284] = reverse * mole_frac[11] * mole_frac[27];
  }
  // 285)  O2 + C3H2 <=> H + CO + HCCO
  {
    double forward = 5.0e+13;
    double reverse = 0.0;
    rr_f[285] = forward * mole_frac[3] * mole_frac[48];
    rr_r[285] = reverse * mole_frac[0] * mole_frac[8] * mole_frac[31];
  }
  // 286)  OH + CH3CHCO <=> CO2 + C2H5
  {
    double forward = 1.73e+12 * exp(1.01e+03*ortc);
    double reverse = 0.0;
    rr_f[286] = forward * mole_frac[4] * mole_frac[54];
    rr_r[286] = reverse * mole_frac[9] * mole_frac[24];
  }
  // 287)  H + CH3CHCO <=> CO + C2H5
  {
    double forward = 4.4e+12 * exp(-1.459e+03*ortc);
    double reverse = 0.0;
    rr_f[287] = forward * mole_frac[0] * mole_frac[54];
    rr_r[287] = reverse * mole_frac[8] * mole_frac[24];
  }
  // 288)  O + CH3CHCO <=> CO + CH3CHO
  {
    double forward = 3.2e+12 * exp(437.0*ortc);
    double reverse = 0.0;
    rr_f[288] = forward * mole_frac[2] * mole_frac[54];
    rr_r[288] = reverse * mole_frac[8] * mole_frac[28];
  }
  // 289)  IC3H7O2 <=> O2 + IC3H7
  {
    double forward = 3.132e+22 * exp(-2.167*vlntemp - 3.816e+04*ortc);
    double reverse = 7.54e+12;
    rr_f[289] = forward * mole_frac[52];
    rr_r[289] = reverse * mole_frac[3] * mole_frac[40];
  }
  // 290)  IC3H7O2 <=> C3H6OOH2-1
  {
    double forward = 1.8e+12 * exp(-2.94e+04*ortc);
    double reverse = 1.122e+10 * exp(0.119*vlntemp - 1.181e+04*ortc);
    rr_f[290] = forward * mole_frac[52];
    rr_r[290] = reverse * mole_frac[50];
  }
  // 291)  C3H6OOH2-1 <=> HO2 + C3H6
  {
    double forward = 3.239e+18 * otc * otc * exp(-1.897e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.175e+04*ortc);
    rr_f[291] = forward * mole_frac[50];
    rr_r[291] = reverse * mole_frac[6] * mole_frac[41];
  }
  // 292)  C3H6OOH2-1O2 <=> O2 + C3H6OOH2-1
  {
    double forward = 5.227e+22 * exp(-2.244*vlntemp - 3.782e+04*ortc);
    double reverse = 4.52e+12;
    rr_f[292] = forward * mole_frac[51];
    rr_r[292] = reverse * mole_frac[3] * mole_frac[50];
  }
  // 293)  C3H6OOH2-1O2 <=> OH + C3KET21
  {
    double forward = 3.0e+11 * exp(-2.385e+04*ortc);
    double reverse = 1.397e+03 * exp(1.834*vlntemp - 4.975e+04*ortc);
    rr_f[293] = forward * mole_frac[51];
    rr_r[293] = reverse * mole_frac[4] * mole_frac[53];
  }
  // 294)  C3KET21 <=> OH + CH2O + CH3CO
  {
    double forward = 1.0e+16 * exp(-4.3e+04*ortc);
    double reverse = 0.0;
    rr_f[294] = forward * mole_frac[53];
    rr_r[294] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[29];
  }
  // 295)  C3H5O <=> H + C2H3CHO
  {
    double forward = 1.0e+14 * exp(-2.91e+04*ortc);
    double reverse = 1.676e+14 * exp(-0.156*vlntemp - 1.969e+04*ortc);
    rr_f[295] = forward * mole_frac[49];
    rr_r[295] = reverse * mole_frac[0] * mole_frac[37];
  }
  // 296)  C3H5O <=> CH2O + C2H3
  {
    double forward = 1.464e+20 * exp(-1.968*vlntemp - 3.509e+04*ortc);
    double reverse = 1.5e+11 * exp(-1.06e+04*ortc);
    rr_f[296] = forward * mole_frac[49];
    rr_r[296] = reverse * mole_frac[10] * mole_frac[26];
  }
  // 297)  O2 + C3H5O <=> HO2 + C2H3CHO
  {
    double forward = 1.0e+12 * exp(-6.0e+03*ortc);
    double reverse = 1.288e+11 * exp(-3.2e+04*ortc);
    rr_f[297] = forward * mole_frac[3] * mole_frac[49];
    rr_r[297] = reverse * mole_frac[6] * mole_frac[37];
  }
  // 298)  IC3H7O2 <=> HO2 + C3H6
  {
    double forward = 1.015e+43 * exp(-9.409000000000001*vlntemp - 
      4.149e+04*ortc); 
    double reverse = 1.954e+33 * exp(-7.289*vlntemp - 1.667e+04*ortc);
    rr_f[298] = forward * mole_frac[52];
    rr_r[298] = reverse * mole_frac[6] * mole_frac[41];
  }
  // 299)  SC4H9 <=> CH3 + C3H6
  {
    double forward = 4.803e+10 * exp(1.044*vlntemp - 3.035e+04*ortc);
    double reverse = 1.76e+04 * exp(2.48*vlntemp - 6.13e+03*ortc);
    rr_f[299] = forward * mole_frac[55];
    rr_r[299] = reverse * mole_frac[20] * mole_frac[41];
  }
  // 300)  HO2 + TC4H9 <=> OH + TC4H9O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 4.083e+18 * exp(-1.329*vlntemp - 2.865e+04*ortc);
    rr_f[300] = forward * mole_frac[6] * mole_frac[57];
    rr_r[300] = reverse * mole_frac[4] * mole_frac[63];
  }
  // 301)  CH3O2 + TC4H9 <=> CH3O + TC4H9O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 2.554e+17 * exp(-1.03*vlntemp - 3.289e+04*ortc);
    rr_f[301] = forward * mole_frac[18] * mole_frac[57];
    rr_r[301] = reverse * mole_frac[16] * mole_frac[63];
  }
  // 302)  IC4H9 <=> H + IC4H8
  {
    double forward = 3.371e+13 * exp(0.124*vlntemp - 3.366e+04*ortc);
    double reverse = 6.25e+11 * exp(0.51*vlntemp - 2.62e+03*ortc);
    rr_f[302] = forward * mole_frac[56];
    rr_r[302] = reverse * mole_frac[0] * mole_frac[58];
  }
  // 303)  IC4H9 <=> CH3 + C3H6
  {
    double forward = 9.504e+11 * exp(0.773*vlntemp - 3.07e+04*ortc);
    double reverse = 1.89e+03 * exp(2.67*vlntemp - 6.85e+03*ortc);
    rr_f[303] = forward * mole_frac[56];
    rr_r[303] = reverse * mole_frac[20] * mole_frac[41];
  }
  // 304)  TC4H9 <=> H + IC4H8
  {
    double forward = 1.128e+12 * exp(0.703*vlntemp - 3.656e+04*ortc);
    double reverse = 1.06e+12 * exp(0.51*vlntemp - 1.23e+03*ortc);
    rr_f[304] = forward * mole_frac[57];
    rr_r[304] = reverse * mole_frac[0] * mole_frac[58];
  }
  // 305)  O2 + TC4H9 <=> HO2 + IC4H8
  {
    double forward = 0.837 * exp(3.59*vlntemp - 1.196e+04*ortc);
    double reverse = 1.648 * exp(3.325*vlntemp - 2.55e+04*ortc);
    rr_f[305] = forward * mole_frac[3] * mole_frac[57];
    rr_r[305] = reverse * mole_frac[6] * mole_frac[58];
  }
  // 306)  O2 + IC4H9 <=> HO2 + IC4H8
  {
    double forward = 1.07 * exp(3.71*vlntemp - 9.322e+03*ortc);
    double reverse = 0.04158 * exp(4.024*vlntemp - 2.715e+04*ortc);
    rr_f[306] = forward * mole_frac[3] * mole_frac[56];
    rr_r[306] = reverse * mole_frac[6] * mole_frac[58];
  }
  // 307)  IC4H9O2 <=> O2 + IC4H9
  {
    double forward = 6.64e+19 * exp(-1.575*vlntemp - 3.608e+04*ortc);
    double reverse = 2.26e+12;
    rr_f[307] = forward * mole_frac[61];
    rr_r[307] = reverse * mole_frac[3] * mole_frac[56];
  }
  // 308)  TC4H9O2 <=> O2 + TC4H9
  {
    double forward = 3.331e+24 * exp(-2.472*vlntemp - 3.787e+04*ortc);
    double reverse = 1.41e+13;
    rr_f[308] = forward * mole_frac[60];
    rr_r[308] = reverse * mole_frac[3] * mole_frac[57];
  }
  // 309)  C3H6 + TC4H9O2 <=> C3H5-A + TC4H9O2H
  {
    double forward = 3.24e+11 * exp(-1.49e+04*ortc);
    double reverse = 2.0e+10 * exp(-1.5e+04*ortc);
    rr_f[309] = forward * mole_frac[41] * mole_frac[60];
    rr_r[309] = reverse * mole_frac[42] * mole_frac[64];
  }
  // 310)  IC4H8 + TC4H9O2 <=> IC4H7 + TC4H9O2H
  {
    double forward = 1.4e+12 * exp(-1.49e+04*ortc);
    double reverse = 3.16e+11 * exp(-1.3e+04*ortc);
    rr_f[310] = forward * mole_frac[58] * mole_frac[60];
    rr_r[310] = reverse * mole_frac[59] * mole_frac[64];
  }
  // 311)  C2H4 + TC4H9O2 <=> C2H3 + TC4H9O2H
  {
    double forward = 7.0e+11 * exp(-1.711e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.0e+04*ortc);
    rr_f[311] = forward * mole_frac[25] * mole_frac[60];
    rr_r[311] = reverse * mole_frac[26] * mole_frac[64];
  }
  // 312)  CH4 + TC4H9O2 <=> CH3 + TC4H9O2H
  {
    double forward = 1.13e+13 * exp(-2.046e+04*ortc);
    double reverse = 7.5e+08 * exp(-1.28e+03*ortc);
    rr_f[312] = forward * mole_frac[19] * mole_frac[60];
    rr_r[312] = reverse * mole_frac[20] * mole_frac[64];
  }
  // 313)  H2 + TC4H9O2 <=> H + TC4H9O2H
  {
    double forward = 3.01e+13 * exp(-2.603e+04*ortc);
    double reverse = 4.8e+13 * exp(-7.95e+03*ortc);
    rr_f[313] = forward * mole_frac[1] * mole_frac[60];
    rr_r[313] = reverse * mole_frac[0] * mole_frac[64];
  }
  // 314)  C2H6 + TC4H9O2 <=> C2H5 + TC4H9O2H
  {
    double forward = 1.7e+13 * exp(-2.046e+04*ortc);
    double reverse = 5.0e+11 * exp(-6.5e+03*ortc);
    rr_f[314] = forward * mole_frac[23] * mole_frac[60];
    rr_r[314] = reverse * mole_frac[24] * mole_frac[64];
  }
  // 315)  CH3CHO + TC4H9O2 <=> CH3CO + TC4H9O2H
  {
    double forward = 2.8e+12 * exp(-1.36e+04*ortc);
    double reverse = 1.0e+12 * exp(-1.0e+04*ortc);
    rr_f[315] = forward * mole_frac[28] * mole_frac[60];
    rr_r[315] = reverse * mole_frac[29] * mole_frac[64];
  }
  // 316)  C2H3CHO + TC4H9O2 <=> C2H3CO + TC4H9O2H
  {
    double forward = 2.8e+12 * exp(-1.36e+04*ortc);
    double reverse = 1.0e+12 * exp(-1.0e+04*ortc);
    rr_f[316] = forward * mole_frac[37] * mole_frac[60];
    rr_r[316] = reverse * mole_frac[38] * mole_frac[64];
  }
  // 317)  HO2 + TC4H9O2 <=> O2 + TC4H9O2H
  {
    double forward = 1.75e+10 * exp(3.275e+03*ortc);
    double reverse = 3.85e+13 * exp(-0.795*vlntemp - 3.362e+04*ortc);
    rr_f[317] = forward * mole_frac[6] * mole_frac[60];
    rr_r[317] = reverse * mole_frac[3] * mole_frac[64];
  }
  // 318)  H2O2 + TC4H9O2 <=> HO2 + TC4H9O2H
  {
    double forward = 2.4e+12 * exp(-1.0e+04*ortc);
    double reverse = 2.4e+12 * exp(-1.0e+04*ortc);
    rr_f[318] = forward * mole_frac[7] * mole_frac[60];
    rr_r[318] = reverse * mole_frac[6] * mole_frac[64];
  }
  // 319)  CH2O + TC4H9O2 <=> HCO + TC4H9O2H
  {
    double forward = 1.3e+11 * exp(-9.0e+03*ortc);
    double reverse = 2.5e+10 * exp(-1.01e+04*ortc);
    rr_f[319] = forward * mole_frac[10] * mole_frac[60];
    rr_r[319] = reverse * mole_frac[11] * mole_frac[64];
  }
  // 320)  CH3O2 + TC4H9O2 <=> O2 + CH3O + TC4H9O
  {
    double forward = 1.4e+16 * exp(-1.61*vlntemp - 1.86e+03*ortc);
    double reverse = 0.0;
    rr_f[320] = forward * mole_frac[18] * mole_frac[60];
    rr_r[320] = reverse * mole_frac[3] * mole_frac[16] * mole_frac[63];
  }
  // 321)  C2H5O2 + TC4H9O2 <=> O2 + C2H5O + TC4H9O
  {
    double forward = 1.4e+16 * exp(-1.61*vlntemp - 1.86e+03*ortc);
    double reverse = 0.0;
    rr_f[321] = forward * mole_frac[33] * mole_frac[60];
    rr_r[321] = reverse * mole_frac[3] * mole_frac[32] * mole_frac[63];
  }
  // 322)  2 TC4H9O2 <=> O2 + 2 TC4H9O
  {
    double forward = 1.4e+16 * exp(-1.61*vlntemp - 1.86e+03*ortc);
    double reverse = 0.0;
    rr_f[322] = forward * mole_frac[60] * mole_frac[60];
    rr_r[322] = reverse * mole_frac[3] * mole_frac[63] * mole_frac[63];
  }
  // 323)  HO2 + TC4H9O2 <=> O2 + OH + TC4H9O
  {
    double forward = 1.4e+16 * exp(-1.61*vlntemp - 1.86e+03*ortc);
    double reverse = 0.0;
    rr_f[323] = forward * mole_frac[6] * mole_frac[60];
    rr_r[323] = reverse * mole_frac[3] * mole_frac[4] * mole_frac[63];
  }
  // 324)  CH3 + TC4H9O2 <=> CH3O + TC4H9O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 7.824e+11 * exp(0.229*vlntemp - 2.834e+04*ortc);
    rr_f[324] = forward * mole_frac[20] * mole_frac[60];
    rr_r[324] = reverse * mole_frac[16] * mole_frac[63];
  }
  // 325)  C2H5 + TC4H9O2 <=> C2H5O + TC4H9O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 1.112e+14 * exp(-0.372*vlntemp - 3.075e+04*ortc);
    rr_f[325] = forward * mole_frac[24] * mole_frac[60];
    rr_r[325] = reverse * mole_frac[32] * mole_frac[63];
  }
  // 326)  TC4H9 + TC4H9O2 <=> 2 TC4H9O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 7.373e+16 * exp(-0.978*vlntemp - 3.275e+04*ortc);
    rr_f[326] = forward * mole_frac[57] * mole_frac[60];
    rr_r[326] = reverse * mole_frac[63] * mole_frac[63];
  }
  // 327)  C3H5-A + TC4H9O2 <=> C3H5O + TC4H9O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 2.898e+10 * exp(0.411*vlntemp - 1.576e+04*ortc);
    rr_f[327] = forward * mole_frac[42] * mole_frac[60];
    rr_r[327] = reverse * mole_frac[49] * mole_frac[63];
  }
  // 328)  IC4H7 + TC4H9O2 <=> TC4H9O + IC4H7O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 6.172e+10 * exp(0.401*vlntemp - 1.492e+04*ortc);
    rr_f[328] = forward * mole_frac[59] * mole_frac[60];
    rr_r[328] = reverse * mole_frac[63] * mole_frac[65];
  }
  // 329)  TC4H9O2H <=> OH + TC4H9O
  {
    double forward = 5.95e+15 * exp(-4.254e+04*ortc);
    double reverse = 6.677e+06 * exp(1.939*vlntemp + 2.582e+03*ortc);
    rr_f[329] = forward * mole_frac[64];
    rr_r[329] = reverse * mole_frac[4] * mole_frac[63];
  }
  // 330)  TC4H9O <=> CH3 + CH3COCH3
  {
    double forward = 9.558e+22 * exp(-2.548*vlntemp - 1.865e+04*ortc);
    double reverse = 1.5e+11 * exp(-1.19e+04*ortc);
    rr_f[330] = forward * mole_frac[63];
    rr_r[330] = reverse * mole_frac[20] * mole_frac[35];
  }
  // 331)  IC3H7CHO <=> H + TC3H6CHO
  {
    double forward = 2.304e+18 * exp(-0.91*vlntemp - 9.2e+04*ortc);
    double reverse = 2.0e+14;
    rr_f[331] = forward * mole_frac[66];
    rr_r[331] = reverse * mole_frac[0] * mole_frac[67];
  }
  // 332)  IC3H7CHO <=> HCO + IC3H7
  {
    double forward = 1.129e+17 * exp(-0.03*vlntemp - 7.976e+04*ortc);
    double reverse = 1.81e+13;
    rr_f[332] = forward * mole_frac[66];
    rr_r[332] = reverse * mole_frac[11] * mole_frac[40];
  }
  // 333)  HO2 + IC3H7CHO <=> H2O2 + TC3H6CHO
  {
    double forward = 8.0e+10 * exp(-1.192e+04*ortc);
    double reverse = 3.366e+12 * exp(-0.42*vlntemp - 1.105e+04*ortc);
    rr_f[333] = forward * mole_frac[6] * mole_frac[66];
    rr_r[333] = reverse * mole_frac[7] * mole_frac[67];
  }
  // 334)  OH + IC3H7CHO <=> H2O + TC3H6CHO
  {
    double forward = 1.684e+12 * exp(781.0*ortc);
    double reverse = 1.194e+13 * exp(-0.09*vlntemp - 2.981e+04*ortc);
    rr_f[334] = forward * mole_frac[4] * mole_frac[66];
    rr_r[334] = reverse * mole_frac[5] * mole_frac[67];
  }
  // 335)  IC4H9O2 <=> IC4H8O2H-I
  {
    double forward = 7.5e+10 * exp(-2.44e+04*ortc);
    double reverse = 1.815e+11 * exp(-0.507*vlntemp - 8.946e+03*ortc);
    rr_f[335] = forward * mole_frac[61];
    rr_r[335] = reverse * mole_frac[62];
  }
  // 336)  IC4H9O2 <=> HO2 + IC4H8
  {
    double forward = 2.265e+35 * exp(-7.22*vlntemp - 3.949e+04*ortc);
    double reverse = 2.996e+26 * exp(-5.331*vlntemp - 2.124e+04*ortc);
    rr_f[336] = forward * mole_frac[61];
    rr_r[336] = reverse * mole_frac[6] * mole_frac[58];
  }
  // 337)  TC4H9O2 <=> HO2 + IC4H8
  {
    double forward = 7.612e+42 * exp(-9.41*vlntemp - 4.149e+04*ortc);
    double reverse = 6.344e+31 * exp(-7.203*vlntemp - 1.716e+04*ortc);
    rr_f[337] = forward * mole_frac[60];
    rr_r[337] = reverse * mole_frac[6] * mole_frac[58];
  }
  // 338)  IC4H8OOH-IO2 <=> O2 + IC4H8O2H-I
  {
    double forward = 1.44e+20 * exp(-1.627*vlntemp - 3.569e+04*ortc);
    double reverse = 2.26e+12;
    rr_f[338] = forward * mole_frac[68];
    rr_r[338] = reverse * mole_frac[3] * mole_frac[62];
  }
  // 339)  IC4H8OOH-IO2 <=> OH + IC4KETII
  {
    double forward = 5.0e+10 * exp(-2.14e+04*ortc);
    double reverse = 1.986e+03 * exp(1.455*vlntemp - 4.442e+04*ortc);
    rr_f[339] = forward * mole_frac[68];
    rr_r[339] = reverse * mole_frac[4] * mole_frac[69];
  }
  // 340)  IC4KETII <=> OH + CH2O + C2H5CO
  {
    double forward = 1.5e+16 * exp(-4.2e+04*ortc);
    double reverse = 0.0;
    rr_f[340] = forward * mole_frac[69];
    rr_r[340] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[39];
  }
  // 341)  IC4H8O2H-I <=> OH + CH2O + C3H6
  {
    double forward = 8.451e+15 * exp(-0.68*vlntemp - 2.917e+04*ortc);
    double reverse = 0.0;
    rr_f[341] = forward * mole_frac[62];
    rr_r[341] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[41];
  }
  // 342)  IC4H8 <=> CH3 + C3H5-T
  {
    double forward = 1.92e+66 * exp(-14.22*vlntemp - 1.281e+05*ortc);
    double reverse = 1.561e+56 * exp(-12.293*vlntemp - 2.61e+04*ortc);
    rr_f[342] = forward * mole_frac[58];
    rr_r[342] = reverse * mole_frac[20] * mole_frac[44];
  }
  // 343)  IC4H8 <=> H + IC4H7
  {
    double forward = 3.07e+55 * exp(-11.49*vlntemp - 1.143e+05*ortc);
    double reverse = 1.428e+55 * exp(-11.738*vlntemp - 2.64e+04*ortc);
    rr_f[343] = forward * mole_frac[58];
    rr_r[343] = reverse * mole_frac[0] * mole_frac[59];
  }
  // 344)  H + IC4H8 <=> CH3 + C3H6
  {
    double forward = 5.68e+33 * exp(-5.72*vlntemp - 2.0e+04*ortc);
    double reverse = 6.093e+26 * exp(-4.209*vlntemp - 2.72e+04*ortc);
    rr_f[344] = forward * mole_frac[0] * mole_frac[58];
    rr_r[344] = reverse * mole_frac[20] * mole_frac[41];
  }
  // 345)  H + IC4H8 <=> H2 + IC4H7
  {
    double forward = 3.4e+05 * exp(2.5*vlntemp - 2.492e+03*ortc);
    double reverse = 6.32e+04 * exp(2.528*vlntemp - 1.816e+04*ortc);
    rr_f[345] = forward * mole_frac[0] * mole_frac[58];
    rr_r[345] = reverse * mole_frac[1] * mole_frac[59];
  }
  // 346)  O + IC4H8 <=> 2 CH3 + CH2CO
  {
    double forward = 3.33e+07 * exp(1.76*vlntemp - 76.0*ortc);
    double reverse = 0.0;
    rr_f[346] = forward * mole_frac[2] * mole_frac[58];
    rr_r[346] = reverse * mole_frac[20] * mole_frac[20] * mole_frac[30];
  }
  // 347)  O + IC4H8 <=> 2 H + IC3H6CO
  {
    double forward = 1.66e+07 * exp(1.76*vlntemp - 76.0*ortc);
    double reverse = 0.0;
    rr_f[347] = forward * mole_frac[2] * mole_frac[58];
    rr_r[347] = reverse * mole_frac[0] * mole_frac[0] * mole_frac[75];
  }
  // 348)  O + IC4H8 <=> OH + IC4H7
  {
    double forward = 1.206e+11 * exp(0.7*vlntemp - 7.633e+03*ortc);
    double reverse = 1.164e+10 * exp(0.709*vlntemp - 2.189e+04*ortc);
    rr_f[348] = forward * mole_frac[2] * mole_frac[58];
    rr_r[348] = reverse * mole_frac[4] * mole_frac[59];
  }
  // 349)  CH3 + IC4H8 <=> CH4 + IC4H7
  {
    double forward = 4.42 * exp(3.5*vlntemp - 5.675e+03*ortc);
    double reverse = 749.5 * exp(3.082*vlntemp - 2.289e+04*ortc);
    rr_f[349] = forward * mole_frac[20] * mole_frac[58];
    rr_r[349] = reverse * mole_frac[19] * mole_frac[59];
  }
  // 350)  HO2 + IC4H8 <=> H2O2 + IC4H7
  {
    double forward = 1.928e+04 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 2.073e+06 * exp(1.933*vlntemp - 1.358e+04*ortc);
    rr_f[350] = forward * mole_frac[6] * mole_frac[58];
    rr_r[350] = reverse * mole_frac[7] * mole_frac[59];
  }
  // 351)  O2CHO + IC4H8 <=> HO2CHO + IC4H7
  {
    double forward = 1.928e+04 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 6.514e-07 * exp(4.9*vlntemp + 3.468e+03*ortc);
    rr_f[351] = forward * mole_frac[13] * mole_frac[58];
    rr_r[351] = reverse * mole_frac[12] * mole_frac[59];
  }
  // 352)  O2 + IC4H8 <=> HO2 + IC4H7
  {
    double forward = 6.0e+12 * exp(-3.99e+04*ortc);
    double reverse = 5.848e+12 * exp(-0.32*vlntemp - 883.0*ortc);
    rr_f[352] = forward * mole_frac[3] * mole_frac[58];
    rr_r[352] = reverse * mole_frac[6] * mole_frac[59];
  }
  // 353)  C3H5-A + IC4H8 <=> C3H6 + IC4H7
  {
    double forward = 7.94e+11 * exp(-2.05e+04*ortc);
    double reverse = 4.4e+20 * exp(-1.33*vlntemp - 6.061e+04*ortc);
    rr_f[353] = forward * mole_frac[42] * mole_frac[58];
    rr_r[353] = reverse * mole_frac[41] * mole_frac[59];
  }
  // 354)  C3H5-S + IC4H8 <=> C3H6 + IC4H7
  {
    double forward = 7.94e+11 * exp(-2.05e+04*ortc);
    double reverse = 5.592e+20 * exp(-1.27*vlntemp - 8.217e+04*ortc);
    rr_f[354] = forward * mole_frac[43] * mole_frac[58];
    rr_r[354] = reverse * mole_frac[41] * mole_frac[59];
  }
  // 355)  C3H5-T + IC4H8 <=> C3H6 + IC4H7
  {
    double forward = 7.94e+11 * exp(-2.05e+04*ortc);
    double reverse = 5.592e+20 * exp(-1.27*vlntemp - 8.017e+04*ortc);
    rr_f[355] = forward * mole_frac[44] * mole_frac[58];
    rr_r[355] = reverse * mole_frac[41] * mole_frac[59];
  }
  // 356)  OH + IC4H8 <=> H2O + IC4H7
  {
    double forward = 5.2e+06 * temperature * temperature * exp(298.0*ortc);
    double reverse = 1.025e+07 * exp(1.922*vlntemp - 3.027e+04*ortc);
    rr_f[356] = forward * mole_frac[4] * mole_frac[58];
    rr_r[356] = reverse * mole_frac[5] * mole_frac[59];
  }
  // 357)  O + IC4H8 <=> HCO + IC3H7
  {
    double forward = 1.58e+07 * exp(1.76*vlntemp + 1.216e+03*ortc);
    double reverse = 4.538 * exp(3.06*vlntemp - 2.169e+04*ortc);
    rr_f[357] = forward * mole_frac[2] * mole_frac[58];
    rr_r[357] = reverse * mole_frac[11] * mole_frac[40];
  }
  // 358)  CH3O2 + IC4H8 <=> CH3O2H + IC4H7
  {
    double forward = 1.928e+04 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 4.034e+07 * exp(1.488*vlntemp - 1.199e+04*ortc);
    rr_f[358] = forward * mole_frac[18] * mole_frac[58];
    rr_r[358] = reverse * mole_frac[17] * mole_frac[59];
  }
  // 359)  O2 + IC4H7 <=> OH + IC3H5CHO
  {
    double forward = 2.47e+13 * exp(-0.45*vlntemp - 2.302e+04*ortc);
    double reverse = 3.372e+13 * exp(-0.577*vlntemp - 7.301e+04*ortc);
    rr_f[359] = forward * mole_frac[3] * mole_frac[59];
    rr_r[359] = reverse * mole_frac[4] * mole_frac[72];
  }
  // 360)  O2 + IC4H7 <=> CH2O + CH3COCH2
  {
    double forward = 7.14e+15 * exp(-1.21*vlntemp - 2.105e+04*ortc);
    double reverse = 1.7e+12 * exp(-0.407*vlntemp - 8.825e+04*ortc);
    rr_f[360] = forward * mole_frac[3] * mole_frac[59];
    rr_r[360] = reverse * mole_frac[10] * mole_frac[36];
  }
  // 361)  O2 + IC4H7 <=> OH + CH2O + C3H4-A
  {
    double forward = 7.29e+29 * exp(-5.71*vlntemp - 2.145e+04*ortc);
    double reverse = 0.0;
    rr_f[361] = forward * mole_frac[3] * mole_frac[59];
    rr_r[361] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[46];
  }
  // 362)  O + IC4H7 <=> H + IC3H5CHO
  {
    double forward = 6.03e+13;
    double reverse = 2.844e+16 * exp(-0.519*vlntemp - 6.673e+04*ortc);
    rr_f[362] = forward * mole_frac[2] * mole_frac[59];
    rr_r[362] = reverse * mole_frac[0] * mole_frac[72];
  }
  // 363)  IC4H7 <=> CH3 + C3H4-A
  {
    double forward = 1.23e+47 * exp(-9.74*vlntemp - 7.426e+04*ortc);
    double reverse = 1.649e+38 * exp(-7.768*vlntemp - 2.254e+04*ortc);
    rr_f[363] = forward * mole_frac[59];
    rr_r[363] = reverse * mole_frac[20] * mole_frac[46];
  }
  // 364)  CH3O2 + IC4H7 <=> CH3O + IC4H7O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 2.138e+11 * exp(0.349*vlntemp - 1.506e+04*ortc);
    rr_f[364] = forward * mole_frac[18] * mole_frac[59];
    rr_r[364] = reverse * mole_frac[16] * mole_frac[65];
  }
  // 365)  HO2 + IC4H7 <=> OH + IC4H7O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 3.418e+12 * exp(0.05*vlntemp - 1.082e+04*ortc);
    rr_f[365] = forward * mole_frac[6] * mole_frac[59];
    rr_r[365] = reverse * mole_frac[4] * mole_frac[65];
  }
  // 366)  IC4H7O <=> CH2O + C3H5-T
  {
    double forward = 2.925e+21 * exp(-2.391*vlntemp - 3.559e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.26e+04*ortc);
    rr_f[366] = forward * mole_frac[65];
    rr_r[366] = reverse * mole_frac[10] * mole_frac[44];
  }
  // 367)  IC4H7O <=> IC4H6OH
  {
    double forward = 1.391e+11 * exp(-1.56e+04*ortc);
    double reverse = 4.233e+11 * exp(-0.164*vlntemp - 3.167e+04*ortc);
    rr_f[367] = forward * mole_frac[65];
    rr_r[367] = reverse * mole_frac[71];
  }
  // 368)  IC4H7O <=> H + IC3H5CHO
  {
    double forward = 5.0e+13 * exp(-2.91e+04*ortc);
    double reverse = 6.67e+13 * exp(-0.105*vlntemp - 1.841e+04*ortc);
    rr_f[368] = forward * mole_frac[65];
    rr_r[368] = reverse * mole_frac[0] * mole_frac[72];
  }
  // 369)  H2 + IC4H6OH <=> H + IC4H7OH
  {
    double forward = 2.16e+04 * exp(2.38*vlntemp - 1.899e+04*ortc);
    double reverse = 561.4 * exp(2.98*vlntemp - 1.399e+03*ortc);
    rr_f[369] = forward * mole_frac[1] * mole_frac[71];
    rr_r[369] = reverse * mole_frac[0] * mole_frac[70];
  }
  // 370)  HO2 + IC4H6OH <=> O2 + IC4H7OH
  {
    double forward = 5.57e+13 * exp(-0.315*vlntemp - 862.0*ortc);
    double reverse = 6.0e+13 * exp(-3.99e+04*ortc);
    rr_f[370] = forward * mole_frac[6] * mole_frac[71];
    rr_r[370] = reverse * mole_frac[3] * mole_frac[70];
  }
  // 371)  CH2O + IC4H6OH <=> HCO + IC4H7OH
  {
    double forward = 6.3e+08 * exp(1.9*vlntemp - 1.819e+04*ortc);
    double reverse = 2.101e+07 * exp(2.153*vlntemp - 1.773e+04*ortc);
    rr_f[371] = forward * mole_frac[10] * mole_frac[71];
    rr_r[371] = reverse * mole_frac[11] * mole_frac[70];
  }
  // 372)  IC4H8 + IC4H6OH <=> IC4H7 + IC4H7OH
  {
    double forward = 470.0 * exp(3.3*vlntemp - 1.984e+04*ortc);
    double reverse = 0.2814 * exp(3.9*vlntemp - 6.521e+03*ortc);
    rr_f[372] = forward * mole_frac[58] * mole_frac[71];
    rr_r[372] = reverse * mole_frac[59] * mole_frac[70];
  }
  // 373)  IC4H7OH <=> H + IC4H6OH
  {
    double forward = 4.902e+16 * exp(-0.4*vlntemp - 8.985e+04*ortc);
    double reverse = 1.0e+14;
    rr_f[373] = forward * mole_frac[70];
    rr_r[373] = reverse * mole_frac[0] * mole_frac[71];
  }
  // 374)  HO2 + IC4H7OH <=> H2O2 + IC4H6OH
  {
    double forward = 7.644e+03 * exp(2.712*vlntemp - 1.393e+04*ortc);
    double reverse = 7.83e+05 * exp(2.05*vlntemp - 1.358e+04*ortc);
    rr_f[374] = forward * mole_frac[6] * mole_frac[70];
    rr_r[374] = reverse * mole_frac[7] * mole_frac[71];
  }
  // 375)  IC4H6OH <=> CH2OH + C3H4-A
  {
    double forward = 7.244e+19 * exp(-1.859*vlntemp - 5.705e+04*ortc);
    double reverse = 1.0e+11 * exp(-9.2e+03*ortc);
    rr_f[375] = forward * mole_frac[71];
    rr_r[375] = reverse * mole_frac[15] * mole_frac[46];
  }
  // 376)  O2 + IC4H7O <=> HO2 + IC3H5CHO
  {
    double forward = 3.0e+10 * exp(-1.649e+03*ortc);
    double reverse = 6.312e+10 * exp(-0.14*vlntemp - 3.898e+04*ortc);
    rr_f[376] = forward * mole_frac[3] * mole_frac[65];
    rr_r[376] = reverse * mole_frac[6] * mole_frac[72];
  }
  // 377)  HO2 + IC4H7O <=> H2O2 + IC3H5CHO
  {
    double forward = 3.0e+11;
    double reverse = 8.93e+14 * exp(-0.8*vlntemp - 7.85e+04*ortc);
    rr_f[377] = forward * mole_frac[6] * mole_frac[65];
    rr_r[377] = reverse * mole_frac[7] * mole_frac[72];
  }
  // 378)  CH3 + IC4H7O <=> CH4 + IC3H5CHO
  {
    double forward = 2.4e+13;
    double reverse = 7.261e+16 * exp(-0.47*vlntemp - 9.529e+04*ortc);
    rr_f[378] = forward * mole_frac[20] * mole_frac[65];
    rr_r[378] = reverse * mole_frac[19] * mole_frac[72];
  }
  // 379)  O + IC4H7O <=> OH + IC3H5CHO
  {
    double forward = 6.0e+12;
    double reverse = 3.052e+14 * exp(-0.47*vlntemp - 9.272e+04*ortc);
    rr_f[379] = forward * mole_frac[2] * mole_frac[65];
    rr_r[379] = reverse * mole_frac[4] * mole_frac[72];
  }
  // 380)  OH + IC4H7O <=> H2O + IC3H5CHO
  {
    double forward = 1.81e+13;
    double reverse = 9.076e+15 * exp(-0.47*vlntemp - 1.1e+05*ortc);
    rr_f[380] = forward * mole_frac[4] * mole_frac[65];
    rr_r[380] = reverse * mole_frac[5] * mole_frac[72];
  }
  // 381)  H + IC4H7O <=> H2 + IC3H5CHO
  {
    double forward = 1.99e+13;
    double reverse = 2.305e+15 * exp(-0.47*vlntemp - 9.481e+04*ortc);
    rr_f[381] = forward * mole_frac[0] * mole_frac[65];
    rr_r[381] = reverse * mole_frac[1] * mole_frac[72];
  }
  // 382)  OH + IC3H5CHO <=> H2O + IC3H5CO
  {
    double forward = 2.69e+10 * exp(0.76*vlntemp + 340.0*ortc);
    double reverse = 4.4e+10 * exp(0.78*vlntemp - 3.608e+04*ortc);
    rr_f[382] = forward * mole_frac[4] * mole_frac[72];
    rr_r[382] = reverse * mole_frac[5] * mole_frac[73];
  }
  // 383)  HO2 + IC3H5CHO <=> H2O2 + IC3H5CO
  {
    double forward = 1.0e+12 * exp(-1.192e+04*ortc);
    double reverse = 9.709e+12 * exp(-0.31*vlntemp - 1.688e+04*ortc);
    rr_f[383] = forward * mole_frac[6] * mole_frac[72];
    rr_r[383] = reverse * mole_frac[7] * mole_frac[73];
  }
  // 384)  CH3 + IC3H5CHO <=> CH4 + IC3H5CO
  {
    double forward = 3.98e+12 * exp(-8.7e+03*ortc);
    double reverse = 3.928e+13 * exp(0.02*vlntemp - 3.045e+04*ortc);
    rr_f[384] = forward * mole_frac[20] * mole_frac[72];
    rr_r[384] = reverse * mole_frac[19] * mole_frac[73];
  }
  // 385)  O + IC3H5CHO <=> OH + IC3H5CO
  {
    double forward = 7.18e+12 * exp(-1.389e+03*ortc);
    double reverse = 1.191e+12 * exp(0.02*vlntemp - 2.056e+04*ortc);
    rr_f[385] = forward * mole_frac[2] * mole_frac[72];
    rr_r[385] = reverse * mole_frac[4] * mole_frac[73];
  }
  // 386)  O2 + IC3H5CHO <=> HO2 + IC3H5CO
  {
    double forward = 2.0e+13 * exp(-4.07e+04*ortc);
    double reverse = 1.824e+11 * exp(0.311*vlntemp - 5.337e+03*ortc);
    rr_f[386] = forward * mole_frac[3] * mole_frac[72];
    rr_r[386] = reverse * mole_frac[6] * mole_frac[73];
  }
  // 387)  H + IC3H5CHO <=> H2 + IC3H5CO
  {
    double forward = 2.6e+12 * exp(-2.6e+03*ortc);
    double reverse = 9.822e+11 * exp(0.02*vlntemp - 2.387e+04*ortc);
    rr_f[387] = forward * mole_frac[0] * mole_frac[72];
    rr_r[387] = reverse * mole_frac[1] * mole_frac[73];
  }
  // 388)  IC3H5CO <=> CO + C3H5-T
  {
    double forward = 1.278e+20 * exp(-1.89*vlntemp - 3.446e+04*ortc);
    double reverse = 1.51e+11 * exp(-4.809e+03*ortc);
    rr_f[388] = forward * mole_frac[73];
    rr_r[388] = reverse * mole_frac[8] * mole_frac[44];
  }
  // 389)  HO2 + TC3H6CHO <=> OH + TC3H6OCHO
  {
    double forward = 9.64e+12;
    double reverse = 2.018e+17 * exp(-1.2*vlntemp - 2.101e+04*ortc);
    rr_f[389] = forward * mole_frac[6] * mole_frac[67];
    rr_r[389] = reverse * mole_frac[4] * mole_frac[74];
  }
  // 390)  TC3H6OCHO <=> HCO + CH3COCH3
  {
    double forward = 3.98e+13 * exp(-9.7e+03*ortc);
    double reverse = 2.173e+08 * exp(0.8*vlntemp - 1.424e+04*ortc);
    rr_f[390] = forward * mole_frac[74];
    rr_r[390] = reverse * mole_frac[11] * mole_frac[35];
  }
  // 391)  TC3H6CHO <=> H + IC3H5CHO
  {
    double forward = 1.325e+14 * exp(0.01*vlntemp - 3.934e+04*ortc);
    double reverse = 1.3e+13 * exp(-1.2e+03*ortc);
    rr_f[391] = forward * mole_frac[67];
    rr_r[391] = reverse * mole_frac[0] * mole_frac[72];
  }
  // 392)  TC3H6CHO <=> H + IC3H6CO
  {
    double forward = 4.086e+14 * exp(-0.07199999999999999*vlntemp - 
      4.241e+04*ortc); 
    double reverse = 1.3e+13 * exp(-4.8e+03*ortc);
    rr_f[392] = forward * mole_frac[67];
    rr_r[392] = reverse * mole_frac[0] * mole_frac[75];
  }
  // 393)  H2 + TC3H6CHO <=> H + IC3H7CHO
  {
    double forward = 2.16e+05 * exp(2.38*vlntemp - 1.899e+04*ortc);
    double reverse = 1.319e+05 * exp(2.47*vlntemp - 3.55e+03*ortc);
    rr_f[393] = forward * mole_frac[1] * mole_frac[67];
    rr_r[393] = reverse * mole_frac[0] * mole_frac[66];
  }
  // 394)  IC4H7OOH <=> OH + IC4H7O
  {
    double forward = 6.4e+15 * exp(-4.555e+04*ortc);
    double reverse = 1.0e+11;
    rr_f[394] = forward * mole_frac[76];
    rr_r[394] = reverse * mole_frac[4] * mole_frac[65];
  }
  // 395)  IC4H7OH <=> H + IC4H7O
  {
    double forward = 5.969e+16 * exp(-0.5600000000000001*vlntemp - 
      1.059e+05*ortc); 
    double reverse = 4.0e+13;
    rr_f[395] = forward * mole_frac[70];
    rr_r[395] = reverse * mole_frac[0] * mole_frac[65];
  }
  // 396)  H2 + IC4H7O <=> H + IC4H7OH
  {
    double forward = 9.05e+06 * temperature * temperature * 
      exp(-1.783e+04*ortc); 
    double reverse = 7.16e+05 * exp(2.44*vlntemp - 1.631e+04*ortc);
    rr_f[396] = forward * mole_frac[1] * mole_frac[65];
    rr_r[396] = reverse * mole_frac[0] * mole_frac[70];
  }
  // 397)  IC4H7OH <=> OH + IC4H7
  {
    double forward = 7.31e+16 * exp(-0.41*vlntemp - 7.97e+04*ortc);
    double reverse = 3.0e+13;
    rr_f[397] = forward * mole_frac[70];
    rr_r[397] = reverse * mole_frac[4] * mole_frac[59];
  }
  // 398)  CH2O + IC4H7O <=> HCO + IC4H7OH
  {
    double forward = 1.15e+11 * exp(-1.28e+03*ortc);
    double reverse = 3.02e+11 * exp(-1.816e+04*ortc);
    rr_f[398] = forward * mole_frac[10] * mole_frac[65];
    rr_r[398] = reverse * mole_frac[11] * mole_frac[70];
  }
  // 399)  CH2O + TC3H6CHO <=> HCO + IC3H7CHO
  {
    double forward = 2.52e+08 * exp(1.9*vlntemp - 1.819e+04*ortc);
    double reverse = 1.229e+07 * exp(1.99*vlntemp - 1.742e+04*ortc);
    rr_f[399] = forward * mole_frac[10] * mole_frac[67];
    rr_r[399] = reverse * mole_frac[11] * mole_frac[66];
  }
  // 400)  IC4H8 + TC3H6CHO <=> IC4H7 + IC3H7CHO
  {
    double forward = 470.0 * exp(3.3*vlntemp - 1.984e+04*ortc);
    double reverse = 6.613 * exp(3.39*vlntemp - 8.672e+03*ortc);
    rr_f[400] = forward * mole_frac[58] * mole_frac[67];
    rr_r[400] = reverse * mole_frac[59] * mole_frac[66];
  }
  // 401)  OH + IC3H6CO <=> CO2 + IC3H7
  {
    double forward = 1.73e+12 * exp(1.01e+03*ortc);
    double reverse = 2.577e+14 * exp(-0.43*vlntemp - 5.548e+04*ortc);
    rr_f[401] = forward * mole_frac[4] * mole_frac[75];
    rr_r[401] = reverse * mole_frac[9] * mole_frac[40];
  }
  // 402)  TC3H6O2CHO <=> O2 + TC3H6CHO
  {
    double forward = 2.458e+25 * exp(-4.065*vlntemp - 2.708e+04*ortc);
    double reverse = 1.99e+17 * exp(-2.1 * vlntemp);
    rr_f[402] = forward * mole_frac[77];
    rr_r[402] = reverse * mole_frac[3] * mole_frac[67];
  }
  // 403)  O2 + TC3H6CHO <=> HO2 + IC3H5CHO
  {
    double forward = 2.725e-19 * exp(-7.24e+03*ortc);
    double reverse = 1.39e+11 * exp(-0.2*vlntemp - 1.731e+04*ortc);
    rr_f[403] = forward * mole_frac[3] * mole_frac[67];
    rr_r[403] = reverse * mole_frac[6] * mole_frac[72];
  }
  // 404)  O2 + TC3H6CHO <=> OH + CO + CH3COCH3
  {
    double forward = 3.62e-20;
    double reverse = 0.0;
    rr_f[404] = forward * mole_frac[3] * mole_frac[67];
    rr_r[404] = reverse * mole_frac[4] * mole_frac[8] * mole_frac[35];
  }
  // 405)  HO2 + TC3H6CHO <=> O2 + IC3H7CHO
  {
    double forward = 3.675e+12 * exp(-1.31e+03*ortc);
    double reverse = 1.236e+14 * exp(-0.24*vlntemp - 4.335e+04*ortc);
    rr_f[405] = forward * mole_frac[6] * mole_frac[67];
    rr_r[405] = reverse * mole_frac[3] * mole_frac[66];
  }
  // 406)  CH3 + TC3H6CHO <=> CH4 + IC3H5CHO
  {
    double forward = 3.01e+12 * exp(-0.32*vlntemp + 131.0*ortc);
    double reverse = 2.207e+15 * exp(-0.85*vlntemp - 6.79e+04*ortc);
    rr_f[406] = forward * mole_frac[20] * mole_frac[67];
    rr_r[406] = reverse * mole_frac[19] * mole_frac[72];
  }
  // 407)  IC4H8 + IC4H7O <=> IC4H7 + IC4H7OH
  {
    double forward = 2.7e+11 * exp(-4.0e+03*ortc);
    double reverse = 1.0e+10 * exp(-9.0e+03*ortc);
    rr_f[407] = forward * mole_frac[58] * mole_frac[65];
    rr_r[407] = reverse * mole_frac[59] * mole_frac[70];
  }
  // 408)  HO2 + IC4H6OH <=> OH + CH2O + CH2CCH2OH
  {
    double forward = 1.446e+13;
    double reverse = 0.0;
    rr_f[408] = forward * mole_frac[6] * mole_frac[71];
    rr_r[408] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[78];
  }
  // 409)  IC4H7OH <=> CH3 + CH2CCH2OH
  {
    double forward = 1.247e+20 * exp(-0.98*vlntemp - 9.857e+04*ortc);
    double reverse = 3.0e+13;
    rr_f[409] = forward * mole_frac[70];
    rr_r[409] = reverse * mole_frac[20] * mole_frac[78];
  }
  // 410)  O2 + CH2CCH2OH <=> CO + CH2O + CH2OH
  {
    double forward = 4.335e+12;
    double reverse = 0.0;
    rr_f[410] = forward * mole_frac[3] * mole_frac[78];
    rr_r[410] = reverse * mole_frac[8] * mole_frac[10] * mole_frac[15];
  }
  // 411)  CH2CCH2OH <=> CH2OH + C2H2
  {
    double forward = 2.163e+40 * exp(-8.31*vlntemp - 4.511e+04*ortc);
    double reverse = 1.61e+40 * exp(-8.58*vlntemp - 2.033e+04*ortc);
    rr_f[411] = forward * mole_frac[78];
    rr_r[411] = reverse * mole_frac[15] * mole_frac[27];
  }
  // 412)  CH2CCH2OH <=> OH + C3H4-A
  {
    double forward = 6.697e+16 * exp(-1.11*vlntemp - 4.258e+04*ortc);
    double reverse = 8.5e+12 * exp(-2.0e+03*ortc);
    rr_f[412] = forward * mole_frac[78];
    rr_r[412] = reverse * mole_frac[4] * mole_frac[46];
  }
  // 413)  BC5H11 <=> CH3 + IC4H8
  {
    double forward = 5.272e+10 * exp(1.192*vlntemp - 3.022e+04*ortc);
    double reverse = 4.4e+04 * exp(2.48*vlntemp - 6.13e+03*ortc);
    rr_f[413] = forward * mole_frac[79];
    rr_r[413] = reverse * mole_frac[20] * mole_frac[58];
  }
  // 414)  BC5H11 <=> H + AC5H10
  {
    double forward = 3.665e+11 * exp(0.732*vlntemp - 3.715e+04*ortc);
    double reverse = 1.06e+12 * exp(0.51*vlntemp - 1.23e+03*ortc);
    rr_f[414] = forward * mole_frac[79];
    rr_r[414] = reverse * mole_frac[0] * mole_frac[80];
  }
  // 415)  BC5H11 <=> H + BC5H10
  {
    double forward = 6.171e+11 * exp(0.487*vlntemp - 3.558e+04*ortc);
    double reverse = 6.25e+11 * exp(0.51*vlntemp - 2.62e+03*ortc);
    rr_f[415] = forward * mole_frac[79];
    rr_r[415] = reverse * mole_frac[0] * mole_frac[81];
  }
  // 416)  O2 + BC5H11 <=> HO2 + AC5H10
  {
    double forward = 2.0e-18 * exp(-5.0e+03*ortc);
    double reverse = 2.0e-19 * exp(-1.75e+04*ortc);
    rr_f[416] = forward * mole_frac[3] * mole_frac[79];
    rr_r[416] = reverse * mole_frac[6] * mole_frac[80];
  }
  // 417)  O2 + BC5H11 <=> HO2 + BC5H10
  {
    double forward = 2.0e-18 * exp(-5.0e+03*ortc);
    double reverse = 2.0e-19 * exp(-1.75e+04*ortc);
    rr_f[417] = forward * mole_frac[3] * mole_frac[79];
    rr_r[417] = reverse * mole_frac[6] * mole_frac[81];
  }
  // 418)  AC5H10 <=> CH3 + IC4H7
  {
    double forward = 1.9e+20 * exp(-1.582*vlntemp - 7.593e+04*ortc);
    double reverse = 2.55e+13 * exp(-0.32*vlntemp + 131.0*ortc);
    rr_f[418] = forward * mole_frac[80];
    rr_r[418] = reverse * mole_frac[20] * mole_frac[59];
  }
  // 419)  AC5H10 <=> C2H5 + C3H5-T
  {
    double forward = 8.922e+24 * exp(-2.409*vlntemp - 1.005e+05*ortc);
    double reverse = 1.0e+13;
    rr_f[419] = forward * mole_frac[80];
    rr_r[419] = reverse * mole_frac[24] * mole_frac[44];
  }
  // 420)  BC5H10 <=> CH3 + IC4H7
  {
    double forward = 2.61e+19 * exp(-1.017*vlntemp - 7.902e+04*ortc);
    double reverse = 1.0e+13;
    rr_f[420] = forward * mole_frac[81];
    rr_r[420] = reverse * mole_frac[20] * mole_frac[59];
  }
  // 421)  OH + AC5H10 <=> CH2O + SC4H9
  {
    double forward = 2.0e+10 * exp(-4.0e+03*ortc);
    double reverse = 2.0e+13 * exp(-2.0e+04*ortc);
    rr_f[421] = forward * mole_frac[4] * mole_frac[80];
    rr_r[421] = reverse * mole_frac[10] * mole_frac[55];
  }
  // 422)  OH + BC5H10 <=> CH3CHO + IC3H7
  {
    double forward = 2.0e+10 * exp(-4.0e+03*ortc);
    double reverse = 2.0e+13 * exp(-2.0e+04*ortc);
    rr_f[422] = forward * mole_frac[4] * mole_frac[81];
    rr_r[422] = reverse * mole_frac[28] * mole_frac[40];
  }
  // 423)  OH + CC5H10 <=> CH2O + IC4H9
  {
    double forward = 2.0e+10 * exp(-4.0e+03*ortc);
    double reverse = 2.0e+13 * exp(-2.0e+04*ortc);
    rr_f[423] = forward * mole_frac[4] * mole_frac[82];
    rr_r[423] = reverse * mole_frac[10] * mole_frac[56];
  }
  // 424)  O + AC5H10 <=> HCO + SC4H9
  {
    double forward = 7.23e+05 * exp(2.34*vlntemp + 1.05e+03*ortc);
    double reverse = 2.0e+05 * exp(2.34*vlntemp - 8.03e+04*ortc);
    rr_f[424] = forward * mole_frac[2] * mole_frac[80];
    rr_r[424] = reverse * mole_frac[11] * mole_frac[55];
  }
  // 425)  O + AC5H10 <=> CH3CO + IC3H7
  {
    double forward = 7.23e+05 * exp(2.34*vlntemp + 1.05e+03*ortc);
    double reverse = 2.0e+05 * exp(2.34*vlntemp - 8.03e+04*ortc);
    rr_f[425] = forward * mole_frac[2] * mole_frac[80];
    rr_r[425] = reverse * mole_frac[29] * mole_frac[40];
  }
  // 426)  O + AC5H10 <=> HCO + IC4H9
  {
    double forward = 7.23e+05 * exp(2.34*vlntemp + 1.05e+03*ortc);
    double reverse = 2.0e+05 * exp(2.34*vlntemp - 8.03e+04*ortc);
    rr_f[426] = forward * mole_frac[2] * mole_frac[80];
    rr_r[426] = reverse * mole_frac[11] * mole_frac[56];
  }
  // 427)  H + AC5H10 <=> H2 + AC5H9-C
  {
    double forward = 3.376e+05 * exp(2.36*vlntemp - 207.0*ortc);
    double reverse = 4.352e+06 * exp(2.1*vlntemp - 2.033e+04*ortc);
    rr_f[427] = forward * mole_frac[0] * mole_frac[80];
    rr_r[427] = reverse * mole_frac[1] * mole_frac[83];
  }
  // 428)  OH + AC5H10 <=> H2O + AC5H9-C
  {
    double forward = 2.764e+04 * exp(2.64*vlntemp + 1.919e+03*ortc);
    double reverse = 1.543e+06 * exp(2.38*vlntemp - 3.336e+04*ortc);
    rr_f[428] = forward * mole_frac[4] * mole_frac[80];
    rr_r[428] = reverse * mole_frac[5] * mole_frac[83];
  }
  // 429)  CH3 + AC5H10 <=> CH4 + AC5H9-C
  {
    double forward = 3.69 * exp(3.31*vlntemp - 4.002e+03*ortc);
    double reverse = 1.243e+03 * exp(3.05*vlntemp - 2.46e+04*ortc);
    rr_f[429] = forward * mole_frac[20] * mole_frac[80];
    rr_r[429] = reverse * mole_frac[19] * mole_frac[83];
  }
  // 430)  HO2 + AC5H10 <=> H2O2 + AC5H9-C
  {
    double forward = 4.82e+03 * exp(2.55*vlntemp - 1.053e+04*ortc);
    double reverse = 1.597e+06 * exp(1.96*vlntemp - 1.434e+04*ortc);
    rr_f[430] = forward * mole_frac[6] * mole_frac[80];
    rr_r[430] = reverse * mole_frac[7] * mole_frac[83];
  }
  // 431)  CH3O2 + AC5H10 <=> CH3O2H + AC5H9-C
  {
    double forward = 4.82e+03 * exp(2.55*vlntemp - 1.053e+04*ortc);
    double reverse = 3.326e+06 * exp(1.79*vlntemp - 1.132e+04*ortc);
    rr_f[431] = forward * mole_frac[18] * mole_frac[80];
    rr_r[431] = reverse * mole_frac[17] * mole_frac[83];
  }
  // 432)  H + BC5H10 <=> H2 + AC5H9-C
  {
    double forward = 3.46e+05 * exp(2.5*vlntemp - 2.492e+03*ortc);
    double reverse = 1.274e+07 * temperature * temperature * 
      exp(-1.965e+04*ortc); 
    rr_f[432] = forward * mole_frac[0] * mole_frac[81];
    rr_r[432] = reverse * mole_frac[1] * mole_frac[83];
  }
  // 433)  H + BC5H10 <=> H2 + CC5H9-B
  {
    double forward = 1.73e+05 * exp(2.5*vlntemp - 2.492e+03*ortc);
    double reverse = 7.021e+06 * exp(2.17*vlntemp - 2.04e+04*ortc);
    rr_f[433] = forward * mole_frac[0] * mole_frac[81];
    rr_r[433] = reverse * mole_frac[1] * mole_frac[84];
  }
  // 434)  OH + BC5H10 <=> H2O + AC5H9-C
  {
    double forward = 6.24e+06 * temperature * temperature * exp(298.0*ortc);
    double reverse = 9.945e+08 * exp(1.5*vlntemp - 3.202e+04*ortc);
    rr_f[434] = forward * mole_frac[4] * mole_frac[81];
    rr_r[434] = reverse * mole_frac[5] * mole_frac[83];
  }
  // 435)  OH + BC5H10 <=> H2O + CC5H9-B
  {
    double forward = 3.12e+06 * temperature * temperature * exp(298.0*ortc);
    double reverse = 5.482e+08 * exp(1.67*vlntemp - 3.277e+04*ortc);
    rr_f[435] = forward * mole_frac[4] * mole_frac[81];
    rr_r[435] = reverse * mole_frac[5] * mole_frac[84];
  }
  // 436)  CH3 + BC5H10 <=> CH4 + AC5H9-C
  {
    double forward = 4.42 * exp(3.5*vlntemp - 5.675e+03*ortc);
    double reverse = 4.25e+03 * exp(3.0*vlntemp - 2.332e+04*ortc);
    rr_f[436] = forward * mole_frac[20] * mole_frac[81];
    rr_r[436] = reverse * mole_frac[19] * mole_frac[83];
  }
  // 437)  CH3 + BC5H10 <=> CH4 + CC5H9-B
  {
    double forward = 2.21 * exp(3.5*vlntemp - 5.675e+03*ortc);
    double reverse = 2.343e+03 * exp(3.17*vlntemp - 2.406e+04*ortc);
    rr_f[437] = forward * mole_frac[20] * mole_frac[81];
    rr_r[437] = reverse * mole_frac[19] * mole_frac[84];
  }
  // 438)  HO2 + BC5H10 <=> H2O2 + AC5H9-C
  {
    double forward = 1.928e+04 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 1.824e+07 * exp(1.77*vlntemp - 1.476e+04*ortc);
    rr_f[438] = forward * mole_frac[6] * mole_frac[81];
    rr_r[438] = reverse * mole_frac[7] * mole_frac[83];
  }
  // 439)  HO2 + BC5H10 <=> H2O2 + CC5H9-B
  {
    double forward = 9.639e+03 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 1.005e+07 * exp(1.94*vlntemp - 1.551e+04*ortc);
    rr_f[439] = forward * mole_frac[6] * mole_frac[81];
    rr_r[439] = reverse * mole_frac[7] * mole_frac[84];
  }
  // 440)  CH3O2 + BC5H10 <=> CH3O2H + AC5H9-C
  {
    double forward = 1.928e+04 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 3.798e+07 * exp(1.59*vlntemp - 1.174e+04*ortc);
    rr_f[440] = forward * mole_frac[18] * mole_frac[81];
    rr_r[440] = reverse * mole_frac[17] * mole_frac[83];
  }
  // 441)  CH3O2 + BC5H10 <=> CH3O2H + CC5H9-B
  {
    double forward = 9.639e+03 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 2.094e+07 * exp(1.77*vlntemp - 1.249e+04*ortc);
    rr_f[441] = forward * mole_frac[18] * mole_frac[81];
    rr_r[441] = reverse * mole_frac[17] * mole_frac[84];
  }
  // 442)  H + CC5H10 <=> H2 + CC5H9-B
  {
    double forward = 2.65e+06 * exp(2.2 * vlntemp);
    double reverse = 1.821e+07 * exp(2.14*vlntemp - 2.228e+04*ortc);
    rr_f[442] = forward * mole_frac[0] * mole_frac[82];
    rr_r[442] = reverse * mole_frac[1] * mole_frac[84];
  }
  // 443)  OH + CC5H10 <=> H2O + CC5H9-B
  {
    double forward = 614.0 * exp(3.2*vlntemp + 3.5e+03*ortc);
    double reverse = 1.827e+04 * exp(3.14*vlntemp - 3.394e+04*ortc);
    rr_f[443] = forward * mole_frac[4] * mole_frac[82];
    rr_r[443] = reverse * mole_frac[5] * mole_frac[84];
  }
  // 444)  CH3 + CC5H10 <=> CH4 + CC5H9-B
  {
    double forward = 4.613 * exp(3.1*vlntemp - 2.33e+03*ortc);
    double reverse = 828.0 * exp(3.04*vlntemp - 2.509e+04*ortc);
    rr_f[444] = forward * mole_frac[20] * mole_frac[82];
    rr_r[444] = reverse * mole_frac[19] * mole_frac[84];
  }
  // 445)  HO2 + CC5H10 <=> H2O2 + CC5H9-B
  {
    double forward = 1.81e+03 * exp(2.5*vlntemp - 7.154e+03*ortc);
    double reverse = 3.196e+05 * exp(2.11*vlntemp - 1.313e+04*ortc);
    rr_f[445] = forward * mole_frac[6] * mole_frac[82];
    rr_r[445] = reverse * mole_frac[7] * mole_frac[84];
  }
  // 446)  CH3O2 + CC5H10 <=> CH3O2H + CC5H9-B
  {
    double forward = 1.81e+03 * exp(2.5*vlntemp - 7.154e+03*ortc);
    double reverse = 6.656e+05 * exp(1.93*vlntemp - 1.011e+04*ortc);
    rr_f[446] = forward * mole_frac[18] * mole_frac[82];
    rr_r[446] = reverse * mole_frac[17] * mole_frac[84];
  }
  // 447)  HO2 + AC5H9-C <=> OH + AC5H9O-C
  {
    double forward = 9.64e+12;
    double reverse = 2.731e+15 * exp(-0.96*vlntemp - 1.562e+04*ortc);
    rr_f[447] = forward * mole_frac[6] * mole_frac[83];
    rr_r[447] = reverse * mole_frac[4] * mole_frac[85];
  }
  // 448)  CH3O2 + AC5H9-C <=> CH3O + AC5H9O-C
  {
    double forward = 9.64e+12;
    double reverse = 2.668e+17 * exp(-1.53*vlntemp - 2.038e+04*ortc);
    rr_f[448] = forward * mole_frac[18] * mole_frac[83];
    rr_r[448] = reverse * mole_frac[16] * mole_frac[85];
  }
  // 449)  C2H5O2 + AC5H9-C <=> C2H5O + AC5H9O-C
  {
    double forward = 9.64e+12;
    double reverse = 1.746e+14 * exp(-0.61*vlntemp - 1.822e+04*ortc);
    rr_f[449] = forward * mole_frac[33] * mole_frac[83];
    rr_r[449] = reverse * mole_frac[32] * mole_frac[85];
  }
  // 450)  HO2 + CC5H9-B <=> OH + CC5H9O-B
  {
    double forward = 9.64e+12;
    double reverse = 2.939e+15 * exp(-1.02*vlntemp - 1.687e+04*ortc);
    rr_f[450] = forward * mole_frac[6] * mole_frac[84];
    rr_r[450] = reverse * mole_frac[4] * mole_frac[86];
  }
  // 451)  CH3O2 + CC5H9-B <=> CH3O + CC5H9O-B
  {
    double forward = 9.64e+12;
    double reverse = 2.871e+17 * exp(-1.59*vlntemp - 2.164e+04*ortc);
    rr_f[451] = forward * mole_frac[18] * mole_frac[84];
    rr_r[451] = reverse * mole_frac[16] * mole_frac[86];
  }
  // 452)  C2H5O2 + CC5H9-B <=> C2H5O + CC5H9O-B
  {
    double forward = 9.64e+12;
    double reverse = 1.879e+14 * exp(-0.67*vlntemp - 1.948e+04*ortc);
    rr_f[452] = forward * mole_frac[33] * mole_frac[84];
    rr_r[452] = reverse * mole_frac[32] * mole_frac[86];
  }
  // 453)  AC5H9O-C <=> CH3CHO + C3H5-T
  {
    double forward = 3.231e+22 * exp(-2.63*vlntemp - 3.031e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.19e+04*ortc);
    rr_f[453] = forward * mole_frac[85];
    rr_r[453] = reverse * mole_frac[28] * mole_frac[44];
  }
  // 454)  CC5H9O-B <=> C2H3 + CH3COCH3
  {
    double forward = 7.813e+13 * exp(-0.25*vlntemp - 2.233e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.19e+04*ortc);
    rr_f[454] = forward * mole_frac[86];
    rr_r[454] = reverse * mole_frac[26] * mole_frac[35];
  }
  // 455)  CH3CHCHO <=> H + C2H3CHO
  {
    double forward = 3.515e+15 * exp(-0.51*vlntemp - 4.106e+04*ortc);
    double reverse = 6.5e+12 * exp(-2.9e+03*ortc);
    rr_f[455] = forward * mole_frac[87];
    rr_r[455] = reverse * mole_frac[0] * mole_frac[37];
  }
  // 456)  CH3CHCHO <=> H + CH3CHCO
  {
    double forward = 1.135e+16 * exp(-0.66*vlntemp - 4.031e+04*ortc);
    double reverse = 5.0e+12 * exp(-1.2e+03*ortc);
    rr_f[456] = forward * mole_frac[87];
    rr_r[456] = reverse * mole_frac[0] * mole_frac[54];
  }
  // 457)  OH + CC6H12 <=> CH3CHO + IC4H9
  {
    double forward = 1.0e+11 * exp(4.0e+03*ortc);
    double reverse = 0.0;
    rr_f[457] = forward * mole_frac[4] * mole_frac[89];
    rr_r[457] = reverse * mole_frac[28] * mole_frac[56];
  }
  // 458)  O + CC6H12 <=> CH3CO + IC4H9
  {
    double forward = 1.0e+11 * exp(1.05e+03*ortc);
    double reverse = 0.0;
    rr_f[458] = forward * mole_frac[2] * mole_frac[89];
    rr_r[458] = reverse * mole_frac[29] * mole_frac[56];
  }
  // 459)  BC6H12 <=> CH3 + CC5H9-B
  {
    double forward = 1.0e+16 * exp(-7.1e+04*ortc);
    double reverse = 1.0e+13;
    rr_f[459] = forward * mole_frac[88];
    rr_r[459] = reverse * mole_frac[20] * mole_frac[84];
  }
  // 460)  O + C5H10-2 <=> CH3CHO + C3H6
  {
    double forward = 1.0e+10;
    double reverse = 1.0e+12 * exp(-8.1e+04*ortc);
    rr_f[460] = forward * mole_frac[2] * mole_frac[90];
    rr_r[460] = reverse * mole_frac[28] * mole_frac[41];
  }
  // 461)  IC4H7-I1 <=> CH3 + C3H4-P
  {
    double forward = 2.103e+12 * exp(0.08*vlntemp - 2.995e+04*ortc);
    double reverse = 1.0e+11 * exp(-9.2e+03*ortc);
    rr_f[461] = forward * mole_frac[91];
    rr_r[461] = reverse * mole_frac[20] * mole_frac[45];
  }
  // 462)  YC7H15 <=> IC3H7 + IC4H8
  {
    double forward = 2.22e+20 * exp(-2.06*vlntemp - 3.247e+04*ortc);
    double reverse = 5.0e+10 * exp(-9.2e+03*ortc);
    rr_f[462] = forward * mole_frac[92];
    rr_r[462] = reverse * mole_frac[40] * mole_frac[58];
  }
  // 463)  YC7H15 <=> H + XC7H14
  {
    double forward = 1.437e+13 * exp(0.23*vlntemp - 3.769e+04*ortc);
    double reverse = 2.6e+13 * exp(-1.2e+03*ortc);
    rr_f[463] = forward * mole_frac[92];
    rr_r[463] = reverse * mole_frac[0] * mole_frac[93];
  }
  // 464)  YC7H15 <=> H + YC7H14
  {
    double forward = 3.093e+13 * exp(0.049*vlntemp - 3.639e+04*ortc);
    double reverse = 2.6e+13 * exp(-2.5e+03*ortc);
    rr_f[464] = forward * mole_frac[92];
    rr_r[464] = reverse * mole_frac[0] * mole_frac[94];
  }
  // 465)  O2 + YC7H15 <=> HO2 + XC7H14
  {
    double forward = 6.000000000000001e-29 * exp(-5.02e+03*ortc);
    double reverse = 2.0e-29 * exp(-1.75e+04*ortc);
    rr_f[465] = forward * mole_frac[3] * mole_frac[92];
    rr_r[465] = reverse * mole_frac[6] * mole_frac[93];
  }
  // 466)  O2 + YC7H15 <=> HO2 + YC7H14
  {
    double forward = 3.0e-29 * exp(-3.0e+03*ortc);
    double reverse = 2.0e-29 * exp(-1.75e+04*ortc);
    rr_f[466] = forward * mole_frac[3] * mole_frac[92];
    rr_r[466] = reverse * mole_frac[6] * mole_frac[94];
  }
  // 467)  XC7H14 <=> IC3H7 + IC4H7
  {
    double forward = 2.211e+24 * exp(-2.392*vlntemp - 7.467e+04*ortc);
    double reverse = 1.28e+14 * exp(-0.35 * vlntemp);
    rr_f[467] = forward * mole_frac[93];
    rr_r[467] = reverse * mole_frac[40] * mole_frac[59];
  }
  // 468)  OH + XC7H14 <=> CH3COCH3 + IC4H9
  {
    double forward = 2.0e+10 * exp(4.0e+03*ortc);
    double reverse = 0.0;
    rr_f[468] = forward * mole_frac[4] * mole_frac[93];
    rr_r[468] = reverse * mole_frac[35] * mole_frac[56];
  }
  // 469)  OH + YC7H14 <=> CH3COCH3 + IC4H9
  {
    double forward = 2.0e+10 * exp(4.0e+03*ortc);
    double reverse = 0.0;
    rr_f[469] = forward * mole_frac[4] * mole_frac[94];
    rr_r[469] = reverse * mole_frac[35] * mole_frac[56];
  }
  // 470)  O + XC7H14 <=> CH2O + CC6H12
  {
    double forward = 2.0e+10 * exp(1.05e+03*ortc);
    double reverse = 0.0;
    rr_f[470] = forward * mole_frac[2] * mole_frac[93];
    rr_r[470] = reverse * mole_frac[10] * mole_frac[89];
  }
  // 471)  O + YC7H14 <=> CH3COCH3 + IC4H8
  {
    double forward = 2.0e+10 * exp(1.05e+03*ortc);
    double reverse = 0.0;
    rr_f[471] = forward * mole_frac[2] * mole_frac[94];
    rr_r[471] = reverse * mole_frac[35] * mole_frac[58];
  }
  // 472)  H + XC7H14 <=> H2 + XC7H13-Z
  {
    double forward = 3.376e+05 * exp(2.36*vlntemp - 207.0*ortc);
    double reverse = 4.418e+06 * exp(2.1*vlntemp - 2.037e+04*ortc);
    rr_f[472] = forward * mole_frac[0] * mole_frac[93];
    rr_r[472] = reverse * mole_frac[1] * mole_frac[95];
  }
  // 473)  OH + XC7H14 <=> H2O + XC7H13-Z
  {
    double forward = 2.764e+04 * exp(2.64*vlntemp + 1.919e+03*ortc);
    double reverse = 1.566e+06 * exp(2.38*vlntemp - 3.34e+04*ortc);
    rr_f[473] = forward * mole_frac[4] * mole_frac[93];
    rr_r[473] = reverse * mole_frac[5] * mole_frac[95];
  }
  // 474)  CH3 + XC7H14 <=> CH4 + XC7H13-Z
  {
    double forward = 3.69 * exp(3.31*vlntemp - 4.002e+03*ortc);
    double reverse = 1.262e+03 * exp(3.05*vlntemp - 2.464e+04*ortc);
    rr_f[474] = forward * mole_frac[20] * mole_frac[93];
    rr_r[474] = reverse * mole_frac[19] * mole_frac[95];
  }
  // 475)  HO2 + XC7H14 <=> H2O2 + XC7H13-Z
  {
    double forward = 4.82e+03 * exp(2.55*vlntemp - 1.053e+04*ortc);
    double reverse = 1.621e+06 * exp(1.96*vlntemp - 1.438e+04*ortc);
    rr_f[475] = forward * mole_frac[6] * mole_frac[93];
    rr_r[475] = reverse * mole_frac[7] * mole_frac[95];
  }
  // 476)  CH3O2 + XC7H14 <=> CH3O2H + XC7H13-Z
  {
    double forward = 4.82e+03 * exp(2.55*vlntemp - 1.053e+04*ortc);
    double reverse = 3.376e+06 * exp(1.79*vlntemp - 1.136e+04*ortc);
    rr_f[476] = forward * mole_frac[18] * mole_frac[93];
    rr_r[476] = reverse * mole_frac[17] * mole_frac[95];
  }
  // 477)  H + YC7H14 <=> H2 + XC7H13-Z
  {
    double forward = 3.46e+05 * exp(2.5*vlntemp - 2.492e+03*ortc);
    double reverse = 9.749e+06 * exp(2.06*vlntemp - 2.005e+04*ortc);
    rr_f[477] = forward * mole_frac[0] * mole_frac[94];
    rr_r[477] = reverse * mole_frac[1] * mole_frac[95];
  }
  // 478)  H + YC7H14 <=> H2 + YC7H13-Y2
  {
    double forward = 2.65e+06 * exp(2.2 * vlntemp);
    double reverse = 1.836e+07 * exp(2.14*vlntemp - 2.229e+04*ortc);
    rr_f[478] = forward * mole_frac[0] * mole_frac[94];
    rr_r[478] = reverse * mole_frac[1] * mole_frac[96];
  }
  // 479)  OH + YC7H14 <=> H2O + XC7H13-Z
  {
    double forward = 6.24e+06 * temperature * temperature * exp(298.0*ortc);
    double reverse = 7.613e+08 * exp(1.56*vlntemp - 3.242e+04*ortc);
    rr_f[479] = forward * mole_frac[4] * mole_frac[94];
    rr_r[479] = reverse * mole_frac[5] * mole_frac[95];
  }
  // 480)  OH + YC7H14 <=> H2O + YC7H13-Y2
  {
    double forward = 614.0 * exp(3.2*vlntemp + 3.5e+03*ortc);
    double reverse = 1.842e+04 * exp(3.14*vlntemp - 3.394e+04*ortc);
    rr_f[480] = forward * mole_frac[4] * mole_frac[94];
    rr_r[480] = reverse * mole_frac[5] * mole_frac[96];
  }
  // 481)  XC7H13-Z <=> CH3 + ACC6H10
  {
    double forward = 2.837e+16 * exp(-0.83*vlntemp - 4.123e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.06e+04*ortc);
    rr_f[481] = forward * mole_frac[95];
    rr_r[481] = reverse * mole_frac[20] * mole_frac[99];
  }
  // 482)  HO2 + YC7H13-Y2 <=> OH + YC7H13O-Y2
  {
    double forward = 9.64e+12;
    double reverse = 2.111e+15 * exp(-0.97*vlntemp - 1.7e+04*ortc);
    rr_f[482] = forward * mole_frac[6] * mole_frac[96];
    rr_r[482] = reverse * mole_frac[4] * mole_frac[97];
  }
  // 483)  CH3O2 + YC7H13-Y2 <=> CH3O + YC7H13O-Y2
  {
    double forward = 9.64e+12;
    double reverse = 2.062e+17 * exp(-1.55*vlntemp - 2.176e+04*ortc);
    rr_f[483] = forward * mole_frac[18] * mole_frac[96];
    rr_r[483] = reverse * mole_frac[16] * mole_frac[97];
  }
  // 484)  C2H5O2 + YC7H13-Y2 <=> C2H5O + YC7H13O-Y2
  {
    double forward = 9.64e+12;
    double reverse = 1.35e+14 * exp(-0.62*vlntemp - 1.961e+04*ortc);
    rr_f[484] = forward * mole_frac[33] * mole_frac[96];
    rr_r[484] = reverse * mole_frac[32] * mole_frac[97];
  }
  // 485)  YC7H13O-Y2 <=> CH3COCH3 + IC4H7-I1
  {
    double forward = 1.31e+18 * exp(-1.3*vlntemp - 2.942e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.19e+04*ortc);
    rr_f[485] = forward * mole_frac[97];
    rr_r[485] = reverse * mole_frac[35] * mole_frac[91];
  }
  // 486)  YC7H15O2 <=> O2 + YC7H15
  {
    double forward = 3.408e+23 * exp(-2.448*vlntemp - 3.721e+04*ortc);
    double reverse = 3.0e+12;
    rr_f[486] = forward * mole_frac[98];
    rr_r[486] = reverse * mole_frac[3] * mole_frac[92];
  }
  // 487)  YC7H15O2 <=> HO2 + XC7H14
  {
    double forward = 1.015e+43 * exp(-9.41*vlntemp - 4.149e+04*ortc);
    double reverse = 3.387e+32 * exp(-7.264*vlntemp - 1.666e+04*ortc);
    rr_f[487] = forward * mole_frac[98];
    rr_r[487] = reverse * mole_frac[6] * mole_frac[93];
  }
  // 488)  YC7H15O2 <=> HO2 + YC7H14
  {
    double forward = 5.044e+38 * exp(-8.109999999999999*vlntemp - 
      4.049e+04*ortc); 
    double reverse = 7.817e+27 * exp(-5.783*vlntemp - 1.826e+04*ortc);
    rr_f[488] = forward * mole_frac[98];
    rr_r[488] = reverse * mole_frac[6] * mole_frac[94];
  }
  // 489)  OH + ACC6H10 <=> H2O + ACC6H9-A
  {
    double forward = 3.12e+06 * temperature * temperature * exp(298.0*ortc);
    double reverse = 5.491e+08 * exp(1.39*vlntemp - 3.246e+04*ortc);
    rr_f[489] = forward * mole_frac[4] * mole_frac[99];
    rr_r[489] = reverse * mole_frac[5] * mole_frac[100];
  }
  // 490)  OH + ACC6H10 <=> H2O + ACC6H9-D
  {
    double forward = 3.12e+06 * temperature * temperature * exp(298.0*ortc);
    double reverse = 5.491e+08 * exp(1.39*vlntemp - 3.246e+04*ortc);
    rr_f[490] = forward * mole_frac[4] * mole_frac[99];
    rr_r[490] = reverse * mole_frac[5] * mole_frac[101];
  }
  // 491)  HO2 + ACC6H10 <=> H2O2 + ACC6H9-A
  {
    double forward = 9.64e+03 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 1.007e+07 * exp(1.66*vlntemp - 1.521e+04*ortc);
    rr_f[491] = forward * mole_frac[6] * mole_frac[99];
    rr_r[491] = reverse * mole_frac[7] * mole_frac[100];
  }
  // 492)  HO2 + ACC6H10 <=> H2O2 + ACC6H9-D
  {
    double forward = 9.64e+03 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 1.007e+07 * exp(1.66*vlntemp - 1.521e+04*ortc);
    rr_f[492] = forward * mole_frac[6] * mole_frac[99];
    rr_r[492] = reverse * mole_frac[7] * mole_frac[101];
  }
  // 493)  CH3O2 + ACC6H10 <=> CH3O2H + ACC6H9-A
  {
    double forward = 9.64e+03 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 4.139e+07 * exp(1.49*vlntemp - 1.449e+04*ortc);
    rr_f[493] = forward * mole_frac[18] * mole_frac[99];
    rr_r[493] = reverse * mole_frac[17] * mole_frac[100];
  }
  // 494)  CH3O2 + ACC6H10 <=> CH3O2H + ACC6H9-D
  {
    double forward = 9.64e+03 * exp(2.6*vlntemp - 1.391e+04*ortc);
    double reverse = 4.139e+07 * exp(1.49*vlntemp - 1.449e+04*ortc);
    rr_f[494] = forward * mole_frac[18] * mole_frac[99];
    rr_r[494] = reverse * mole_frac[17] * mole_frac[101];
  }
  // 495)  ACC6H9-A <=> C3H5-S + C3H4-A
  {
    double forward = 1.194e+24 * exp(-2.85*vlntemp - 7.431e+04*ortc);
    double reverse = 1.0e+11 * exp(-9.2e+03*ortc);
    rr_f[495] = forward * mole_frac[100];
    rr_r[495] = reverse * mole_frac[43] * mole_frac[46];
  }
  // 496)  HO2 + ACC6H9-D <=> OH + C2H3 + IC3H5CHO
  {
    double forward = 8.91e+12;
    double reverse = 0.0;
    rr_f[496] = forward * mole_frac[6] * mole_frac[101];
    rr_r[496] = reverse * mole_frac[4] * mole_frac[26] * mole_frac[72];
  }
  // 497)  NEOC5H11 <=> CH3 + IC4H8
  {
    double forward = 8.466e+17 * exp(-1.111*vlntemp - 3.293e+04*ortc);
    double reverse = 1.3e+03 * exp(2.48*vlntemp - 8.52e+03*ortc);
    rr_f[497] = forward * mole_frac[102];
    rr_r[497] = reverse * mole_frac[20] * mole_frac[58];
  }
  // 498)  NEOC5H11O2 <=> O2 + NEOC5H11
  {
    double forward = 9.747e+20 * exp(-2.437*vlntemp - 3.453e+04*ortc);
    double reverse = 1.99e+17 * exp(-2.1 * vlntemp);
    rr_f[498] = forward * mole_frac[103];
    rr_r[498] = reverse * mole_frac[3] * mole_frac[102];
  }
  // 499)  NEOC5H11O2 <=> NEOC5H10OOH
  {
    double forward = 1.125e+11 * exp(-2.44e+04*ortc);
    double reverse = 9.144e+10 * exp(-0.509*vlntemp - 8.95e+03*ortc);
    rr_f[499] = forward * mole_frac[103];
    rr_r[499] = reverse * mole_frac[104];
  }
  // 500)  NEOC5H10OOH <=> OH + CH2O + IC4H8
  {
    double forward = 3.011e+17 * exp(-1.17*vlntemp - 2.995e+04*ortc);
    double reverse = 0.0;
    rr_f[500] = forward * mole_frac[104];
    rr_r[500] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[58];
  }
  // 501)  NEOC5H10OOH <=> CH3 + IC4H7OOH
  {
    double forward = 9.027e+21 * exp(-2.341*vlntemp - 3.283e+04*ortc);
    double reverse = 1.25e+11 * exp(-1.01e+04*ortc);
    rr_f[501] = forward * mole_frac[104];
    rr_r[501] = reverse * mole_frac[20] * mole_frac[76];
  }
  // 502)  HO2 + TC4H9CHO <=> H2O2 + TC4H9CO
  {
    double forward = 1.0e+12 * exp(-1.192e+04*ortc);
    double reverse = 3.852e+12 * exp(-0.33*vlntemp - 1.2e+04*ortc);
    rr_f[502] = forward * mole_frac[6] * mole_frac[105];
    rr_r[502] = reverse * mole_frac[7] * mole_frac[106];
  }
  // 503)  CH3 + TC4H9CHO <=> CH4 + TC4H9CO
  {
    double forward = 3.98e+12 * exp(-8.7e+03*ortc);
    double reverse = 1.558e+13 * exp(-2.557e+04*ortc);
    rr_f[503] = forward * mole_frac[20] * mole_frac[105];
    rr_r[503] = reverse * mole_frac[19] * mole_frac[106];
  }
  // 504)  O + TC4H9CHO <=> OH + TC4H9CO
  {
    double forward = 7.18e+12 * exp(-1.389e+03*ortc);
    double reverse = 4.726e+11 * exp(-1.568e+04*ortc);
    rr_f[504] = forward * mole_frac[2] * mole_frac[105];
    rr_r[504] = reverse * mole_frac[4] * mole_frac[106];
  }
  // 505)  O2 + TC4H9CHO <=> HO2 + TC4H9CO
  {
    double forward = 4.0e+13 * exp(-3.76e+04*ortc);
    double reverse = 1.089e+11 * exp(0.32*vlntemp + 3.492e+03*ortc);
    rr_f[505] = forward * mole_frac[3] * mole_frac[105];
    rr_r[505] = reverse * mole_frac[6] * mole_frac[106];
  }
  // 506)  OH + TC4H9CHO <=> H2O + TC4H9CO
  {
    double forward = 2.69e+10 * exp(0.76*vlntemp + 340.0*ortc);
    double reverse = 1.746e+10 * exp(0.76*vlntemp - 3.12e+04*ortc);
    rr_f[506] = forward * mole_frac[4] * mole_frac[105];
    rr_r[506] = reverse * mole_frac[5] * mole_frac[106];
  }
  // 507)  TC4H9CO <=> CO + TC4H9
  {
    double forward = 2.517e+23 * exp(-2.881*vlntemp - 1.349e+04*ortc);
    double reverse = 1.5e+11 * exp(-4.81e+03*ortc);
    rr_f[507] = forward * mole_frac[106];
    rr_r[507] = reverse * mole_frac[8] * mole_frac[57];
  }
  // 508)  IC8H18 <=> H + AC8H17
  {
    double forward = 5.748e+17 * exp(-0.36*vlntemp - 1.012e+05*ortc);
    double reverse = 1.0e+14;
    rr_f[508] = forward * mole_frac[107];
    rr_r[508] = reverse * mole_frac[0] * mole_frac[108];
  }
  // 509)  IC8H18 <=> H + BC8H17
  {
    double forward = 3.299e+18 * exp(-0.721*vlntemp - 9.873e+04*ortc);
    double reverse = 1.0e+14;
    rr_f[509] = forward * mole_frac[107];
    rr_r[509] = reverse * mole_frac[0] * mole_frac[109];
  }
  // 510)  IC8H18 <=> H + CC8H17
  {
    double forward = 1.146e+19 * exp(-0.9409999999999999*vlntemp - 
      9.543e+04*ortc); 
    double reverse = 1.0e+14;
    rr_f[510] = forward * mole_frac[107];
    rr_r[510] = reverse * mole_frac[0] * mole_frac[110];
  }
  // 511)  IC8H18 <=> H + DC8H17
  {
    double forward = 1.919e+17 * exp(-0.36*vlntemp - 1.004e+05*ortc);
    double reverse = 1.0e+14;
    rr_f[511] = forward * mole_frac[107];
    rr_r[511] = reverse * mole_frac[0] * mole_frac[111];
  }
  // 512)  IC8H18 <=> CH3 + YC7H15
  {
    double forward = 1.635e+27 * exp(-2.794*vlntemp - 8.393e+04*ortc);
    double reverse = 1.63e+13 * exp(596.0*ortc);
    rr_f[512] = forward * mole_frac[107];
    rr_r[512] = reverse * mole_frac[20] * mole_frac[92];
  }
  // 513)  IC8H18 <=> IC4H9 + TC4H9
  {
    double forward = 7.828e+29 * exp(-3.925*vlntemp - 8.415e+04*ortc);
    double reverse = 3.59e+14 * exp(-0.75 * vlntemp);
    rr_f[513] = forward * mole_frac[107];
    rr_r[513] = reverse * mole_frac[56] * mole_frac[57];
  }
  // 514)  IC8H18 <=> IC3H7 + NEOC5H11
  {
    double forward = 2.455e+23 * exp(-2.013*vlntemp - 8.34e+04*ortc);
    double reverse = 3.59e+14 * exp(-0.75 * vlntemp);
    rr_f[514] = forward * mole_frac[107];
    rr_r[514] = reverse * mole_frac[40] * mole_frac[102];
  }
  // 515)  H + IC8H18 <=> H2 + AC8H17
  {
    double forward = 7.341e+05 * exp(2.768*vlntemp - 8.147e+03*ortc);
    double reverse = 51.0 * exp(3.404*vlntemp - 1.048e+04*ortc);
    rr_f[515] = forward * mole_frac[0] * mole_frac[107];
    rr_r[515] = reverse * mole_frac[1] * mole_frac[108];
  }
  // 516)  H + IC8H18 <=> H2 + BC8H17
  {
    double forward = 5.736e+05 * exp(2.491*vlntemp - 4.124e+03*ortc);
    double reverse = 6.942 * exp(3.488*vlntemp - 8.954e+03*ortc);
    rr_f[516] = forward * mole_frac[0] * mole_frac[107];
    rr_r[516] = reverse * mole_frac[1] * mole_frac[109];
  }
  // 517)  H + IC8H18 <=> H2 + CC8H17
  {
    double forward = 6.02e+05 * exp(2.4*vlntemp - 2.583e+03*ortc);
    double reverse = 2.097 * exp(3.617*vlntemp - 1.071e+04*ortc);
    rr_f[517] = forward * mole_frac[0] * mole_frac[107];
    rr_r[517] = reverse * mole_frac[1] * mole_frac[110];
  }
  // 518)  H + IC8H18 <=> H2 + DC8H17
  {
    double forward = 1.88e+05 * exp(2.75*vlntemp - 6.28e+03*ortc);
    double reverse = 39.11 * exp(3.386*vlntemp - 9.417e+03*ortc);
    rr_f[518] = forward * mole_frac[0] * mole_frac[107];
    rr_r[518] = reverse * mole_frac[1] * mole_frac[111];
  }
  // 519)  O + IC8H18 <=> OH + AC8H17
  {
    double forward = 8.55e+03 * exp(3.05*vlntemp - 3.123e+03*ortc);
    double reverse = 0.3118 * exp(3.666*vlntemp - 4.048e+03*ortc);
    rr_f[519] = forward * mole_frac[2] * mole_frac[107];
    rr_r[519] = reverse * mole_frac[4] * mole_frac[108];
  }
  // 520)  O + IC8H18 <=> OH + BC8H17
  {
    double forward = 4.77e+04 * exp(2.71*vlntemp - 2.106e+03*ortc);
    double reverse = 0.303 * exp(3.687*vlntemp - 5.524e+03*ortc);
    rr_f[520] = forward * mole_frac[2] * mole_frac[107];
    rr_r[520] = reverse * mole_frac[4] * mole_frac[109];
  }
  // 521)  O + IC8H18 <=> OH + CC8H17
  {
    double forward = 3.83e+05 * exp(2.41*vlntemp - 1.14e+03*ortc);
    double reverse = 0.7003 * exp(3.607*vlntemp - 7.858e+03*ortc);
    rr_f[521] = forward * mole_frac[2] * mole_frac[107];
    rr_r[521] = reverse * mole_frac[4] * mole_frac[110];
  }
  // 522)  O + IC8H18 <=> OH + DC8H17
  {
    double forward = 2.853e+05 * exp(2.5*vlntemp - 3.645e+03*ortc);
    double reverse = 31.16 * exp(3.116*vlntemp - 5.37e+03*ortc);
    rr_f[522] = forward * mole_frac[2] * mole_frac[107];
    rr_r[522] = reverse * mole_frac[4] * mole_frac[111];
  }
  // 523)  OH + IC8H18 <=> H2O + AC8H17
  {
    double forward = 2.63e+07 * exp(1.8*vlntemp - 1.431e+03*ortc);
    double xik = -cgspl[4] + cgspl[5] - cgspl[107] + cgspl[108];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[523] = forward * mole_frac[4] * mole_frac[107];
    rr_r[523] = reverse * mole_frac[5] * mole_frac[108];
  }
  // 524)  OH + IC8H18 <=> H2O + BC8H17
  {
    double forward = 9.0e+05 * temperature * temperature * exp(1.133e+03*ortc);
    double reverse = 115.9 * exp(2.891*vlntemp - 1.859e+04*ortc);
    rr_f[524] = forward * mole_frac[4] * mole_frac[107];
    rr_r[524] = reverse * mole_frac[5] * mole_frac[109];
  }
  // 525)  OH + IC8H18 <=> H2O + CC8H17
  {
    double forward = 1.7e+06 * exp(1.9*vlntemp + 1.45e+03*ortc);
    double reverse = 63.01 * exp(3.011*vlntemp - 2.157e+04*ortc);
    rr_f[525] = forward * mole_frac[4] * mole_frac[107];
    rr_r[525] = reverse * mole_frac[5] * mole_frac[110];
  }
  // 526)  OH + IC8H18 <=> H2O + DC8H17
  {
    double forward = 1.78e+07 * exp(1.8*vlntemp - 1.431e+03*ortc);
    double reverse = 3.94e+04 * exp(2.33*vlntemp - 1.946e+04*ortc);
    rr_f[526] = forward * mole_frac[4] * mole_frac[107];
    rr_r[526] = reverse * mole_frac[5] * mole_frac[111];
  }
  // 527)  CH3 + IC8H18 <=> CH4 + AC8H17
  {
    double forward = 4.257e-14 * exp(8.06*vlntemp - 4.154e+03*ortc);
    double reverse = 2.699e-15 * exp(8.25*vlntemp - 8.031e+03*ortc);
    rr_f[527] = forward * mole_frac[20] * mole_frac[107];
    rr_r[527] = reverse * mole_frac[19] * mole_frac[108];
  }
  // 528)  CH3 + IC8H18 <=> CH4 + BC8H17
  {
    double forward = 2.705e+04 * exp(2.26*vlntemp - 7.287e+03*ortc);
    double reverse = 298.8 * exp(2.811*vlntemp - 1.366e+04*ortc);
    rr_f[528] = forward * mole_frac[20] * mole_frac[107];
    rr_r[528] = reverse * mole_frac[19] * mole_frac[109];
  }
  // 529)  CH3 + IC8H18 <=> CH4 + CC8H17
  {
    double forward = 6.01e-10 * exp(6.36*vlntemp - 893.0*ortc);
    double reverse = 1.911e-12 * exp(7.131*vlntemp - 1.056e+04*ortc);
    rr_f[529] = forward * mole_frac[20] * mole_frac[107];
    rr_r[529] = reverse * mole_frac[19] * mole_frac[110];
  }
  // 530)  CH3 + IC8H18 <=> CH4 + DC8H17
  {
    double forward = 0.147 * exp(3.87*vlntemp - 6.808e+03*ortc);
    double reverse = 0.02791 * exp(4.06*vlntemp - 1.148e+04*ortc);
    rr_f[530] = forward * mole_frac[20] * mole_frac[107];
    rr_r[530] = reverse * mole_frac[19] * mole_frac[111];
  }
  // 531)  HO2 + IC8H18 <=> H2O2 + AC8H17
  {
    double forward = 61.2 * exp(3.59*vlntemp - 1.716e+04*ortc);
    double xik = -cgspl[6] + cgspl[7] - cgspl[107] + cgspl[108];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[531] = forward * mole_frac[6] * mole_frac[107];
    rr_r[531] = reverse * mole_frac[7] * mole_frac[108];
  }
  // 532)  HO2 + IC8H18 <=> H2O2 + BC8H17
  {
    double forward = 63.2 * exp(3.37*vlntemp - 1.372e+04*ortc);
    double xik = -cgspl[6] + cgspl[7] - cgspl[107] + cgspl[109];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[532] = forward * mole_frac[6] * mole_frac[107];
    rr_r[532] = reverse * mole_frac[7] * mole_frac[109];
  }
  // 533)  HO2 + IC8H18 <=> H2O2 + CC8H17
  {
    double forward = 433.2 * exp(3.01*vlntemp - 1.209e+04*ortc);
    double xik = -cgspl[6] + cgspl[7] - cgspl[107] + cgspl[110];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[533] = forward * mole_frac[6] * mole_frac[107];
    rr_r[533] = reverse * mole_frac[7] * mole_frac[110];
  }
  // 534)  HO2 + IC8H18 <=> H2O2 + DC8H17
  {
    double forward = 40.8 * exp(3.59*vlntemp - 1.716e+04*ortc);
    double xik = -cgspl[6] + cgspl[7] - cgspl[107] + cgspl[111];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[534] = forward * mole_frac[6] * mole_frac[107];
    rr_r[534] = reverse * mole_frac[7] * mole_frac[111];
  }
  // 535)  O2 + IC8H18 <=> HO2 + AC8H17
  {
    double forward = 6.3e+13 * exp(-5.076e+04*ortc);
    double reverse = 2.296e+10 * exp(0.288*vlntemp + 1.592e+03*ortc);
    rr_f[535] = forward * mole_frac[3] * mole_frac[107];
    rr_r[535] = reverse * mole_frac[6] * mole_frac[108];
  }
  // 536)  O2 + IC8H18 <=> HO2 + BC8H17
  {
    double forward = 1.4e+13 * exp(-4.821e+04*ortc);
    double reverse = 8.889e+08 * exp(0.649*vlntemp + 1.649e+03*ortc);
    rr_f[536] = forward * mole_frac[3] * mole_frac[107];
    rr_r[536] = reverse * mole_frac[6] * mole_frac[109];
  }
  // 537)  O2 + IC8H18 <=> HO2 + CC8H17
  {
    double forward = 7.0e+12 * exp(-4.606e+04*ortc);
    double reverse = 1.279e+08 * exp(0.869*vlntemp + 499.0*ortc);
    rr_f[537] = forward * mole_frac[3] * mole_frac[107];
    rr_r[537] = reverse * mole_frac[6] * mole_frac[110];
  }
  // 538)  O2 + IC8H18 <=> HO2 + DC8H17
  {
    double forward = 4.2e+13 * exp(-5.076e+04*ortc);
    double reverse = 4.584e+10 * exp(0.288*vlntemp + 792.0*ortc);
    rr_f[538] = forward * mole_frac[3] * mole_frac[107];
    rr_r[538] = reverse * mole_frac[6] * mole_frac[111];
  }
  // 539)  C2H5 + IC8H18 <=> C2H6 + AC8H17
  {
    double forward = 1.5e+11 * exp(-1.34e+04*ortc);
    double reverse = 3.2e+11 * exp(-1.23e+04*ortc);
    rr_f[539] = forward * mole_frac[24] * mole_frac[107];
    rr_r[539] = reverse * mole_frac[23] * mole_frac[108];
  }
  // 540)  C2H5 + IC8H18 <=> C2H6 + BC8H17
  {
    double forward = 5.0e+10 * exp(-1.04e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.29e+04*ortc);
    rr_f[540] = forward * mole_frac[24] * mole_frac[107];
    rr_r[540] = reverse * mole_frac[23] * mole_frac[109];
  }
  // 541)  C2H5 + IC8H18 <=> C2H6 + CC8H17
  {
    double forward = 1.0e+11 * exp(-7.9e+03*ortc);
    double reverse = 3.0e+11 * exp(-2.1e+04*ortc);
    rr_f[541] = forward * mole_frac[24] * mole_frac[107];
    rr_r[541] = reverse * mole_frac[23] * mole_frac[110];
  }
  // 542)  C2H5 + IC8H18 <=> C2H6 + DC8H17
  {
    double forward = 1.0e+11 * exp(-1.34e+04*ortc);
    double reverse = 3.2e+11 * exp(-1.23e+04*ortc);
    rr_f[542] = forward * mole_frac[24] * mole_frac[107];
    rr_r[542] = reverse * mole_frac[23] * mole_frac[111];
  }
  // 543)  C2H3 + IC8H18 <=> C2H4 + AC8H17
  {
    double forward = 1.5e+12 * exp(-1.8e+04*ortc);
    double reverse = 2.57e+12 * exp(-2.54e+04*ortc);
    rr_f[543] = forward * mole_frac[26] * mole_frac[107];
    rr_r[543] = reverse * mole_frac[25] * mole_frac[108];
  }
  // 544)  C2H3 + IC8H18 <=> C2H4 + BC8H17
  {
    double forward = 4.0e+11 * exp(-1.68e+04*ortc);
    double reverse = 2.0e+12 * exp(-2.42e+04*ortc);
    rr_f[544] = forward * mole_frac[26] * mole_frac[107];
    rr_r[544] = reverse * mole_frac[25] * mole_frac[109];
  }
  // 545)  C2H3 + IC8H18 <=> C2H4 + CC8H17
  {
    double forward = 2.0e+11 * exp(-1.43e+04*ortc);
    double reverse = 2.5e+12 * exp(-2.3e+04*ortc);
    rr_f[545] = forward * mole_frac[26] * mole_frac[107];
    rr_r[545] = reverse * mole_frac[25] * mole_frac[110];
  }
  // 546)  C2H3 + IC8H18 <=> C2H4 + DC8H17
  {
    double forward = 1.0e+12 * exp(-1.8e+04*ortc);
    double reverse = 2.57e+12 * exp(-2.54e+04*ortc);
    rr_f[546] = forward * mole_frac[26] * mole_frac[107];
    rr_r[546] = reverse * mole_frac[25] * mole_frac[111];
  }
  // 547)  CH3 + XC7H14 <=> AC8H17
  {
    double forward = 1.3e+03 * exp(2.5*vlntemp - 8.52e+03*ortc);
    double reverse = 1.254e+13 * exp(0.3*vlntemp - 2.832e+04*ortc);
    rr_f[547] = forward * mole_frac[20] * mole_frac[93];
    rr_r[547] = reverse * mole_frac[108];
  }
  // 548)  IC4H9 + IC4H8 <=> AC8H17
  {
    double forward = 609.0 * exp(2.48*vlntemp - 8.52e+03*ortc);
    double reverse = 2.458e+14 * exp(-0.14*vlntemp - 2.678e+04*ortc);
    rr_f[548] = forward * mole_frac[56] * mole_frac[58];
    rr_r[548] = reverse * mole_frac[108];
  }
  // 549)  CH3 + YC7H14 <=> BC8H17
  {
    double forward = 1.3e+03 * exp(2.5*vlntemp - 8.52e+03*ortc);
    double reverse = 4.702e+12 * exp(0.48*vlntemp - 2.821e+04*ortc);
    rr_f[549] = forward * mole_frac[20] * mole_frac[94];
    rr_r[549] = reverse * mole_frac[109];
  }
  // 550)  TC4H9 + IC4H8 <=> CC8H17
  {
    double forward = 609.0 * exp(2.48*vlntemp - 6.13e+03*ortc);
    double reverse = 6.245e+14 * exp(-0.14*vlntemp - 2.589e+04*ortc);
    rr_f[550] = forward * mole_frac[57] * mole_frac[58];
    rr_r[550] = reverse * mole_frac[110];
  }
  // 551)  C3H6 + NEOC5H11 <=> DC8H17
  {
    double forward = 400.0 * exp(2.5*vlntemp - 8.52e+03*ortc);
    double reverse = 3.734e+08 * exp(1.57*vlntemp - 2.702e+04*ortc);
    rr_f[551] = forward * mole_frac[41] * mole_frac[102];
    rr_r[551] = reverse * mole_frac[111];
  }
  // 552)  BC8H17 <=> H + IC8H16
  {
    double forward = 1.843e+12 * exp(0.376*vlntemp - 3.524e+04*ortc);
    double reverse = 6.25e+11 * exp(0.5*vlntemp - 2.62e+03*ortc);
    rr_f[552] = forward * mole_frac[109];
    rr_r[552] = reverse * mole_frac[0] * mole_frac[112];
  }
  // 553)  CC8H17 <=> H + IC8H16
  {
    double forward = 8.995e+11 * exp(0.596*vlntemp - 3.715e+04*ortc);
    double reverse = 1.06e+12 * exp(0.5*vlntemp - 1.23e+03*ortc);
    rr_f[553] = forward * mole_frac[110];
    rr_r[553] = reverse * mole_frac[0] * mole_frac[112];
  }
  // 554)  CC8H17 <=> H + JC8H16
  {
    double forward = 4.213e+11 * exp(0.777*vlntemp - 3.669e+04*ortc);
    double reverse = 1.06e+12 * exp(0.5*vlntemp - 1.23e+03*ortc);
    rr_f[554] = forward * mole_frac[110];
    rr_r[554] = reverse * mole_frac[0] * mole_frac[113];
  }
  // 555)  O2 + CC8H17 <=> HO2 + IC8H16
  {
    double forward = 3.0e-19 * exp(-5.0e+03*ortc);
    double reverse = 2.0e-19 * exp(-1.75e+04*ortc);
    rr_f[555] = forward * mole_frac[3] * mole_frac[110];
    rr_r[555] = reverse * mole_frac[6] * mole_frac[112];
  }
  // 556)  O2 + CC8H17 <=> HO2 + JC8H16
  {
    double forward = 1.5e-19 * exp(-4.0e+03*ortc);
    double reverse = 2.0e-19 * exp(-1.75e+04*ortc);
    rr_f[556] = forward * mole_frac[3] * mole_frac[110];
    rr_r[556] = reverse * mole_frac[6] * mole_frac[113];
  }
  // 557)  DC8H17 <=> H + JC8H16
  {
    double forward = 1.484e+13 * exp(0.196*vlntemp - 3.309e+04*ortc);
    double reverse = 6.25e+11 * exp(0.5*vlntemp - 2.62e+03*ortc);
    rr_f[557] = forward * mole_frac[111];
    rr_r[557] = reverse * mole_frac[0] * mole_frac[113];
  }
  // 558)  AC8H17 <=> DC8H17
  {
    double forward = 1.39e+11 * exp(-1.54e+04*ortc);
    double reverse = 4.163e+11 * exp(-1.62e+04*ortc);
    rr_f[558] = forward * mole_frac[108];
    rr_r[558] = reverse * mole_frac[111];
  }
  // 559)  AC8H17 <=> CC8H17
  {
    double forward = 3.708e+11 * exp(-2.04e+04*ortc);
    double reverse = 1.859e+10 * exp(0.581*vlntemp - 2.619e+04*ortc);
    rr_f[559] = forward * mole_frac[108];
    rr_r[559] = reverse * mole_frac[110];
  }
  // 560)  O2 + DC8H17 <=> HO2 + JC8H16
  {
    double forward = 2.0e-18 * exp(-5.0e+03*ortc);
    double reverse = 2.0e-19 * exp(-1.75e+04*ortc);
    rr_f[560] = forward * mole_frac[3] * mole_frac[111];
    rr_r[560] = reverse * mole_frac[6] * mole_frac[113];
  }
  // 561)  OH + IC8H16 <=> CH3COCH3 + NEOC5H11
  {
    double forward = 1.0e+11 * exp(4.0e+03*ortc);
    double reverse = 0.0;
    rr_f[561] = forward * mole_frac[4] * mole_frac[112];
    rr_r[561] = reverse * mole_frac[35] * mole_frac[102];
  }
  // 562)  CH3O2 + IC8H18 <=> CH3O2H + AC8H17
  {
    double forward = 2.079 * exp(3.97*vlntemp - 1.828e+04*ortc);
    double xik = cgspl[17] - cgspl[18] - cgspl[107] + cgspl[108];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[562] = forward * mole_frac[18] * mole_frac[107];
    rr_r[562] = reverse * mole_frac[17] * mole_frac[108];
  }
  // 563)  CH3O2 + IC8H18 <=> CH3O2H + BC8H17
  {
    double forward = 10.165 * exp(3.58*vlntemp - 1.481e+04*ortc);
    double xik = cgspl[17] - cgspl[18] - cgspl[107] + cgspl[109];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[563] = forward * mole_frac[18] * mole_frac[107];
    rr_r[563] = reverse * mole_frac[17] * mole_frac[109];
  }
  // 564)  CH3O2 + IC8H18 <=> CH3O2H + CC8H17
  {
    double forward = 136.6 * exp(3.12*vlntemp - 1.319e+04*ortc);
    double xik = cgspl[17] - cgspl[18] - cgspl[107] + cgspl[110];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[564] = forward * mole_frac[18] * mole_frac[107];
    rr_r[564] = reverse * mole_frac[17] * mole_frac[110];
  }
  // 565)  CH3O2 + IC8H18 <=> CH3O2H + DC8H17
  {
    double forward = 1.386 * exp(3.97*vlntemp - 1.828e+04*ortc);
    double xik = cgspl[17] - cgspl[18] - cgspl[107] + cgspl[111];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[565] = forward * mole_frac[18] * mole_frac[107];
    rr_r[565] = reverse * mole_frac[17] * mole_frac[111];
  }
  // 566)  IC8H18 + CC8H17O2 <=> AC8H17 + CC8H17O2H
  {
    double forward = 1.814e+13 * exp(-2.043e+04*ortc);
    double reverse = 1.44e+10 * exp(-1.5e+04*ortc);
    rr_f[566] = forward * mole_frac[107] * mole_frac[116];
    rr_r[566] = reverse * mole_frac[108] * mole_frac[118];
  }
  // 567)  IC8H18 + CC8H17O2 <=> BC8H17 + CC8H17O2H
  {
    double forward = 4.032e+12 * exp(-1.77e+04*ortc);
    double reverse = 1.44e+10 * exp(-1.5e+04*ortc);
    rr_f[567] = forward * mole_frac[107] * mole_frac[116];
    rr_r[567] = reverse * mole_frac[109] * mole_frac[118];
  }
  // 568)  IC8H18 + CC8H17O2 <=> CC8H17 + CC8H17O2H
  {
    double forward = 2.0e+12 * exp(-1.6e+04*ortc);
    double reverse = 1.44e+10 * exp(-1.5e+04*ortc);
    rr_f[568] = forward * mole_frac[107] * mole_frac[116];
    rr_r[568] = reverse * mole_frac[110] * mole_frac[118];
  }
  // 569)  IC8H18 + CC8H17O2 <=> DC8H17 + CC8H17O2H
  {
    double forward = 1.21e+13 * exp(-2.043e+04*ortc);
    double reverse = 1.44e+10 * exp(-1.5e+04*ortc);
    rr_f[569] = forward * mole_frac[107] * mole_frac[116];
    rr_r[569] = reverse * mole_frac[111] * mole_frac[118];
  }
  // 570)  O2CHO + IC8H18 <=> HO2CHO + AC8H17
  {
    double forward = 2.52e+13 * exp(-2.044e+04*ortc);
    double reverse = 558.1 * exp(2.3*vlntemp - 3.062e+03*ortc);
    rr_f[570] = forward * mole_frac[13] * mole_frac[107];
    rr_r[570] = reverse * mole_frac[12] * mole_frac[108];
  }
  // 571)  O2CHO + IC8H18 <=> HO2CHO + BC8H17
  {
    double forward = 5.6e+12 * exp(-1.769e+04*ortc);
    double reverse = 21.6 * exp(2.66*vlntemp - 2.806e+03*ortc);
    rr_f[571] = forward * mole_frac[13] * mole_frac[107];
    rr_r[571] = reverse * mole_frac[12] * mole_frac[109];
  }
  // 572)  O2CHO + IC8H18 <=> HO2CHO + CC8H17
  {
    double forward = 2.8e+12 * exp(-1.601e+04*ortc);
    double reverse = 3.109 * exp(2.88*vlntemp - 4.633e+03*ortc);
    rr_f[572] = forward * mole_frac[13] * mole_frac[107];
    rr_r[572] = reverse * mole_frac[12] * mole_frac[110];
  }
  // 573)  O2CHO + IC8H18 <=> HO2CHO + DC8H17
  {
    double forward = 1.68e+13 * exp(-2.044e+04*ortc);
    double reverse = 1.114e+03 * exp(2.3*vlntemp - 3.862e+03*ortc);
    rr_f[573] = forward * mole_frac[13] * mole_frac[107];
    rr_r[573] = reverse * mole_frac[12] * mole_frac[111];
  }
  // 574)  IC4H6OH + IC8H18 <=> IC4H7OH + AC8H17
  {
    double forward = 705.0 * exp(3.3*vlntemp - 1.984e+04*ortc);
    double reverse = 0.2768 * exp(3.903*vlntemp - 6.526e+03*ortc);
    rr_f[574] = forward * mole_frac[71] * mole_frac[107];
    rr_r[574] = reverse * mole_frac[70] * mole_frac[108];
  }
  // 575)  IC4H6OH + IC8H18 <=> IC4H7OH + BC8H17
  {
    double forward = 156.8 * exp(3.3*vlntemp - 1.817e+04*ortc);
    double reverse = 0.01072 * exp(4.264*vlntemp - 7.35e+03*ortc);
    rr_f[575] = forward * mole_frac[71] * mole_frac[107];
    rr_r[575] = reverse * mole_frac[70] * mole_frac[109];
  }
  // 576)  IC4H6OH + IC8H18 <=> IC4H7OH + CC8H17
  {
    double forward = 84.40000000000001 * exp(3.3*vlntemp - 1.717e+04*ortc);
    double reverse = 1.661e-03 * exp(4.484*vlntemp - 9.648e+03*ortc);
    rr_f[576] = forward * mole_frac[71] * mole_frac[107];
    rr_r[576] = reverse * mole_frac[70] * mole_frac[110];
  }
  // 577)  IC4H6OH + IC8H18 <=> IC4H7OH + DC8H17
  {
    double forward = 470.0 * exp(3.3*vlntemp - 1.984e+04*ortc);
    double reverse = 0.5526 * exp(3.903*vlntemp - 7.326e+03*ortc);
    rr_f[577] = forward * mole_frac[71] * mole_frac[107];
    rr_r[577] = reverse * mole_frac[70] * mole_frac[111];
  }
  // 578)  AC8H17O2 <=> O2 + AC8H17
  {
    double forward = 3.465e+20 * exp(-1.653*vlntemp - 3.572e+04*ortc);
    double reverse = 4.52e+12;
    rr_f[578] = forward * mole_frac[114];
    rr_r[578] = reverse * mole_frac[3] * mole_frac[108];
  }
  // 579)  BC8H17O2 <=> O2 + BC8H17
  {
    double forward = 1.046e+23 * exp(-2.323*vlntemp - 3.884e+04*ortc);
    double reverse = 7.54e+12;
    rr_f[579] = forward * mole_frac[115];
    rr_r[579] = reverse * mole_frac[3] * mole_frac[109];
  }
  // 580)  CC8H17O2 <=> O2 + CC8H17
  {
    double forward = 3.62e+24 * exp(-2.56*vlntemp - 3.601e+04*ortc);
    double reverse = 1.41e+13;
    rr_f[580] = forward * mole_frac[116];
    rr_r[580] = reverse * mole_frac[3] * mole_frac[110];
  }
  // 581)  DC8H17O2 <=> O2 + DC8H17
  {
    double forward = 3.465e+20 * exp(-1.653*vlntemp - 3.492e+04*ortc);
    double reverse = 4.52e+12;
    rr_f[581] = forward * mole_frac[117];
    rr_r[581] = reverse * mole_frac[3] * mole_frac[111];
  }
  // 582)  CC8H17 + CC8H17O2 <=> 2 CC8H17O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 9.797e+16 * exp(-1.078*vlntemp - 3.127e+04*ortc);
    rr_f[582] = forward * mole_frac[110] * mole_frac[116];
    rr_r[582] = reverse * mole_frac[119] * mole_frac[119];
  }
  // 583)  HO2 + CC8H17 <=> OH + CC8H17O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 4.906e+18 * exp(-1.423*vlntemp - 2.698e+04*ortc);
    rr_f[583] = forward * mole_frac[6] * mole_frac[110];
    rr_r[583] = reverse * mole_frac[4] * mole_frac[119];
  }
  // 584)  CH3O2 + CC8H17 <=> CH3O + CC8H17O
  {
    double forward = 7.0e+12 * exp(1.0e+03*ortc);
    double reverse = 3.069e+17 * exp(-1.124*vlntemp - 3.122e+04*ortc);
    rr_f[584] = forward * mole_frac[18] * mole_frac[110];
    rr_r[584] = reverse * mole_frac[16] * mole_frac[119];
  }
  // 585)  HO2 + CC8H17O2 <=> O2 + CC8H17O2H
  {
    double forward = 1.75e+10 * exp(3.275e+03*ortc);
    double reverse = 3.752e+13 * exp(-0.792*vlntemp - 3.361e+04*ortc);
    rr_f[585] = forward * mole_frac[6] * mole_frac[116];
    rr_r[585] = reverse * mole_frac[3] * mole_frac[118];
  }
  // 586)  H2O2 + CC8H17O2 <=> HO2 + CC8H17O2H
  {
    double forward = 2.4e+12 * exp(-1.0e+04*ortc);
    double reverse = 2.4e+12 * exp(-1.0e+04*ortc);
    rr_f[586] = forward * mole_frac[7] * mole_frac[116];
    rr_r[586] = reverse * mole_frac[6] * mole_frac[118];
  }
  // 587)  CH3O2 + CC8H17O2 <=> O2 + CH3O + CC8H17O
  {
    double forward = 1.4e+16 * exp(-1.61*vlntemp - 1.86e+03*ortc);
    double reverse = 0.0;
    rr_f[587] = forward * mole_frac[18] * mole_frac[116];
    rr_r[587] = reverse * mole_frac[3] * mole_frac[16] * mole_frac[119];
  }
  // 588)  2 CC8H17O2 <=> O2 + 2 CC8H17O
  {
    double forward = 1.4e+16 * exp(-1.61*vlntemp - 1.86e+03*ortc);
    double reverse = 0.0;
    rr_f[588] = forward * mole_frac[116] * mole_frac[116];
    rr_r[588] = reverse * mole_frac[3] * mole_frac[119] * mole_frac[119];
  }
  // 589)  CC8H17O2H <=> OH + CC8H17O
  {
    double forward = 1.0e+16 * exp(-3.9e+04*ortc);
    double reverse = 1.273e+07 * exp(1.929*vlntemp + 5.922e+03*ortc);
    rr_f[589] = forward * mole_frac[118];
    rr_r[589] = reverse * mole_frac[4] * mole_frac[119];
  }
  // 590)  CC8H17O <=> CH3COCH3 + NEOC5H11
  {
    double forward = 1.206e+20 * exp(-1.671*vlntemp - 1.234e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.19e+04*ortc);
    rr_f[590] = forward * mole_frac[119];
    rr_r[590] = reverse * mole_frac[35] * mole_frac[102];
  }
  // 591)  AC8H17O2 <=> AC8H16OOH-A
  {
    double forward = 7.5e+10 * exp(-2.4e+04*ortc);
    double xik = -cgspl[114] + cgspl[120];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[591] = forward * mole_frac[114];
    rr_r[591] = reverse * mole_frac[120];
  }
  // 592)  AC8H17O2 <=> AC8H16OOH-B
  {
    double forward = 2.5e+10 * exp(-2.045e+04*ortc);
    double xik = -cgspl[114] + cgspl[121];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[592] = forward * mole_frac[114];
    rr_r[592] = reverse * mole_frac[121];
  }
  // 593)  AC8H17O2 <=> AC8H16OOH-C
  {
    double forward = 1.563e+09 * exp(-1.665e+04*ortc);
    double xik = -cgspl[114] + cgspl[122];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[593] = forward * mole_frac[114];
    rr_r[593] = reverse * mole_frac[122];
  }
  // 594)  BC8H17O2 <=> BC8H16OOH-A
  {
    double forward = 1.125e+11 * exp(-2.4e+04*ortc);
    double xik = -cgspl[115] + cgspl[123];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[594] = forward * mole_frac[115];
    rr_r[594] = reverse * mole_frac[123];
  }
  // 595)  BC8H17O2 <=> BC8H16OOH-D
  {
    double forward = 7.5e+10 * exp(-2.4e+04*ortc);
    double xik = -cgspl[115] + cgspl[124];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[595] = forward * mole_frac[115];
    rr_r[595] = reverse * mole_frac[124];
  }
  // 596)  CC8H17O2 <=> CC8H16OOH-A
  {
    double forward = 1.406e+10 * exp(-2.195e+04*ortc);
    double xik = -cgspl[116] + cgspl[125];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[596] = forward * mole_frac[116];
    rr_r[596] = reverse * mole_frac[125];
  }
  // 597)  DC8H17O2 <=> DC8H16OOH-B
  {
    double forward = 2.5e+10 * exp(-2.045e+04*ortc);
    double xik = -cgspl[117] + cgspl[127];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[597] = forward * mole_frac[117];
    rr_r[597] = reverse * mole_frac[127];
  }
  // 598)  DC8H17O2 <=> DC8H16OOH-C
  {
    double forward = 1.0e+11 * exp(-2.37e+04*ortc);
    double xik = -cgspl[117] + cgspl[126];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[598] = forward * mole_frac[117];
    rr_r[598] = reverse * mole_frac[126];
  }
  // 599)  BC8H17O2 <=> HO2 + IC8H16
  {
    double forward = 8.530000000000001e+35 * exp(-7.22*vlntemp - 
      4.149e+04*ortc); 
    double xik = cgspl[6] + cgspl[112] - cgspl[115];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[599] = forward * mole_frac[115];
    rr_r[599] = reverse * mole_frac[6] * mole_frac[112];
  }
  // 600)  CC8H17O2 <=> HO2 + IC8H16
  {
    double forward = 1.0044e+39 * exp(-8.109999999999999*vlntemp - 
      4.249e+04*ortc); 
    double xik = cgspl[6] + cgspl[112] - cgspl[116];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[600] = forward * mole_frac[116];
    rr_r[600] = reverse * mole_frac[6] * mole_frac[112];
  }
  // 601)  CC8H17O2 <=> HO2 + JC8H16
  {
    double forward = 2.015e+43 * exp(-9.41*vlntemp - 4.349e+04*ortc);
    double xik = cgspl[6] + cgspl[113] - cgspl[116];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[601] = forward * mole_frac[116];
    rr_r[601] = reverse * mole_frac[6] * mole_frac[113];
  }
  // 602)  DC8H17O2 <=> HO2 + JC8H16
  {
    double forward = 8.530000000000001e+35 * exp(-7.22*vlntemp - 
      4.149e+04*ortc); 
    double xik = cgspl[6] + cgspl[113] - cgspl[117];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[602] = forward * mole_frac[117];
    rr_r[602] = reverse * mole_frac[6] * mole_frac[113];
  }
  // 603)  AC8H16OOH-B <=> OH + IC8ETERAB
  {
    double forward = 3.0e+11 * exp(-1.425e+04*ortc);
    double reverse = 0.0;
    rr_f[603] = forward * mole_frac[121];
    rr_r[603] = reverse * mole_frac[4] * mole_frac[128];
  }
  // 604)  AC8H16OOH-C <=> OH + IC8ETERAC
  {
    double forward = 2.7375e+10 * exp(-7.0e+03*ortc);
    double reverse = 0.0;
    rr_f[604] = forward * mole_frac[122];
    rr_r[604] = reverse * mole_frac[4] * mole_frac[129];
  }
  // 605)  BC8H16OOH-A <=> OH + IC8ETERAB
  {
    double forward = 3.0e+11 * exp(-1.425e+04*ortc);
    double reverse = 0.0;
    rr_f[605] = forward * mole_frac[123];
    rr_r[605] = reverse * mole_frac[4] * mole_frac[128];
  }
  // 606)  BC8H16OOH-D <=> OH + IC8ETERBD
  {
    double forward = 3.0e+11 * exp(-1.425e+04*ortc);
    double reverse = 0.0;
    rr_f[606] = forward * mole_frac[124];
    rr_r[606] = reverse * mole_frac[4] * mole_frac[130];
  }
  // 607)  CC8H16OOH-A <=> OH + IC8ETERAC
  {
    double forward = 2.7375e+10 * exp(-7.0e+03*ortc);
    double reverse = 0.0;
    rr_f[607] = forward * mole_frac[125];
    rr_r[607] = reverse * mole_frac[4] * mole_frac[129];
  }
  // 608)  DC8H16OOH-B <=> OH + IC8ETERBD
  {
    double forward = 3.0e+11 * exp(-1.425e+04*ortc);
    double reverse = 0.0;
    rr_f[608] = forward * mole_frac[127];
    rr_r[608] = reverse * mole_frac[4] * mole_frac[130];
  }
  // 609)  DC8H16OOH-C <=> HO2 + JC8H16
  {
    double forward = 1.883e+18 * exp(-1.821*vlntemp - 1.496e+04*ortc);
    double reverse = 1.0e+11 * exp(-9.6e+03*ortc);
    rr_f[609] = forward * mole_frac[126];
    rr_r[609] = reverse * mole_frac[6] * mole_frac[113];
  }
  // 610)  AC8H16OOH-A <=> OH + CH2O + XC7H14
  {
    double forward = 9.087e+17 * exp(-1.26*vlntemp - 2.858e+04*ortc);
    double reverse = 0.0;
    rr_f[610] = forward * mole_frac[120];
    rr_r[610] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[93];
  }
  // 611)  AC8H16OOH-B <=> OH + CH2O + YC7H14
  {
    double forward = 1.252e+17 * exp(-1.08*vlntemp - 2.821e+04*ortc);
    double reverse = 0.0;
    rr_f[611] = forward * mole_frac[121];
    rr_r[611] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[94];
  }
  // 612)  BC8H16OOH-A <=> OH + IC4H8 + IC3H7CHO
  {
    double forward = 3.118e+21 * exp(-2.43*vlntemp - 2.633e+04*ortc);
    double reverse = 0.0;
    rr_f[612] = forward * mole_frac[123];
    rr_r[612] = reverse * mole_frac[4] * mole_frac[58] * mole_frac[66];
  }
  // 613)  BC8H16OOH-D <=> OH + C3H6 + TC4H9CHO
  {
    double forward = 1.293e+21 * exp(-2.2*vlntemp - 3.297e+04*ortc);
    double reverse = 0.0;
    rr_f[613] = forward * mole_frac[124];
    rr_r[613] = reverse * mole_frac[4] * mole_frac[41] * mole_frac[105];
  }
  // 614)  AC8H16OOH-A <=> TC4H9 + IC4H7OOH
  {
    double forward = 1.513e+24 * exp(-3.08*vlntemp - 2.684e+04*ortc);
    double reverse = 1.25e+11 * exp(-1.23e+04*ortc);
    rr_f[614] = forward * mole_frac[120];
    rr_r[614] = reverse * mole_frac[57] * mole_frac[76];
  }
  // 615)  AC8H16OOH-BO2 <=> O2 + AC8H16OOH-B
  {
    double forward = 1.361e+23 * exp(-2.357*vlntemp - 3.728e+04*ortc);
    double reverse = 7.54e+12;
    rr_f[615] = forward * mole_frac[131];
    rr_r[615] = reverse * mole_frac[3] * mole_frac[121];
  }
  // 616)  BC8H16OOH-AO2 <=> O2 + BC8H16OOH-A
  {
    double forward = 2.979e+20 * exp(-1.632*vlntemp - 3.49e+04*ortc);
    double reverse = 4.52e+12;
    rr_f[616] = forward * mole_frac[132];
    rr_r[616] = reverse * mole_frac[3] * mole_frac[123];
  }
  // 617)  BC8H16OOH-DO2 <=> O2 + BC8H16OOH-D
  {
    double forward = 2.98e+20 * exp(-1.632*vlntemp - 3.49e+04*ortc);
    double reverse = 4.52e+12;
    rr_f[617] = forward * mole_frac[133];
    rr_r[617] = reverse * mole_frac[3] * mole_frac[124];
  }
  // 618)  CC8H16OOH-AO2 <=> O2 + CC8H16OOH-A
  {
    double forward = 3.355e+20 * exp(-1.647*vlntemp - 3.572e+04*ortc);
    double reverse = 4.52e+12;
    rr_f[618] = forward * mole_frac[134];
    rr_r[618] = reverse * mole_frac[3] * mole_frac[125];
  }
  // 619)  DC8H16OOH-BO2 <=> O2 + DC8H16OOH-B
  {
    double forward = 1.362e+23 * exp(-2.357*vlntemp - 3.808e+04*ortc);
    double reverse = 7.54e+12;
    rr_f[619] = forward * mole_frac[135];
    rr_r[619] = reverse * mole_frac[3] * mole_frac[127];
  }
  // 620)  AC8H16OOH-BO2 <=> OH + IC8KETAB
  {
    double forward = 2.5e+10 * exp(-2.1e+04*ortc);
    double xik = cgspl[4] - cgspl[131] + cgspl[136];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[620] = forward * mole_frac[131];
    rr_r[620] = reverse * mole_frac[4] * mole_frac[136];
  }
  // 621)  BC8H16OOH-AO2 <=> OH + IC8KETBA
  {
    double forward = 1.25e+10 * exp(-1.745e+04*ortc);
    double xik = cgspl[4] - cgspl[132] + cgspl[137];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[621] = forward * mole_frac[132];
    rr_r[621] = reverse * mole_frac[4] * mole_frac[137];
  }
  // 622)  BC8H16OOH-DO2 <=> OH + IC8KETBD
  {
    double forward = 1.25e+10 * exp(-1.745e+04*ortc);
    double xik = cgspl[4] - cgspl[133] + cgspl[138];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[622] = forward * mole_frac[133];
    rr_r[622] = reverse * mole_frac[4] * mole_frac[138];
  }
  // 623)  DC8H16OOH-BO2 <=> OH + IC8KETDB
  {
    double forward = 2.5e+10 * exp(-2.1e+04*ortc);
    double xik = cgspl[4] - cgspl[135] + cgspl[139];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[623] = forward * mole_frac[135];
    rr_r[623] = reverse * mole_frac[4] * mole_frac[139];
  }
  // 624)  OH + IC8ETERAB <=> H2O + HCO + YC7H14
  {
    double forward = 1.25e+12;
    double reverse = 0.0;
    rr_f[624] = forward * mole_frac[4] * mole_frac[128];
    rr_r[624] = reverse * mole_frac[5] * mole_frac[11] * mole_frac[94];
  }
  // 625)  OH + IC8ETERAC <=> H2O + IC4H8 + TC3H6CHO
  {
    double forward = 1.25e+12;
    double reverse = 0.0;
    rr_f[625] = forward * mole_frac[4] * mole_frac[129];
    rr_r[625] = reverse * mole_frac[5] * mole_frac[58] * mole_frac[67];
  }
  // 626)  OH + IC8ETERAC <=> H2O + CH2O + YC7H13-Y2
  {
    double forward = 1.25e+12;
    double reverse = 0.0;
    rr_f[626] = forward * mole_frac[4] * mole_frac[129];
    rr_r[626] = reverse * mole_frac[5] * mole_frac[10] * mole_frac[96];
  }
  // 627)  OH + IC8ETERBD <=> H2O + C3H6 + TC4H9CO
  {
    double forward = 1.25e+12;
    double reverse = 0.0;
    rr_f[627] = forward * mole_frac[4] * mole_frac[130];
    rr_r[627] = reverse * mole_frac[5] * mole_frac[41] * mole_frac[106];
  }
  // 628)  HO2 + IC8ETERAB <=> H2O2 + HCO + YC7H14
  {
    double forward = 2.5e+12 * exp(-1.77e+04*ortc);
    double reverse = 0.0;
    rr_f[628] = forward * mole_frac[6] * mole_frac[128];
    rr_r[628] = reverse * mole_frac[7] * mole_frac[11] * mole_frac[94];
  }
  // 629)  HO2 + IC8ETERAC <=> H2O2 + IC4H8 + TC3H6CHO
  {
    double forward = 2.5e+12 * exp(-1.77e+04*ortc);
    double reverse = 0.0;
    rr_f[629] = forward * mole_frac[6] * mole_frac[129];
    rr_r[629] = reverse * mole_frac[7] * mole_frac[58] * mole_frac[67];
  }
  // 630)  HO2 + IC8ETERAC <=> H2O2 + CH2O + YC7H13-Y2
  {
    double forward = 2.5e+12 * exp(-1.77e+04*ortc);
    double reverse = 0.0;
    rr_f[630] = forward * mole_frac[6] * mole_frac[129];
    rr_r[630] = reverse * mole_frac[7] * mole_frac[10] * mole_frac[96];
  }
  // 631)  HO2 + IC8ETERBD <=> H2O2 + C3H6 + TC4H9CO
  {
    double forward = 2.5e+12 * exp(-1.77e+04*ortc);
    double reverse = 0.0;
    rr_f[631] = forward * mole_frac[6] * mole_frac[130];
    rr_r[631] = reverse * mole_frac[7] * mole_frac[41] * mole_frac[106];
  }
  // 632)  IC8KETAB <=> OH + IC3H7CHO + TC3H6CHO
  {
    double forward = 1.0e+16 * exp(-3.9e+04*ortc);
    double reverse = 0.0;
    rr_f[632] = forward * mole_frac[136];
    rr_r[632] = reverse * mole_frac[4] * mole_frac[66] * mole_frac[67];
  }
  // 633)  IC8KETBA <=> OH + CH2O + IC3H7COC3H6-T
  {
    double forward = 1.0e+16 * exp(-3.9e+04*ortc);
    double reverse = 0.0;
    rr_f[633] = forward * mole_frac[137];
    rr_r[633] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[140];
  }
  // 634)  IC8KETBD <=> OH + CH2O + TC4H9COC2H4S
  {
    double forward = 1.0e+16 * exp(-3.9e+04*ortc);
    double reverse = 0.0;
    rr_f[634] = forward * mole_frac[138];
    rr_r[634] = reverse * mole_frac[4] * mole_frac[10] * mole_frac[141];
  }
  // 635)  IC8KETDB <=> OH + CH3CHCHO + TC4H9CHO
  {
    double forward = 1.0e+16 * exp(-3.9e+04*ortc);
    double reverse = 0.0;
    rr_f[635] = forward * mole_frac[139];
    rr_r[635] = reverse * mole_frac[4] * mole_frac[87] * mole_frac[105];
  }
  // 636)  IC3H7COC3H6-T <=> IC3H7 + IC3H6CO
  {
    double forward = 1.217e+17 * exp(-0.63*vlntemp - 4.205e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.19e+04*ortc);
    rr_f[636] = forward * mole_frac[140];
    rr_r[636] = reverse * mole_frac[40] * mole_frac[75];
  }
  // 637)  TC4H9COC2H4S <=> CH3CHCO + TC4H9
  {
    double forward = 4.6e+20 * exp(-1.89*vlntemp - 3.825e+04*ortc);
    double reverse = 1.0e+11 * exp(-1.06e+04*ortc);
    rr_f[637] = forward * mole_frac[141];
    rr_r[637] = reverse * mole_frac[54] * mole_frac[57];
  }
  // 638)  H2O + IC4H6OH <=> OH + IC4H7OH
  {
    double forward = 5.875e+06 * exp(1.927*vlntemp - 3.024e+04*ortc);
    double reverse = 3.12e+06 * temperature * temperature * exp(298.0*ortc);
    rr_f[638] = forward * mole_frac[5] * mole_frac[71];
    rr_r[638] = reverse * mole_frac[4] * mole_frac[70];
  }
  // 639)  CH4 + IC4H6OH <=> CH3 + IC4H7OH
  {
    double forward = 357.0 * exp(3.087*vlntemp - 2.287e+04*ortc);
    double reverse = 2.21 * exp(3.5*vlntemp - 5.675e+03*ortc);
    rr_f[639] = forward * mole_frac[19] * mole_frac[71];
    rr_r[639] = reverse * mole_frac[20] * mole_frac[70];
  }
  // 640)  C3H6 + IC4H6OH <=> C3H5-A + IC4H7OH
  {
    double forward = 250.8 * exp(3.168*vlntemp - 1.864e+04*ortc);
    double reverse = 575.0 * exp(3.15*vlntemp - 1.866e+04*ortc);
    rr_f[640] = forward * mole_frac[41] * mole_frac[71];
    rr_r[640] = reverse * mole_frac[42] * mole_frac[70];
  }
  // 641)  CH3CHO + IC4H6OH <=> CH3CO + IC4H7OH
  {
    double forward = 1.357e+09 * exp(1.4*vlntemp - 1.794e+04*ortc);
    double reverse = 5.3e+06 * temperature * temperature * exp(-1.624e+04*ortc);
    rr_f[641] = forward * mole_frac[28] * mole_frac[71];
    rr_r[641] = reverse * mole_frac[29] * mole_frac[70];
  }
  // 642)  CH2O + C3H5-A <=> HCO + C3H6
  {
    double forward = 6.3e+08 * exp(1.9*vlntemp - 1.819e+04*ortc);
    double reverse = 9.165e+06 * exp(2.171*vlntemp - 1.77e+04*ortc);
    rr_f[642] = forward * mole_frac[10] * mole_frac[42];
    rr_r[642] = reverse * mole_frac[11] * mole_frac[41];
  }
  double wdot[143];
  double ropl[643];
  for (int i = 0; i < 643; i++)
  {
    ropl[i] = rr_f[i] - rr_r[i];
  }
  // 0. H
  wdot[0] = -ropl[0] + ropl[1] + ropl[2] + 2.0*ropl[4] + ropl[6] + ropl[7] - 
    ropl[8] - ropl[9] - ropl[10] - ropl[16] - ropl[17] + ropl[23] + ropl[25] - 
    ropl[27] + ropl[29] + ropl[33] + ropl[37] - ropl[40] - ropl[43] + ropl[47] - 
    ropl[50] - ropl[52] - ropl[55] - ropl[60] - ropl[61] + ropl[68] + ropl[69] + 
    ropl[73] - ropl[82] + ropl[87] + ropl[88] + 2.0*ropl[90] + ropl[91] - 
    ropl[93] + 2.0*ropl[95] + ropl[96] + 2.0*ropl[97] - ropl[99] - ropl[100] - 
    ropl[108] + ropl[109] - ropl[112] - ropl[113] + ropl[114] + ropl[119] - 
    ropl[128] + ropl[262] - ropl[136] - ropl[140] - ropl[141] - ropl[148] + 
    ropl[149] - ropl[151] - ropl[153] + ropl[159] - ropl[160] - ropl[164] + 
    ropl[168] + ropl[169] - ropl[173] - ropl[181] + ropl[191] - ropl[192] + 
    ropl[195] + ropl[198] + ropl[199] + ropl[200] + ropl[202] + 2.0*ropl[203] - 
    ropl[213] - ropl[214] - ropl[215] - ropl[216] + ropl[226] - ropl[229] + 
    ropl[239] - ropl[241] + ropl[244] + ropl[245] - ropl[249] + ropl[251] - 
    ropl[515] - ropl[259] + ropl[270] - ropl[272] + ropl[281] + ropl[282] - 
    ropl[283] + ropl[285] - ropl[287] + ropl[295] + ropl[302] + ropl[304] + 
    ropl[313] + ropl[331] + ropl[343] - ropl[344] - ropl[345] + 2.0*ropl[347] + 
    ropl[362] + ropl[368] + ropl[369] + ropl[373] - ropl[381] - ropl[387] + 
    ropl[391] + ropl[392] + ropl[393] + ropl[395] + ropl[396] + ropl[414] + 
    ropl[415] - ropl[427] - ropl[432] - ropl[433] - ropl[442] + ropl[455] + 
    ropl[456] + ropl[463] + ropl[464] - ropl[472] - ropl[477] - ropl[478] + 
    ropl[508] + ropl[509] + ropl[510] + ropl[511] - ropl[516] - ropl[517] - 
    ropl[518] + ropl[552] + ropl[553] + ropl[554] + ropl[557]; 
  // 1. H2
  wdot[1] = -ropl[1] - ropl[2] - ropl[4] + ropl[9] + ropl[17] + ropl[27] + 
    ropl[39] - ropl[41] + ropl[43] + ropl[50] + ropl[55] + ropl[61] + ropl[66] - 
    ropl[88] + ropl[100] - ropl[109] + ropl[113] + ropl[128] + ropl[136] + 
    ropl[141] + ropl[147] + ropl[152] + ropl[153] + ropl[164] + ropl[173] + 
    ropl[181] + ropl[213] + ropl[214] + ropl[215] + ropl[229] + ropl[241] + 
    ropl[249] + ropl[515] + ropl[259] + ropl[272] + ropl[283] - ropl[313] + 
    ropl[345] - ropl[369] + ropl[381] + ropl[387] - ropl[393] - ropl[396] + 
    ropl[427] + ropl[432] + ropl[433] + ropl[442] + ropl[472] + ropl[477] + 
    ropl[478] + ropl[516] + ropl[517] + ropl[518]; 
  // 2. O
  wdot[2] = ropl[0] - ropl[1] - ropl[3] + 2.0*ropl[5] + ropl[6] - ropl[11] - 
    ropl[18] - ropl[21] + ropl[22] - ropl[28] - ropl[29] - ropl[44] - ropl[59] - 
    ropl[63] - ropl[73] + ropl[74] - ropl[83] - ropl[90] + ropl[94] - ropl[97] - 
    ropl[101] - ropl[114] - ropl[257] - ropl[129] - ropl[137] - ropl[142] - 
    ropl[143] - ropl[149] - ropl[154] - ropl[167] - ropl[168] - ropl[174] - 
    ropl[182] - ropl[195] - ropl[196] - ropl[201] - ropl[202] - ropl[203] - 
    ropl[204] - ropl[205] - ropl[206] + ropl[247] - ropl[258] - ropl[268] - 
    ropl[269] - ropl[270] - ropl[271] - ropl[288] - ropl[346] - ropl[347] - 
    ropl[348] - ropl[357] - ropl[362] - ropl[379] - ropl[385] - ropl[424] - 
    ropl[425] - ropl[426] - ropl[458] - ropl[460] - ropl[470] - ropl[471] - 
    ropl[504] - ropl[519] - ropl[520] - ropl[521] - ropl[522]; 
  // 3. O2
  wdot[3] = -ropl[0] - ropl[5] - ropl[8] + ropl[9] + ropl[11] + ropl[12] - 
    ropl[13] - ropl[14] - ropl[22] - ropl[26] + ropl[32] + ropl[34] - ropl[48] - 
    ropl[53] - ropl[54] - ropl[131] + ropl[72] - ropl[74] - ropl[75] - ropl[76] 
    + ropl[80] + ropl[81] + ropl[83] - ropl[87] - ropl[94] - ropl[95] - ropl[96] 
    - ropl[103] - ropl[117] + ropl[120] - ropl[121] - ropl[122] - ropl[123] - 
    ropl[263] - ropl[264] - ropl[150] - ropl[157] - ropl[161] - ropl[162] - 
    ropl[166] - ropl[176] - ropl[184] - ropl[193] - ropl[217] - ropl[218] - 
    ropl[219] - ropl[235] - ropl[236] - ropl[237] - ropl[240] - ropl[246] - 
    ropl[247] - ropl[248] - ropl[253] - ropl[535] - ropl[536] - ropl[537] - 
    ropl[538] - ropl[277] - ropl[278] - ropl[285] + ropl[289] + ropl[292] - 
    ropl[297] - ropl[305] - ropl[306] + ropl[307] + ropl[308] + ropl[317] + 
    ropl[320] + ropl[321] + ropl[322] + ropl[323] + ropl[338] - ropl[352] - 
    ropl[359] - ropl[360] - ropl[361] + ropl[370] - ropl[376] - ropl[386] + 
    ropl[402] - ropl[403] - ropl[404] + ropl[405] - ropl[410] - ropl[416] - 
    ropl[417] - ropl[465] - ropl[466] + ropl[486] + ropl[498] - ropl[505] - 
    ropl[555] - ropl[556] - ropl[560] + ropl[578] + ropl[579] + ropl[580] + 
    ropl[581] + ropl[585] + ropl[587] + ropl[588] + ropl[615] + ropl[616] + 
    ropl[617] + ropl[618] + ropl[619]; 
  // 4. OH
  wdot[4] = ropl[0] + ropl[1] - ropl[2] + 2.0*ropl[3] - ropl[6] + ropl[7] + 
    2.0*ropl[10] + ropl[11] - ropl[12] + 2.0*ropl[15] + ropl[16] + ropl[18] - 
    ropl[19] - ropl[20] - ropl[23] + ropl[24] + ropl[28] - ropl[30] + ropl[33] + 
    ropl[36] - ropl[42] + ropl[44] - ropl[58] + ropl[59] - ropl[62] + ropl[63] - 
    ropl[66] - ropl[67] - ropl[68] - ropl[69] - ropl[70] + ropl[71] + ropl[75] + 
    ropl[82] + ropl[84] + ropl[87] - ropl[91] + ropl[96] + ropl[101] - ropl[102] 
    + ropl[115] + ropl[123] + ropl[124] + ropl[129] - ropl[130] + ropl[263] + 
    ropl[265] - ropl[266] + ropl[137] + ropl[143] - ropl[144] - ropl[145] - 
    ropl[147] + ropl[150] - ropl[155] - ropl[165] + ropl[166] - ropl[169] - 
    ropl[170] - ropl[172] + ropl[174] + ropl[182] - ropl[183] - ropl[194] + 
    ropl[204] + ropl[205] + ropl[206] - ropl[207] - ropl[208] - ropl[209] + 
    ropl[227] + ropl[236] + ropl[237] + ropl[254] - ropl[255] - ropl[256] - 
    ropl[267] + ropl[271] - ropl[276] + ropl[279] - ropl[284] - ropl[286] + 
    ropl[293] + ropl[294] + ropl[300] + ropl[323] + ropl[329] - ropl[334] + 
    ropl[339] + ropl[340] + ropl[341] + ropl[348] - ropl[356] + ropl[359] + 
    ropl[361] + ropl[365] + ropl[379] - ropl[380] - ropl[382] + ropl[385] + 
    ropl[389] + ropl[394] + ropl[397] - ropl[401] + ropl[404] + ropl[408] + 
    ropl[412] - ropl[421] - ropl[422] - ropl[423] - ropl[428] - ropl[434] - 
    ropl[435] - ropl[443] + ropl[447] + ropl[450] - ropl[457] - ropl[468] - 
    ropl[469] - ropl[473] - ropl[479] - ropl[480] + ropl[482] - ropl[489] - 
    ropl[490] + ropl[496] + ropl[500] + ropl[504] - ropl[506] + ropl[519] + 
    ropl[520] + ropl[521] + ropl[522] - ropl[523] - ropl[524] - ropl[525] - 
    ropl[526] - ropl[561] + ropl[583] + ropl[589] + ropl[603] + ropl[604] + 
    ropl[605] + ropl[606] + ropl[607] + ropl[608] + ropl[610] + ropl[611] + 
    ropl[612] + ropl[613] + ropl[620] + ropl[621] + ropl[622] + ropl[623] - 
    ropl[624] - ropl[625] - ropl[626] - ropl[627] + ropl[632] + ropl[633] + 
    ropl[634] + ropl[635] + ropl[638]; 
  // 5. H2O
  wdot[5] = ropl[2] - ropl[3] - ropl[7] + ropl[12] + ropl[16] + ropl[19] + 
    ropl[20] + ropl[30] + ropl[42] + ropl[58] + ropl[62] + ropl[67] + ropl[70] + 
    ropl[102] + ropl[130] + ropl[266] + ropl[144] + ropl[155] + ropl[165] + 
    ropl[172] + ropl[183] + ropl[194] + ropl[207] + ropl[208] + ropl[209] + 
    ropl[256] + ropl[276] + ropl[334] + ropl[356] + ropl[380] + ropl[382] + 
    ropl[428] + ropl[434] + ropl[435] + ropl[443] + ropl[473] + ropl[479] + 
    ropl[480] + ropl[489] + ropl[490] + ropl[506] + ropl[523] + ropl[524] + 
    ropl[525] + ropl[526] + ropl[624] + ropl[625] + ropl[626] + ropl[627] - 
    ropl[638]; 
  // 6. HO2
  wdot[6] = ropl[8] - ropl[9] - ropl[10] - ropl[11] - ropl[12] + 2.0*ropl[13] + 
    2.0*ropl[14] + ropl[17] + ropl[18] + ropl[19] + ropl[20] - ropl[24] + 
    ropl[26] - ropl[64] - ropl[32] - ropl[33] - ropl[46] + ropl[48] - ropl[51] + 
    ropl[53] + ropl[54] - ropl[56] + ropl[131] - ropl[71] - ropl[72] - ropl[80] 
    + ropl[103] - ropl[105] - ropl[115] + ropl[117] + ropl[121] + ropl[122] + 
    ropl[125] + ropl[264] - ropl[265] - ropl[133] + ropl[157] + ropl[161] + 
    ropl[176] - ropl[177] + ropl[184] - ropl[185] + ropl[193] - ropl[210] - 
    ropl[211] - ropl[212] + ropl[217] + ropl[218] + ropl[219] - ropl[227] + 
    ropl[235] + ropl[246] + ropl[253] - ropl[254] - ropl[534] + ropl[535] + 
    ropl[536] + ropl[537] + ropl[538] - ropl[279] - ropl[280] + ropl[291] + 
    ropl[297] + ropl[298] - ropl[300] + ropl[305] + ropl[306] - ropl[317] + 
    ropl[318] - ropl[323] - ropl[333] + ropl[336] + ropl[337] - ropl[350] + 
    ropl[352] - ropl[365] - ropl[370] - ropl[374] + ropl[376] - ropl[377] - 
    ropl[383] + ropl[386] - ropl[389] + ropl[403] - ropl[405] - ropl[408] + 
    ropl[416] + ropl[417] - ropl[430] - ropl[438] - ropl[439] - ropl[445] - 
    ropl[447] - ropl[450] + ropl[465] + ropl[466] - ropl[475] - ropl[482] + 
    ropl[487] + ropl[488] - ropl[491] - ropl[492] - ropl[496] - ropl[502] + 
    ropl[505] - ropl[531] - ropl[532] - ropl[533] + ropl[555] + ropl[556] + 
    ropl[560] - ropl[583] - ropl[585] + ropl[586] + ropl[599] + ropl[600] + 
    ropl[601] + ropl[602] + ropl[609] - ropl[628] - ropl[629] - ropl[630] - 
    ropl[631]; 
  // 7. H2O2
  wdot[7] = -ropl[13] - ropl[14] - ropl[15] - ropl[16] - ropl[17] - ropl[18] - 
    ropl[19] - ropl[20] + ropl[64] + ropl[46] + ropl[51] + ropl[56] + ropl[105] 
    + ropl[133] + ropl[177] + ropl[185] + ropl[210] + ropl[211] + ropl[212] + 
    ropl[534] + ropl[280] - ropl[318] + ropl[333] + ropl[350] + ropl[374] + 
    ropl[377] + ropl[383] + ropl[430] + ropl[438] + ropl[439] + ropl[445] + 
    ropl[475] + ropl[491] + ropl[492] + ropl[502] + ropl[531] + ropl[532] + 
    ropl[533] - ropl[586] + ropl[628] + ropl[629] + ropl[630] + ropl[631]; 
  // 8. CO
  wdot[8] = -ropl[21] - ropl[22] - ropl[23] - ropl[24] + ropl[25] + ropl[26] + 
    ropl[27] + ropl[28] + ropl[30] + ropl[31] - ropl[38] + 2.0*ropl[39] - 
    ropl[41] + ropl[87] + ropl[90] + ropl[92] + ropl[96] + ropl[97] + ropl[257] 
    + ropl[265] + ropl[135] - ropl[139] + ropl[140] + ropl[145] + ropl[146] + 
    2.0*ropl[147] + ropl[148] + 2.0*ropl[149] + 2.0*ropl[150] + ropl[167] + 
    ropl[170] + ropl[189] + ropl[190] + ropl[279] + ropl[285] + ropl[287] + 
    ropl[288] + ropl[388] + ropl[404] + ropl[410] + ropl[507]; 
  // 9. CO2
  wdot[9] = ropl[21] + ropl[22] + ropl[23] + ropl[24] + ropl[29] + ropl[33] + 
    ropl[37] - ropl[92] + ropl[95] + ropl[142] + ropl[286] + ropl[401]; 
  // 10. CH2O
  wdot[10] = ropl[32] - ropl[35] - ropl[38] + ropl[40] + ropl[41] - ropl[42] - 
    ropl[43] - ropl[44] - ropl[45] - ropl[46] + ropl[47] + ropl[48] + ropl[49] + 
    ropl[50] + ropl[51] - ropl[52] + ropl[53] + ropl[54] + ropl[55] + ropl[56] + 
    2.0*ropl[57] + ropl[58] + ropl[59] + ropl[66] + ropl[73] + ropl[75] - 
    ropl[77] + ropl[91] + ropl[92] + ropl[94] + ropl[118] + ropl[162] + 
    ropl[237] + ropl[248] + ropl[258] + ropl[294] + ropl[296] - ropl[319] + 
    ropl[340] + ropl[341] + ropl[360] + ropl[361] + ropl[366] - ropl[371] - 
    ropl[398] - ropl[399] + ropl[408] + ropl[410] + ropl[421] + ropl[423] + 
    ropl[470] + ropl[500] + ropl[610] + ropl[611] + ropl[626] + ropl[630] + 
    ropl[633] + ropl[634] - ropl[642]; 
  // 11. HCO
  wdot[11] = -ropl[25] - ropl[26] - ropl[27] - ropl[28] - ropl[29] - ropl[30] - 
    ropl[31] - ropl[32] - ropl[33] + ropl[34] + ropl[35] + 2.0*ropl[38] - 
    2.0*ropl[39] - ropl[40] + ropl[42] + ropl[43] + ropl[44] + ropl[45] + 
    ropl[46] - ropl[57] + ropl[77] + ropl[127] + ropl[154] + ropl[162] + 
    ropl[180] + ropl[201] + ropl[240] + ropl[268] + ropl[277] + ropl[278] + 
    ropl[284] + ropl[319] + ropl[332] + ropl[357] + ropl[371] + ropl[390] + 
    ropl[398] + ropl[399] + ropl[424] + ropl[426] + ropl[624] + ropl[628] + 
    ropl[642]; 
  // 12. HO2CHO
  wdot[12] = ropl[35] - ropl[36] + ropl[351] + ropl[570] + ropl[571] + ropl[572] 
    + ropl[573]; 
  // 13. O2CHO
  wdot[13] = -ropl[34] - ropl[35] - ropl[351] - ropl[570] - ropl[571] - 
    ropl[572] - ropl[573]; 
  // 14. OCHO
  wdot[14] = ropl[36] - ropl[37];
  // 15. CH2OH
  wdot[15] = ropl[52] - ropl[53] - ropl[54] - ropl[55] - ropl[56] - ropl[57] - 
    ropl[58] - ropl[59] + ropl[69] + ropl[145] + ropl[375] + ropl[410] + 
    ropl[411]; 
  // 16. CH3O
  wdot[16] = -ropl[47] - ropl[48] - ropl[49] - ropl[50] - ropl[51] + ropl[68] + 
    ropl[71] + ropl[74] + 2.0*ropl[79] + 2.0*ropl[81] + ropl[82] + ropl[83] + 
    ropl[84] + ropl[116] + ropl[228] + ropl[301] + ropl[320] + ropl[324] + 
    ropl[364] + ropl[448] + ropl[451] + ropl[483] + ropl[584] + ropl[587]; 
  // 17. CH3O2H
  wdot[17] = ropl[77] + ropl[78] + ropl[80] - ropl[84] + ropl[106] + ropl[109] + 
    ropl[134] + ropl[158] + ropl[178] + ropl[188] + ropl[224] + ropl[358] + 
    ropl[431] + ropl[440] + ropl[441] + ropl[446] + ropl[476] + ropl[493] + 
    ropl[494] + ropl[562] + ropl[563] + ropl[564] + ropl[565]; 
  // 18. CH3O2
  wdot[18] = ropl[76] - ropl[77] - ropl[78] - ropl[79] - ropl[80] - 2.0*ropl[81] 
    - ropl[82] - ropl[83] - ropl[106] - ropl[109] - ropl[116] - ropl[134] - 
    ropl[158] - ropl[178] - ropl[188] - ropl[224] - ropl[228] - ropl[301] - 
    ropl[320] - ropl[358] - ropl[364] - ropl[431] - ropl[440] - ropl[441] - 
    ropl[446] - ropl[448] - ropl[451] - ropl[476] - ropl[483] - ropl[493] - 
    ropl[494] - ropl[562] - ropl[563] - ropl[564] - ropl[565] - ropl[584] - 
    ropl[587]; 
  // 19. CH4
  wdot[19] = -ropl[64] + ropl[31] + ropl[45] + ropl[49] + ropl[60] - ropl[61] - 
    ropl[62] - ropl[63] + ropl[132] - ropl[65] + ropl[72] - ropl[78] - ropl[86] 
    + ropl[104] + ropl[111] + ropl[138] + ropl[156] + ropl[163] + ropl[175] + 
    ropl[186] + ropl[220] + ropl[221] + ropl[222] + ropl[230] + ropl[242] + 
    ropl[250] + ropl[260] + ropl[273] - ropl[312] + ropl[349] + ropl[378] + 
    ropl[384] + ropl[406] + ropl[429] + ropl[436] + ropl[437] + ropl[444] + 
    ropl[474] + ropl[503] + ropl[527] + ropl[528] + ropl[529] + ropl[530] - 
    ropl[639]; 
  // 20. CH3
  wdot[20] = ropl[64] - ropl[31] - ropl[45] - ropl[49] - ropl[60] + ropl[61] + 
    ropl[62] + ropl[63] - ropl[132] + 2.0*ropl[65] - ropl[66] - ropl[67] - 
    ropl[68] - ropl[69] - ropl[70] - ropl[71] - ropl[72] - ropl[73] - ropl[74] - 
    ropl[75] - ropl[76] + ropl[78] - ropl[79] + 2.0*ropl[86] + ropl[88] + 
    ropl[93] - 2.0*ropl[98] - ropl[104] + ropl[107] - ropl[111] + 2.0*ropl[112] 
    + ropl[118] + ropl[127] + ropl[135] - ropl[138] + ropl[140] + ropl[154] - 
    ropl[156] - ropl[159] - ropl[163] + ropl[170] + ropl[171] - ropl[175] + 
    ropl[179] - ropl[186] + ropl[192] + ropl[196] + ropl[197] + ropl[202] + 
    ropl[216] - ropl[220] - ropl[221] - ropl[222] + ropl[225] - ropl[230] + 
    ropl[238] - ropl[242] + ropl[243] - ropl[250] + ropl[255] - ropl[260] + 
    ropl[267] + ropl[269] - ropl[273] - ropl[281] - ropl[282] + ropl[299] + 
    ropl[303] + ropl[312] - ropl[324] + ropl[330] + ropl[342] + ropl[344] + 
    2.0*ropl[346] - ropl[349] + ropl[363] - ropl[378] - ropl[384] - ropl[406] + 
    ropl[409] + ropl[413] + ropl[418] + ropl[420] - ropl[429] - ropl[436] - 
    ropl[437] - ropl[444] + ropl[459] + ropl[461] - ropl[474] + ropl[481] + 
    ropl[497] + ropl[501] - ropl[503] + ropl[512] - ropl[527] - ropl[528] - 
    ropl[529] - ropl[530] - ropl[547] - ropl[549] + ropl[639]; 
  // 21. CH2
  wdot[21] = -ropl[65] + ropl[70] + ropl[85] + ropl[89] - ropl[93] - ropl[94] - 
    ropl[95] - ropl[96] - ropl[97] + ropl[263] - ropl[139] + ropl[142] + 
    ropl[167] + ropl[254] + ropl[270]; 
  // 22. CH2(S)
  wdot[22] = ropl[67] - ropl[85] - ropl[86] - ropl[87] - ropl[88] - ropl[89] - 
    ropl[90] - ropl[91] - ropl[92] - ropl[107] - ropl[146] + ropl[148] - 
    ropl[159]; 
  // 23. C2H6
  wdot[23] = ropl[98] + ropl[99] - ropl[100] - ropl[101] - ropl[102] - ropl[103] 
    - ropl[104] - ropl[105] - ropl[106] - ropl[107] + ropl[223] + ropl[231] + 
    ropl[539] + ropl[540] + ropl[541] + ropl[542] - ropl[314]; 
  // 24. C2H5
  wdot[24] = -ropl[99] + ropl[100] + ropl[101] + ropl[102] + ropl[103] + 
    ropl[104] + ropl[105] + ropl[106] + ropl[107] + ropl[108] - ropl[110] - 
    ropl[111] - ropl[112] - ropl[113] - ropl[114] - ropl[115] - ropl[116] + 
    ropl[120] - ropl[121] - ropl[122] - ropl[123] + ropl[190] + ropl[192] + 
    ropl[201] - ropl[223] - ropl[231] - ropl[232] - ropl[539] - ropl[540] - 
    ropl[541] - ropl[542] + ropl[286] + ropl[287] + ropl[314] - ropl[325] + 
    ropl[419]; 
  // 25. C2H4
  wdot[25] = -ropl[108] + 2.0*ropl[110] + ropl[111] + ropl[113] + ropl[121] + 
    ropl[122] + ropl[125] + ropl[257] + ropl[265] + ropl[146] + ropl[151] - 
    ropl[152] - ropl[153] - ropl[154] - ropl[155] - ropl[156] - ropl[157] - 
    ropl[158] + ropl[159] + ropl[187] + ropl[216] + ropl[232] + ropl[233] + 
    ropl[543] + ropl[274] + ropl[279] - ropl[311] + ropl[544] + ropl[545] + 
    ropl[546]; 
  // 26. C2H3
  wdot[26] = -ropl[110] - ropl[151] + ropl[153] + ropl[155] + ropl[156] + 
    ropl[157] + ropl[158] + ropl[160] - ropl[161] - ropl[162] - ropl[163] - 
    ropl[164] - ropl[165] + ropl[180] - ropl[187] + ropl[189] + ropl[197] - 
    ropl[233] - ropl[543] + ropl[268] - ropl[274] + ropl[296] + ropl[311] + 
    ropl[454] + ropl[496] - ropl[544] - ropl[545] - ropl[546]; 
  // 27. C2H2
  wdot[27] = ropl[152] - ropl[160] + ropl[161] + ropl[163] + ropl[164] + 
    ropl[165] - ropl[166] - ropl[167] - ropl[168] - ropl[169] - ropl[170] + 
    ropl[225] + ropl[237] + ropl[238] + ropl[243] + ropl[258] - ropl[281] - 
    ropl[282] + ropl[284] + ropl[411]; 
  // 28. CH3CHO
  wdot[28] = -ropl[131] - ropl[132] + ropl[114] + ropl[117] + ropl[119] + 
    ropl[123] + ropl[124] - ropl[127] - ropl[128] - ropl[129] - ropl[130] - 
    ropl[133] - ropl[134] + ropl[196] + ropl[240] + ropl[288] - ropl[315] + 
    ropl[422] + ropl[453] + ropl[457] + ropl[460] - ropl[641]; 
  // 29. CH3CO
  wdot[29] = ropl[131] + ropl[132] + ropl[126] + ropl[128] + ropl[129] + 
    ropl[130] + ropl[133] + ropl[134] - ropl[135] - ropl[136] - ropl[137] - 
    ropl[138] + ropl[171] + ropl[248] + ropl[294] + ropl[315] + ropl[425] + 
    ropl[458] + ropl[641]; 
  // 30. CH2CO
  wdot[30] = ropl[136] + ropl[137] + ropl[138] + ropl[139] - ropl[140] - 
    ropl[141] - ropl[142] - ropl[143] - ropl[144] - ropl[145] - ropl[146] + 
    ropl[169] + ropl[179] + ropl[202] + ropl[254] + ropl[255] + ropl[267] + 
    ropl[277] + ropl[346]; 
  // 31. HCCO
  wdot[31] = ropl[263] + ropl[141] + ropl[143] + ropl[144] - ropl[147] - 
    ropl[148] - ropl[149] - ropl[150] + ropl[166] + ropl[168] + ropl[269] + 
    ropl[270] + ropl[278] + ropl[285]; 
  // 32. C2H5O
  wdot[32] = ropl[115] + ropl[116] - ropl[117] - ropl[118] - ropl[119] + 
    ropl[321] + ropl[325] + ropl[449] + ropl[452] + ropl[484]; 
  // 33. C2H5O2
  wdot[33] = -ropl[120] - ropl[124] - ropl[125] - ropl[321] - ropl[449] - 
    ropl[452] - ropl[484]; 
  // 34. C2H3O1-2
  wdot[34] = -ropl[126];
  // 35. CH3COCH3
  wdot[35] = -ropl[171] - ropl[172] - ropl[173] - ropl[174] - ropl[175] - 
    ropl[176] - ropl[177] - ropl[178] + ropl[195] + ropl[330] + ropl[390] + 
    ropl[404] + ropl[454] + ropl[468] + ropl[469] + ropl[471] + ropl[485] + 
    ropl[561] + ropl[590]; 
  // 36. CH3COCH2
  wdot[36] = ropl[172] + ropl[173] + ropl[174] + ropl[175] + ropl[176] + 
    ropl[177] + ropl[178] - ropl[179] + ropl[247] + ropl[360]; 
  // 37. C2H3CHO
  wdot[37] = -ropl[180] - ropl[181] - ropl[182] - ropl[183] - ropl[184] - 
    ropl[185] - ropl[186] - ropl[187] - ropl[188] + ropl[236] + ropl[295] + 
    ropl[297] - ropl[316] + ropl[455]; 
  // 38. C2H3CO
  wdot[38] = ropl[181] + ropl[182] + ropl[183] + ropl[184] + ropl[185] + 
    ropl[186] + ropl[187] + ropl[188] - ropl[189] + ropl[316]; 
  // 39. C2H5CO
  wdot[39] = -ropl[190] + ropl[340];
  // 40. IC3H7
  wdot[40] = -ropl[191] - ropl[192] - ropl[193] - ropl[194] - ropl[195] - 
    ropl[196] + ropl[514] + ropl[289] + ropl[332] + ropl[357] + ropl[401] + 
    ropl[422] + ropl[425] + ropl[462] + ropl[467] + ropl[636]; 
  // 41. C3H6
  wdot[41] = ropl[191] + ropl[193] + ropl[194] - ropl[197] - ropl[198] - 
    ropl[199] - ropl[200] - ropl[201] - ropl[202] - ropl[203] - ropl[204] - 
    ropl[205] - ropl[206] - ropl[207] - ropl[208] - ropl[209] - ropl[210] - 
    ropl[211] - ropl[212] - ropl[213] - ropl[214] - ropl[215] - ropl[216] - 
    ropl[217] - ropl[218] - ropl[219] - ropl[220] - ropl[221] - ropl[222] - 
    ropl[223] - ropl[224] + ropl[232] - ropl[234] + ropl[261] + ropl[275] + 
    ropl[291] + ropl[298] + ropl[299] + ropl[303] - ropl[309] + ropl[341] + 
    ropl[344] + ropl[353] + ropl[354] + ropl[355] + ropl[460] - ropl[551] + 
    ropl[613] + ropl[627] + ropl[631] - ropl[640] + ropl[642]; 
  // 42. C3H5-A
  wdot[42] = ropl[198] + ropl[204] + ropl[207] + ropl[210] + ropl[213] + 
    ropl[217] + ropl[220] + ropl[223] + ropl[224] - ropl[225] - ropl[226] - 
    ropl[227] - ropl[228] - ropl[229] - ropl[230] - ropl[231] - ropl[232] - 
    ropl[233] + 2.0*ropl[234] - ropl[235] - ropl[236] - ropl[237] - ropl[261] - 
    ropl[275] + ropl[309] - ropl[327] - ropl[353] + ropl[640] - ropl[642]; 
  // 43. C3H5-S
  wdot[43] = ropl[199] + ropl[205] + ropl[208] + ropl[211] + ropl[214] + 
    ropl[218] + ropl[221] - ropl[238] - ropl[239] - ropl[240] - ropl[241] - 
    ropl[242] - ropl[354] + ropl[495]; 
  // 44. C3H5-T
  wdot[44] = ropl[200] + ropl[206] + ropl[209] + ropl[212] + ropl[215] + 
    ropl[219] + ropl[222] - ropl[243] - ropl[244] - ropl[245] - ropl[246] - 
    ropl[247] - ropl[248] - ropl[249] - ropl[250] + ropl[342] - ropl[355] + 
    ropl[366] + ropl[388] + ropl[419] + ropl[453]; 
  // 45. C3H4-P
  wdot[45] = -ropl[262] - ropl[263] - ropl[264] - ropl[265] - ropl[266] + 
    ropl[239] + ropl[245] + ropl[249] + ropl[250] + ropl[252] - ropl[267] - 
    ropl[268] - ropl[269] - ropl[270] - ropl[271] - ropl[272] - ropl[273] - 
    ropl[274] - ropl[275] + ropl[281] + ropl[461]; 
  // 46. C3H4-A
  wdot[46] = -ropl[257] + ropl[226] + ropl[229] + ropl[230] + ropl[231] + 
    ropl[233] - ropl[234] + ropl[235] + ropl[241] + ropl[242] + ropl[244] + 
    ropl[246] - ropl[251] - ropl[252] - ropl[253] - ropl[254] - ropl[255] - 
    ropl[256] - ropl[258] - ropl[259] - ropl[260] - ropl[261] - ropl[279] - 
    ropl[280] + ropl[282] + ropl[361] + ropl[363] + ropl[375] + ropl[412] + 
    ropl[495]; 
  // 47. C3H3
  wdot[47] = ropl[262] + ropl[264] + ropl[266] + ropl[251] + ropl[253] + 
    ropl[256] + ropl[259] + ropl[260] + ropl[261] + ropl[271] + ropl[272] + 
    ropl[273] + ropl[274] + ropl[275] - ropl[276] - ropl[277] + ropl[280] - 
    ropl[283]; 
  // 48. C3H2
  wdot[48] = ropl[276] - ropl[278] + ropl[283] - ropl[284] - ropl[285];
  // 49. C3H5O
  wdot[49] = ropl[227] + ropl[228] - ropl[295] - ropl[296] - ropl[297] + 
    ropl[327]; 
  // 50. C3H6OOH2-1
  wdot[50] = ropl[290] - ropl[291] + ropl[292];
  // 51. C3H6OOH2-1O2
  wdot[51] = -ropl[292] - ropl[293];
  // 52. IC3H7O2
  wdot[52] = -ropl[289] - ropl[290] - ropl[298];
  // 53. C3KET21
  wdot[53] = ropl[293] - ropl[294];
  // 54. CH3CHCO
  wdot[54] = ropl[203] - ropl[286] - ropl[287] - ropl[288] + ropl[456] + 
    ropl[637]; 
  // 55. SC4H9
  wdot[55] = -ropl[299] + ropl[421] + ropl[424];
  // 56. IC4H9
  wdot[56] = ropl[513] - ropl[302] - ropl[303] - ropl[306] + ropl[307] + 
    ropl[423] + ropl[426] + ropl[457] + ropl[458] + ropl[468] + ropl[469] - 
    ropl[548]; 
  // 57. TC4H9
  wdot[57] = ropl[513] - ropl[300] - ropl[301] - ropl[304] - ropl[305] + 
    ropl[308] - ropl[326] + ropl[507] - ropl[550] + ropl[614] + ropl[637]; 
  // 58. IC4H8
  wdot[58] = ropl[302] + ropl[304] + ropl[305] + ropl[306] - ropl[310] + 
    ropl[336] + ropl[337] - ropl[342] - ropl[343] - ropl[344] - ropl[345] - 
    ropl[346] - ropl[347] - ropl[348] - ropl[349] - ropl[350] - ropl[351] - 
    ropl[352] - ropl[353] - ropl[354] - ropl[355] - ropl[356] - ropl[357] - 
    ropl[358] - ropl[372] - ropl[400] - ropl[407] + ropl[413] + ropl[462] + 
    ropl[471] + ropl[497] + ropl[500] - ropl[548] - ropl[550] + ropl[612] + 
    ropl[625] + ropl[629]; 
  // 59. IC4H7
  wdot[59] = ropl[310] - ropl[328] + ropl[343] + ropl[345] + ropl[348] + 
    ropl[349] + ropl[350] + ropl[351] + ropl[352] + ropl[353] + ropl[354] + 
    ropl[355] + ropl[356] + ropl[358] - ropl[359] - ropl[360] - ropl[361] - 
    ropl[362] - ropl[363] - ropl[364] - ropl[365] + ropl[372] + ropl[397] + 
    ropl[400] + ropl[407] + ropl[418] + ropl[420] + ropl[467]; 
  // 60. TC4H9O2
  wdot[60] = -ropl[308] - ropl[309] - ropl[310] - ropl[311] - ropl[312] - 
    ropl[313] - ropl[314] - ropl[315] - ropl[316] - ropl[317] - ropl[318] - 
    ropl[319] - ropl[320] - ropl[321] - 2.0*ropl[322] - ropl[323] - ropl[324] - 
    ropl[325] - ropl[326] - ropl[327] - ropl[328] - ropl[337]; 
  // 61. IC4H9O2
  wdot[61] = -ropl[307] - ropl[335] - ropl[336];
  // 62. IC4H8O2H-I
  wdot[62] = ropl[335] + ropl[338] - ropl[341];
  // 63. TC4H9O
  wdot[63] = ropl[300] + ropl[301] + ropl[320] + ropl[321] + 2.0*ropl[322] + 
    ropl[323] + ropl[324] + ropl[325] + 2.0*ropl[326] + ropl[327] + ropl[328] + 
    ropl[329] - ropl[330]; 
  // 64. TC4H9O2H
  wdot[64] = ropl[309] + ropl[310] + ropl[311] + ropl[312] + ropl[313] + 
    ropl[314] + ropl[315] + ropl[316] + ropl[317] + ropl[318] + ropl[319] - 
    ropl[329]; 
  // 65. IC4H7O
  wdot[65] = ropl[328] + ropl[364] + ropl[365] - ropl[366] - ropl[367] - 
    ropl[368] - ropl[376] - ropl[377] - ropl[378] - ropl[379] - ropl[380] - 
    ropl[381] + ropl[394] + ropl[395] - ropl[396] - ropl[398] - ropl[407]; 
  // 66. IC3H7CHO
  wdot[66] = -ropl[331] - ropl[332] - ropl[333] - ropl[334] + ropl[393] + 
    ropl[399] + ropl[400] + ropl[405] + ropl[612] + ropl[632]; 
  // 67. TC3H6CHO
  wdot[67] = ropl[331] + ropl[333] + ropl[334] - ropl[389] - ropl[391] - 
    ropl[392] - ropl[393] - ropl[399] - ropl[400] + ropl[402] - ropl[403] - 
    ropl[404] - ropl[405] - ropl[406] + ropl[625] + ropl[629] + ropl[632]; 
  // 68. IC4H8OOH-IO2
  wdot[68] = -ropl[338] - ropl[339];
  // 69. IC4KETII
  wdot[69] = ropl[339] - ropl[340];
  // 70. IC4H7OH
  wdot[70] = ropl[369] + ropl[370] + ropl[371] + ropl[372] - ropl[373] - 
    ropl[374] - ropl[395] + ropl[396] - ropl[397] + ropl[398] + ropl[407] - 
    ropl[409] + ropl[574] + ropl[575] + ropl[576] + ropl[577] + ropl[638] + 
    ropl[639] + ropl[640] + ropl[641]; 
  // 71. IC4H6OH
  wdot[71] = ropl[367] - ropl[369] - ropl[370] - ropl[371] - ropl[372] + 
    ropl[373] + ropl[374] - ropl[375] - ropl[408] - ropl[574] - ropl[575] - 
    ropl[576] - ropl[577] - ropl[638] - ropl[639] - ropl[640] - ropl[641]; 
  // 72. IC3H5CHO
  wdot[72] = ropl[359] + ropl[362] + ropl[368] + ropl[376] + ropl[377] + 
    ropl[378] + ropl[379] + ropl[380] + ropl[381] - ropl[382] - ropl[383] - 
    ropl[384] - ropl[385] - ropl[386] - ropl[387] + ropl[391] + ropl[403] + 
    ropl[406] + ropl[496]; 
  // 73. IC3H5CO
  wdot[73] = ropl[382] + ropl[383] + ropl[384] + ropl[385] + ropl[386] + 
    ropl[387] - ropl[388]; 
  // 74. TC3H6OCHO
  wdot[74] = ropl[389] - ropl[390];
  // 75. IC3H6CO
  wdot[75] = ropl[347] + ropl[392] - ropl[401] + ropl[636];
  // 76. IC4H7OOH
  wdot[76] = -ropl[394] + ropl[501] + ropl[614];
  // 77. TC3H6O2CHO
  wdot[77] = -ropl[402];
  // 78. CH2CCH2OH
  wdot[78] = ropl[408] + ropl[409] - ropl[410] - ropl[411] - ropl[412];
  // 79. BC5H11
  wdot[79] = -ropl[413] - ropl[414] - ropl[415] - ropl[416] - ropl[417];
  // 80. AC5H10
  wdot[80] = ropl[414] + ropl[416] - ropl[418] - ropl[419] - ropl[421] - 
    ropl[424] - ropl[425] - ropl[426] - ropl[427] - ropl[428] - ropl[429] - 
    ropl[430] - ropl[431]; 
  // 81. BC5H10
  wdot[81] = ropl[415] + ropl[417] - ropl[420] - ropl[422] - ropl[432] - 
    ropl[433] - ropl[434] - ropl[435] - ropl[436] - ropl[437] - ropl[438] - 
    ropl[439] - ropl[440] - ropl[441]; 
  // 82. CC5H10
  wdot[82] = -ropl[423] - ropl[442] - ropl[443] - ropl[444] - ropl[445] - 
    ropl[446]; 
  // 83. AC5H9-C
  wdot[83] = ropl[427] + ropl[428] + ropl[429] + ropl[430] + ropl[431] + 
    ropl[432] + ropl[434] + ropl[436] + ropl[438] + ropl[440] - ropl[447] - 
    ropl[448] - ropl[449]; 
  // 84. CC5H9-B
  wdot[84] = ropl[433] + ropl[435] + ropl[437] + ropl[439] + ropl[441] + 
    ropl[442] + ropl[443] + ropl[444] + ropl[445] + ropl[446] - ropl[450] - 
    ropl[451] - ropl[452] + ropl[459]; 
  // 85. AC5H9O-C
  wdot[85] = ropl[447] + ropl[448] + ropl[449] - ropl[453];
  // 86. CC5H9O-B
  wdot[86] = ropl[450] + ropl[451] + ropl[452] - ropl[454];
  // 87. CH3CHCHO
  wdot[87] = -ropl[455] - ropl[456] + ropl[635];
  // 88. BC6H12
  wdot[88] = -ropl[459];
  // 89. CC6H12
  wdot[89] = -ropl[457] - ropl[458] + ropl[470];
  // 90. C5H10-2
  wdot[90] = -ropl[460];
  // 91. IC4H7-I1
  wdot[91] = -ropl[461] + ropl[485];
  // 92. YC7H15
  wdot[92] = -ropl[462] - ropl[463] - ropl[464] - ropl[465] - ropl[466] + 
    ropl[486] + ropl[512]; 
  // 93. XC7H14
  wdot[93] = ropl[463] + ropl[465] - ropl[467] - ropl[468] - ropl[470] - 
    ropl[472] - ropl[473] - ropl[474] - ropl[475] - ropl[476] + ropl[487] - 
    ropl[547] + ropl[610]; 
  // 94. YC7H14
  wdot[94] = ropl[464] + ropl[466] - ropl[469] - ropl[471] - ropl[477] - 
    ropl[478] - ropl[479] - ropl[480] + ropl[488] - ropl[549] + ropl[611] + 
    ropl[624] + ropl[628]; 
  // 95. XC7H13-Z
  wdot[95] = ropl[472] + ropl[473] + ropl[474] + ropl[475] + ropl[476] + 
    ropl[477] + ropl[479] - ropl[481]; 
  // 96. YC7H13-Y2
  wdot[96] = ropl[478] + ropl[480] - ropl[482] - ropl[483] - ropl[484] + 
    ropl[626] + ropl[630]; 
  // 97. YC7H13O-Y2
  wdot[97] = ropl[482] + ropl[483] + ropl[484] - ropl[485];
  // 98. YC7H15O2
  wdot[98] = -ropl[486] - ropl[487] - ropl[488];
  // 99. ACC6H10
  wdot[99] = ropl[481] - ropl[489] - ropl[490] - ropl[491] - ropl[492] - 
    ropl[493] - ropl[494]; 
  // 100. ACC6H9-A
  wdot[100] = ropl[489] + ropl[491] + ropl[493] - ropl[495];
  // 101. ACC6H9-D
  wdot[101] = ropl[490] + ropl[492] + ropl[494] - ropl[496];
  // 102. NEOC5H11
  wdot[102] = ropl[514] - ropl[497] + ropl[498] - ropl[551] + ropl[561] + 
    ropl[590]; 
  // 103. NEOC5H11O2
  wdot[103] = -ropl[498] - ropl[499];
  // 104. NEOC5H10OOH
  wdot[104] = ropl[499] - ropl[500] - ropl[501];
  // 105. TC4H9CHO
  wdot[105] = -ropl[502] - ropl[503] - ropl[504] - ropl[505] - ropl[506] + 
    ropl[613] + ropl[635]; 
  // 106. TC4H9CO
  wdot[106] = ropl[502] + ropl[503] + ropl[504] + ropl[505] + ropl[506] - 
    ropl[507] + ropl[627] + ropl[631]; 
  // 107. IC8H18
  wdot[107] = -ropl[513] - ropl[514] - ropl[515] - ropl[534] - ropl[535] - 
    ropl[536] - ropl[537] - ropl[538] - ropl[539] - ropl[540] - ropl[541] - 
    ropl[542] - ropl[543] - ropl[508] - ropl[509] - ropl[510] - ropl[511] - 
    ropl[512] - ropl[516] - ropl[517] - ropl[518] - ropl[519] - ropl[520] - 
    ropl[521] - ropl[522] - ropl[523] - ropl[524] - ropl[525] - ropl[526] - 
    ropl[527] - ropl[528] - ropl[529] - ropl[530] - ropl[531] - ropl[532] - 
    ropl[533] - ropl[544] - ropl[545] - ropl[546] - ropl[562] - ropl[563] - 
    ropl[564] - ropl[565] - ropl[566] - ropl[567] - ropl[568] - ropl[569] - 
    ropl[570] - ropl[571] - ropl[572] - ropl[573] - ropl[574] - ropl[575] - 
    ropl[576] - ropl[577]; 
  // 108. AC8H17
  wdot[108] = ropl[515] + ropl[535] + ropl[539] + ropl[543] + ropl[508] + 
    ropl[519] + ropl[523] + ropl[527] + ropl[531] + ropl[547] + ropl[548] - 
    ropl[558] - ropl[559] + ropl[562] + ropl[566] + ropl[570] + ropl[574] + 
    ropl[578]; 
  // 109. BC8H17
  wdot[109] = ropl[536] + ropl[540] + ropl[509] + ropl[516] + ropl[520] + 
    ropl[524] + ropl[528] + ropl[532] + ropl[544] + ropl[549] - ropl[552] + 
    ropl[563] + ropl[567] + ropl[571] + ropl[575] + ropl[579]; 
  // 110. CC8H17
  wdot[110] = ropl[537] + ropl[541] + ropl[510] + ropl[517] + ropl[521] + 
    ropl[525] + ropl[529] + ropl[533] + ropl[545] + ropl[550] - ropl[553] - 
    ropl[554] - ropl[555] - ropl[556] + ropl[559] + ropl[564] + ropl[568] + 
    ropl[572] + ropl[576] + ropl[580] - ropl[582] - ropl[583] - ropl[584]; 
  // 111. DC8H17
  wdot[111] = ropl[534] + ropl[538] + ropl[542] + ropl[511] + ropl[518] + 
    ropl[522] + ropl[526] + ropl[530] + ropl[546] + ropl[551] - ropl[557] + 
    ropl[558] - ropl[560] + ropl[565] + ropl[569] + ropl[573] + ropl[577] + 
    ropl[581]; 
  // 112. IC8H16
  wdot[112] = ropl[552] + ropl[553] + ropl[555] - ropl[561] + ropl[599] + 
    ropl[600]; 
  // 113. JC8H16
  wdot[113] = ropl[554] + ropl[556] + ropl[557] + ropl[560] + ropl[601] + 
    ropl[602] + ropl[609]; 
  // 114. AC8H17O2
  wdot[114] = -ropl[578] - ropl[591] - ropl[592] - ropl[593];
  // 115. BC8H17O2
  wdot[115] = -ropl[579] - ropl[594] - ropl[595] - ropl[599];
  // 116. CC8H17O2
  wdot[116] = -ropl[566] - ropl[567] - ropl[568] - ropl[569] - ropl[580] - 
    ropl[582] - ropl[585] - ropl[586] - ropl[587] - 2.0*ropl[588] - ropl[596] - 
    ropl[600] - ropl[601]; 
  // 117. DC8H17O2
  wdot[117] = -ropl[581] - ropl[597] - ropl[598] - ropl[602];
  // 118. CC8H17O2H
  wdot[118] = ropl[566] + ropl[567] + ropl[568] + ropl[569] + ropl[585] + 
    ropl[586] - ropl[589]; 
  // 119. CC8H17O
  wdot[119] = 2.0*ropl[582] + ropl[583] + ropl[584] + ropl[587] + 2.0*ropl[588] 
    + ropl[589] - ropl[590]; 
  // 120. AC8H16OOH-A
  wdot[120] = ropl[591] - ropl[610] - ropl[614];
  // 121. AC8H16OOH-B
  wdot[121] = ropl[592] - ropl[603] - ropl[611] + ropl[615];
  // 122. AC8H16OOH-C
  wdot[122] = ropl[593] - ropl[604];
  // 123. BC8H16OOH-A
  wdot[123] = ropl[594] - ropl[605] - ropl[612] + ropl[616];
  // 124. BC8H16OOH-D
  wdot[124] = ropl[595] - ropl[606] - ropl[613] + ropl[617];
  // 125. CC8H16OOH-A
  wdot[125] = ropl[596] - ropl[607] + ropl[618];
  // 126. DC8H16OOH-C
  wdot[126] = ropl[598] - ropl[609];
  // 127. DC8H16OOH-B
  wdot[127] = ropl[597] - ropl[608] + ropl[619];
  // 128. IC8ETERAB
  wdot[128] = ropl[603] + ropl[605] - ropl[624] - ropl[628];
  // 129. IC8ETERAC
  wdot[129] = ropl[604] + ropl[607] - ropl[625] - ropl[626] - ropl[629] - 
    ropl[630]; 
  // 130. IC8ETERBD
  wdot[130] = ropl[606] + ropl[608] - ropl[627] - ropl[631];
  // 131. AC8H16OOH-BO2
  wdot[131] = -ropl[615] - ropl[620];
  // 132. BC8H16OOH-AO2
  wdot[132] = -ropl[616] - ropl[621];
  // 133. BC8H16OOH-DO2
  wdot[133] = -ropl[617] - ropl[622];
  // 134. CC8H16OOH-AO2
  wdot[134] = -ropl[618];
  // 135. DC8H16OOH-BO2
  wdot[135] = -ropl[619] - ropl[623];
  // 136. IC8KETAB
  wdot[136] = ropl[620] - ropl[632];
  // 137. IC8KETBA
  wdot[137] = ropl[621] - ropl[633];
  // 138. IC8KETBD
  wdot[138] = ropl[622] - ropl[634];
  // 139. IC8KETDB
  wdot[139] = ropl[623] - ropl[635];
  // 140. IC3H7COC3H6-T
  wdot[140] = ropl[633] - ropl[636];
  // 141. TC4H9COC2H4S
  wdot[141] = ropl[634] - ropl[637];
  // 142. N2
  wdot[142] = 0.0;
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+0*spec_stride) , 
    "d"(wdot[0]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+1*spec_stride) , 
    "d"(wdot[1]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+2*spec_stride) , 
    "d"(wdot[2]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+3*spec_stride) , 
    "d"(wdot[3]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+4*spec_stride) , 
    "d"(wdot[4]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+5*spec_stride) , 
    "d"(wdot[5]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+6*spec_stride) , 
    "d"(wdot[6]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+7*spec_stride) , 
    "d"(wdot[7]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+8*spec_stride) , 
    "d"(wdot[8]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+9*spec_stride) , 
    "d"(wdot[9]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+10*spec_stride) , 
    "d"(wdot[10]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+11*spec_stride) , 
    "d"(wdot[11]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+12*spec_stride) , 
    "d"(wdot[12]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+13*spec_stride) , 
    "d"(wdot[13]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+14*spec_stride) , 
    "d"(wdot[14]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+15*spec_stride) , 
    "d"(wdot[15]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+16*spec_stride) , 
    "d"(wdot[16]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+17*spec_stride) , 
    "d"(wdot[17]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+18*spec_stride) , 
    "d"(wdot[18]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+19*spec_stride) , 
    "d"(wdot[19]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+20*spec_stride) , 
    "d"(wdot[20]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+21*spec_stride) , 
    "d"(wdot[21]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+22*spec_stride) , 
    "d"(wdot[22]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+23*spec_stride) , 
    "d"(wdot[23]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+24*spec_stride) , 
    "d"(wdot[24]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+25*spec_stride) , 
    "d"(wdot[25]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+26*spec_stride) , 
    "d"(wdot[26]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+27*spec_stride) , 
    "d"(wdot[27]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+28*spec_stride) , 
    "d"(wdot[28]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+29*spec_stride) , 
    "d"(wdot[29]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+30*spec_stride) , 
    "d"(wdot[30]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+31*spec_stride) , 
    "d"(wdot[31]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+32*spec_stride) , 
    "d"(wdot[32]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+33*spec_stride) , 
    "d"(wdot[33]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+34*spec_stride) , 
    "d"(wdot[34]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+35*spec_stride) , 
    "d"(wdot[35]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+36*spec_stride) , 
    "d"(wdot[36]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+37*spec_stride) , 
    "d"(wdot[37]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+38*spec_stride) , 
    "d"(wdot[38]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+39*spec_stride) , 
    "d"(wdot[39]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+40*spec_stride) , 
    "d"(wdot[40]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+41*spec_stride) , 
    "d"(wdot[41]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+42*spec_stride) , 
    "d"(wdot[42]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+43*spec_stride) , 
    "d"(wdot[43]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+44*spec_stride) , 
    "d"(wdot[44]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+45*spec_stride) , 
    "d"(wdot[45]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+46*spec_stride) , 
    "d"(wdot[46]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+47*spec_stride) , 
    "d"(wdot[47]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+48*spec_stride) , 
    "d"(wdot[48]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+49*spec_stride) , 
    "d"(wdot[49]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+50*spec_stride) , 
    "d"(wdot[50]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+51*spec_stride) , 
    "d"(wdot[51]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+52*spec_stride) , 
    "d"(wdot[52]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+53*spec_stride) , 
    "d"(wdot[53]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+54*spec_stride) , 
    "d"(wdot[54]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+55*spec_stride) , 
    "d"(wdot[55]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+56*spec_stride) , 
    "d"(wdot[56]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+57*spec_stride) , 
    "d"(wdot[57]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+58*spec_stride) , 
    "d"(wdot[58]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+59*spec_stride) , 
    "d"(wdot[59]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+60*spec_stride) , 
    "d"(wdot[60]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+61*spec_stride) , 
    "d"(wdot[61]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+62*spec_stride) , 
    "d"(wdot[62]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+63*spec_stride) , 
    "d"(wdot[63]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+64*spec_stride) , 
    "d"(wdot[64]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+65*spec_stride) , 
    "d"(wdot[65]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+66*spec_stride) , 
    "d"(wdot[66]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+67*spec_stride) , 
    "d"(wdot[67]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+68*spec_stride) , 
    "d"(wdot[68]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+69*spec_stride) , 
    "d"(wdot[69]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+70*spec_stride) , 
    "d"(wdot[70]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+71*spec_stride) , 
    "d"(wdot[71]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+72*spec_stride) , 
    "d"(wdot[72]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+73*spec_stride) , 
    "d"(wdot[73]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+74*spec_stride) , 
    "d"(wdot[74]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+75*spec_stride) , 
    "d"(wdot[75]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+76*spec_stride) , 
    "d"(wdot[76]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+77*spec_stride) , 
    "d"(wdot[77]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+78*spec_stride) , 
    "d"(wdot[78]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+79*spec_stride) , 
    "d"(wdot[79]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+80*spec_stride) , 
    "d"(wdot[80]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+81*spec_stride) , 
    "d"(wdot[81]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+82*spec_stride) , 
    "d"(wdot[82]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+83*spec_stride) , 
    "d"(wdot[83]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+84*spec_stride) , 
    "d"(wdot[84]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+85*spec_stride) , 
    "d"(wdot[85]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+86*spec_stride) , 
    "d"(wdot[86]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+87*spec_stride) , 
    "d"(wdot[87]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+88*spec_stride) , 
    "d"(wdot[88]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+89*spec_stride) , 
    "d"(wdot[89]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+90*spec_stride) , 
    "d"(wdot[90]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+91*spec_stride) , 
    "d"(wdot[91]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+92*spec_stride) , 
    "d"(wdot[92]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+93*spec_stride) , 
    "d"(wdot[93]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+94*spec_stride) , 
    "d"(wdot[94]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+95*spec_stride) , 
    "d"(wdot[95]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+96*spec_stride) , 
    "d"(wdot[96]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+97*spec_stride) , 
    "d"(wdot[97]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+98*spec_stride) , 
    "d"(wdot[98]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+99*spec_stride) , 
    "d"(wdot[99]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+100*spec_stride) 
    , "d"(wdot[100]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+101*spec_stride) 
    , "d"(wdot[101]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+102*spec_stride) 
    , "d"(wdot[102]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+103*spec_stride) 
    , "d"(wdot[103]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+104*spec_stride) 
    , "d"(wdot[104]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+105*spec_stride) 
    , "d"(wdot[105]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+106*spec_stride) 
    , "d"(wdot[106]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+107*spec_stride) 
    , "d"(wdot[107]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+108*spec_stride) 
    , "d"(wdot[108]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+109*spec_stride) 
    , "d"(wdot[109]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+110*spec_stride) 
    , "d"(wdot[110]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+111*spec_stride) 
    , "d"(wdot[111]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+112*spec_stride) 
    , "d"(wdot[112]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+113*spec_stride) 
    , "d"(wdot[113]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+114*spec_stride) 
    , "d"(wdot[114]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+115*spec_stride) 
    , "d"(wdot[115]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+116*spec_stride) 
    , "d"(wdot[116]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+117*spec_stride) 
    , "d"(wdot[117]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+118*spec_stride) 
    , "d"(wdot[118]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+119*spec_stride) 
    , "d"(wdot[119]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+120*spec_stride) 
    , "d"(wdot[120]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+121*spec_stride) 
    , "d"(wdot[121]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+122*spec_stride) 
    , "d"(wdot[122]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+123*spec_stride) 
    , "d"(wdot[123]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+124*spec_stride) 
    , "d"(wdot[124]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+125*spec_stride) 
    , "d"(wdot[125]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+126*spec_stride) 
    , "d"(wdot[126]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+127*spec_stride) 
    , "d"(wdot[127]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+128*spec_stride) 
    , "d"(wdot[128]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+129*spec_stride) 
    , "d"(wdot[129]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+130*spec_stride) 
    , "d"(wdot[130]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+131*spec_stride) 
    , "d"(wdot[131]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+132*spec_stride) 
    , "d"(wdot[132]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+133*spec_stride) 
    , "d"(wdot[133]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+134*spec_stride) 
    , "d"(wdot[134]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+135*spec_stride) 
    , "d"(wdot[135]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+136*spec_stride) 
    , "d"(wdot[136]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+137*spec_stride) 
    , "d"(wdot[137]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+138*spec_stride) 
    , "d"(wdot[138]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+139*spec_stride) 
    , "d"(wdot[139]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+140*spec_stride) 
    , "d"(wdot[140]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+141*spec_stride) 
    , "d"(wdot[141]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+142*spec_stride) 
    , "d"(wdot[142]) : "memory"); 
}

