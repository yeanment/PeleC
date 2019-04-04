#ifndef PELEC_INDEX_H_
#define PELEC_INDEX_H_

#include <mechanism.h>

#ifdef __cplusplus
#define URHO      0
#define UMX       1
#define UMY       2
#define UMZ       3
#define UEDEN     4
#define UEINT     5
#define UTEMP     6
#define UFA       7

#define QRHO      0
#define QU        1
#define QV        2
#define QW        3
#define QGAME     4
#define QPRES     5
#define QREINT    6
#define QTEMP     7
#define QFA       8

#define QGAMC     0
#define QC        1
#define QCSML     2
#define QDPDR     3
#define QDPDE     4
#define QRSPEC    5

#define GDRHO     0
#define GDU       1
#define GDV       2
#define GDW       3
#define GDPRES    4
#define GDGAME    5

#else
#define URHO      1
#define UMX       2
#define UMY       3
#define UMZ       4
#define UEDEN     5
#define UEINT     6
#define UTEMP     7
#define UFA       8

#define QRHO      1
#define QU        2
#define QV        3
#define QW        4
#define QGAME     5
#define QPRES     6
#define QREINT    7
#define QTEMP     8
#define QFA       9

#define QGAMC     1
#define QC        2
#define QCSML     3
#define QDPDR     4
#define QDPDE     5
#define QRSPEC    6

#define GDRHO     1
#define GDU       2
#define GDV       3
#define GDW       4
#define GDPRES    5
#define GDGAME    6
#endif


#define NTHERM    7
#define QTHERM    8

#define UFS       UFA   + NUM_ADVECT
#define UFX       UFS   + NUM_SPECIES
#define NVAR      NTHERM + NUM_ADVECT + NUM_SPECIES + NUM_AUXILIARY

#define QFS       QFA    + NUM_ADVECT
#define QFX       QFS    + NUM_SPECIES
#define QVAR      QTHERM + NUM_ADVECT + NUM_SPECIES + NUM_AUXILIARY
#define NQ        QVAR
#define NQAUX     6
#define NGDNV     6

#define npassive  NUM_SPECIES + NUM_ADVECT + NUM_AUXILIARY + 3 - BL_SPACEDIM

#ifdef __cplusplus
#define Density   URHO
#define Xmom      UMX
#define Ymom      UMY
#define Zmom      UMZ
#define Eden      UEDEN
#define Eint      UEINT
#define Temp      UTEMP
#define FirstAdv  UFA
#define NumAdv    NUM_ADVECT
#define FirstSpec UFS
#define NumSpec   NUM_SPECIES
#define FirstAux  UFX
#define NumAux    NUM_AUXILIARY
#endif

#endif 
