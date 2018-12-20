#ifndef PELEC_INDEX_H_
#define PELEC_INDEX_H_

#include <mechanism.h>

#ifdef __cplusplus
#define URHO      0
#define QRHO      0
#define QGAMC     0
#else
#define URHO      1
#define QRHO      1
#define QGAMC     1
#endif

#define UMX       URHO  + 1
#define UMY       UMX   + 1
#define UMZ       UMY   + 1
#define UEDEN     UMZ   + 1
#define UEINT     UEDEN + 1
#define UTEMP     UEINT + 1
#define NTHERM    7
#define UFA       UTEMP + 1
#define UFS       UFA   + NUM_ADVECT
#define UFX       UFS   + NUM_SPECIES
#define NVAR      UFX   + NUM_AUXILIARY

#define QU        QRHO   + 1
#define QV        QU     + 1
#define QW        QV     + 1
#define QGAME     QW     + 1
#define QPRES     QGAME  + 1
#define QREINT    QPRES  + 1
#define QTEMP     QREINT + 1
#define QFA       QTEMP  + 1
#define QTHERM    NTHERM + 1
#define QFS       QFA    + NUM_ADVECT
#define QFX       QFS    + NUM_SPECIES
#define QVAR      QFX    + NUM_AUXILIARY

#define QC        QGAMC  + 1
#define QCSML     QC     + 1
#define QDPDR     QCSML  + 1
#define QDPDE     QDPDR  + 1
#define QRSPEC    QDPDE  + 1
#define NQAUX     QRSPEC + 1

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
