/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    pbd_types.h ( CE Library, OpenCL)
 *
 * COMMENTS:
 *    PDB constraint types for deformable materials.
 */

#ifndef __PBD_TYPES_H__
#define __PBD_TYPES_H__

#define DISTANCE        -264586729
#define BEND            5106433
#define STRETCHSHEAR    1143749888
#define BENDTWIST       1915235160
#define PIN             157323
#define PINORIENT       1780757740
#define PRESSURE        1396538961
#define TRIAREA         788656672
#define TETVOLUME       -215389979
#define TRIANGLEBEND    -120001733
#define ANGLE           187510551
#define TETFIBER        892515453
#define TETFIBERNORM    -303462111
#define PTPRIM          -600175536
#define DISTANCELINE    1621136047
#define DISTANCEPLANE   -139877165
#define TRIARAP         788656539
#define TRIARAPNL       1634014773
#define TRIARAPNORM     -711728545
#define TETARAP         -92199131
#define TETARAPNL       -1666554577
#define TETARAPVOL      -1532966034
#define TETARAPNLVOL    1593379856
#define TETARAPNORM     -885573303
#define TETARAPNORMVOL  -305911678
#define SHAPEMATCH      -841721998

// Tests for various ARAP constraint types and bit flags for solving.
#define LINEARENERGY    (1 << 0)
#define NORMSTIFFNESS   (1 << 1)

static int
isTriARAP(const int ctype)
{
    return ctype == TRIARAP || ctype == TRIARAPNL || ctype == TRIARAPNORM;
}

static int
isTetARAPVol(const int ctype)
{
    return ctype == TETARAPVOL || ctype == TETARAPNLVOL || ctype == TETARAPNORMVOL;
}

static int
isTetARAP(const int ctype)
{
    return ctype == TETARAP || ctype == TETARAPNL || ctype == TETARAPNORM || isTetARAPVol(ctype);
}

static int
isLinearARAP(const int ctype)
{
    return ctype == TRIARAP || ctype == TETARAP || ctype == TETARAPVOL || ctype == TETARAPNORM || ctype == TETARAPNORMVOL;
}

static int
isNonLinearARAP(const int ctype)
{
    return ctype == TRIARAPNL || ctype == TETARAPNL || ctype == TETARAPNLVOL;
}

static int
isStiffnessNormalized(const int ctype)
{
    return ctype == TRIARAPNORM || ctype == TETARAPNORM || ctype == TETARAPNORMVOL || ctype == TETFIBERNORM;
}

static int
isTetFiber(const int ctype)
{
    return ctype == TETFIBER || ctype == TETFIBERNORM;
}

static uint
FEMFlags(const int ctype)
{
    return (isLinearARAP(ctype) ? LINEARENERGY : 0u) | 
           (isStiffnessNormalized(ctype) ? NORMSTIFFNESS : 0u);
}

// Returns inverse mass of the point, possibly considering
// the stopped variable if provided (bit 0 == no position update.)
static float
safediv(float a, float b)
{
    return select(0.0f, a / b, b > 0.0f);
}

static float
invMass(global const float *mass, global const int *stopped, int pt)
{
#ifdef HAS_stopped
    return select(safediv(1.0f, mass[pt]), 0.0f, stopped[pt] & 1);
#else
    return safediv(1.0f, mass[pt]);
#endif
}

// Returns inverse inertia of the point, possibly considering
// the stopped variable if provided (bit 1 == no orientation update.)
static float
invInertia(global const float *inertia, global const int *stopped, int pt)
{
#ifdef HAS_stopped
    return select(safediv(1.0f, inertia[pt]), 0.0f, stopped[pt] & 2);
#else
    return safediv(1.0f, inertia[pt]);
#endif
}

#endif
