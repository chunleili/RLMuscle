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
 * NAME:    pbd_constraints.cl ( CE Library, OpenCL)
 *
 * COMMENTS:
 *    PDB constraints for deformable materials.
 */

// Constraint types.
#include <pbd_types.h>

// For 64-bit compilation testing.
#if 0
#define USE_DOUBLE
#define USE_LONG
#endif

#include <platform.h>
#include <typedefines.h>
#include <matrix.h>
#include <quaternion.h>
#include <reduce.h>
#include <svd3.h>


// For compilation testing.
//#define HAS_dP
//#define HAS_dPw

#if defined(HAS_dP) && defined(HAS_dPw)
#define JACOBI
#define DPPARM   dP,dPw,
#define DPPARMIN global fpreal *dP, global fpreal *dPw,
#else
#define DPPARM
#define DPPARMIN
#endif


// For compilation testing.
#ifdef CONSTRAINT_all
#define CONSTRAINT_distance
#define CONSTRAINT_pin
#define CONSTRAINT_distanceline
#define CONSTRAINT_distanceplane
#define CONSTRAINT_triarap
#define CONSTRAINT_tetarap
#define CONSTRAINT_triarea
#define CONSTRAINT_tetvolume
#define CONSTRAINT_bend
#define CONSTRAINT_trianglebend
#define CONSTRAINT_angle
#define CONSTRAINT_tetfiber
#define CONSTRAINT_ptprim
#define CONSTRAINT_stretchshear
#define CONSTRAINT_bendtwist
#define CONSTRAINT_pinorient
#define CONSTRAINT_pressure
#define CONSTRAINT_shapematch
#endif

#ifdef HAS_allconstraintparm
#define HAS_restvector
#define HAS_restdir
#define HAS_restmatrix
#define HAS_orient
#define HAS_orientprevious
#define HAS_inertia
#define HAS_pressuregradient
#define HAS_volume
#define HAS_rest
#endif


static int
inCompressBand(const float curlen, const float restlen)
{
    // Returns whether in the compression band and should use compression stiffness.
    // No-op if parameter is not supplied to the kernel.
#if defined(HAS_compressstiffness)
    return (curlen < restlen);
#else
    return 0;
#endif
}

// Update dP and dPw with input vector delta.
static void
updatedP(fpreal3 dp, int pt, global fpreal *dP, global fpreal *dPw)
{
    dp += vload3(pt, dP);
    vstore3(dp, pt, dP);
    dPw[pt] += 1;
}

static void
distanceUpdateXPBD(float timeinc,
                   int idx,
                   int ptidx,
                   global const int *pts,
                   global fpreal *L,
                   global fpreal *P,
                   global const fpreal *pprev,
                   DPPARMIN
                   global const float *masses,
                   global const int *stopped,
                   float restlength,
                   float kstiff,
                   float kdampratio,
                   float kstiffcompress)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float wsum = invmass0 + invmass1;
    if (wsum == 0.0f)
        return;

    fpreal3 n = p1 - p0;
    fpreal d = length(n);
    if (d < 1e-6f)
        return;

    // Check if we should use compression stiffness value, which also
    // implies we need to use a separate Lagrange multiplier in L.
    int loff = inCompressBand(d, restlength);
    kstiff = select(kstiff, kstiffcompress, loff);
    if (kstiff == 0.0f)
        return;
    fpreal l = L[idx * 3 + loff];

    // XPBD term.
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    // Constraint calc.
    fpreal C = d - restlength;
    n /= d;
    fpreal3 gradC = n;

    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        dsum = gamma * (-dot(gradC, p0 - prev0) + dot(gradC, p1 - prev1));
        gamma += 1.0f;
    }
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);
    fpreal3 dp =  n * -dL;

#ifdef JACOBI
    updatedP(invmass0 * dp, pt0, dP, dPw);
    updatedP(-invmass1 * dp, pt1, dP, dPw);
#else
    p0 += invmass0 * dp;
    p1 -= invmass1 * dp;
    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    // Don't update L for Jacobi iterations.
    L[idx * 3 + loff] += dL;
#endif
}


static void
dihedralUpdateXPBD(float timeinc,
                   int idx,
                   int ptidx,
                   global const int *pts,
                   global fpreal *L,
                   global fpreal *P,
                   global const fpreal *pprev,
                   global const float *masses,
                   global const int *stopped,
                   float restlength,
                   float kstiff,
                   float kdampratio)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    int pt3 = pts[ptidx + 3];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    fpreal3 p3 = vload3(pt3, P);
    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);
    float invmass3 = invMass(masses, stopped, pt3);
    fpreal l = L[idx*3];

    fpreal3 e = p3 - p2;
    fpreal elen = length(e);

    // Gradients of bend constraint, from Appendix A of
    // Müller M, Chentanez N, Kim T, Macklin M. Strain Based Dynamics
    fpreal3 n1 = cross(p3 - p0, p2 - p0);
    fpreal3 n2 = cross(p2 - p1, p3 - p1);
    fpreal n1len2 = dot(n1, n1);
    fpreal n2len2 = dot(n2, n2);
    if (n1len2 < 1e-12f || n2len2 < 1e-12f || elen < 1e-12f)
        return;
    n1 /= n1len2;
    n2 /= n2len2;
    fpreal invElen = 1.0f / elen;

    // Negate derivatives if necessary.
    float s = select(-1.0f, 1.0f, dot(cross(n1, n2), e) > 0.0f);

    fpreal3 grad0 = s * elen * n1;
    fpreal3 grad1 = s * elen * n2;
    fpreal3 grad2 = s * (dot(p0 - p3, e) * invElen * n1 + dot(p1 - p3, e) * invElen * n2);
    fpreal3 grad3 = s * (dot(p2 - p0, e) * invElen * n1 + dot(p2 - p1, e) * invElen * n2);

    fpreal wsum = invmass0 * dot(grad0, grad0) +
                  invmass1 * dot(grad1, grad1) +
                  invmass2 * dot(grad2, grad2) +
                  invmass3 * dot(grad3, grad3);
    if (wsum == 0.0)
        return;

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    // Constraint value.
    n1 = normalize(n1);
    n2 = normalize(n2);
    fpreal phi = acos(clamp(dot(n1, n2), (fpreal)-1.0, (fpreal)1.0));

    // Use s to express phi as -PI..PI
    phi *= s;
    // restlength is in degrees.
    fpreal C = phi - radians(restlength);
    C *= s;

    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal3 prev3 = vload3(pt3, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        dsum = dot(grad0, p0 - prev0) +
               dot(grad1, p1 - prev1) +
               dot(grad2, p2 - prev2) +
               dot(grad3, p3 - prev3);
        dsum *= gamma;
        gamma += 1.0f;
    }

    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);

    // Update points.
    p0 += dL * invmass0 * grad0;
    p1 += dL * invmass1 * grad1;
    p2 += dL * invmass2 * grad2;
    p3 += dL * invmass3 * grad3;

    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    vstore3(p3, pt3, P);
    L[idx*3] += dL;
}

static void
distancePosUpdateXPBD(float timeinc,
                      int idx,
                      int pt1,
                      fpreal3 p0,
                      fpreal3 p1,
                      global fpreal *L,
                      global fpreal *P,
                      global const fpreal *pprev,
                      DPPARMIN
                      global const float *masses,
                      global const int *stopped,
                      float restlength,
                      float kstiff,
                      float kdampratio,
                      float kstiffcompress)
{
    float invmass1 = invMass(masses, stopped, pt1);
    if (invmass1 == 0.0)
        return;
    fpreal wsum = invmass1;

    fpreal3 n = p1 - p0;
    fpreal d = length(n);
    if (d < 1e-6)
        return;

    // Check if we should use compression stiffness value, which also
    // implies we need to use a separate Lagrange multiplier in L.
    int loff = inCompressBand(d, restlength);
    kstiff = select(kstiff, kstiffcompress, loff);
    if (kstiff == 0.0f)
        return;
    fpreal l = L[idx * 3 + loff];

    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    fpreal C = d - restlength;
    n /= d;
    fpreal3 gradC = n;

    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        dsum = gamma * dot(gradC, p1 - prev1);
        gamma += 1.0f;
    }
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);
    fpreal3 dp =  n * -dL;

#ifdef JACOBI
    updatedP(-invmass1 * dp, pt1, dP, dPw);
#else
    p1 -= invmass1 * dp;
    vstore3(p1, pt1, P);
    // Don't update L for Jacobi iterations.
    L[idx * 3 + loff] += dL;
#endif
}


static void
triangleBendUpdateXPBD(float timeinc,
                       int idx,
                       int ptidx,
                       global const int *pts,
                       global fpreal *L,
                       global fpreal *P,
                       global const fpreal *pprev,
                       global const float *masses,
                       global const int *stopped,
                       float restlength,
                       float kstiff,
                       float kdampratio)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);
    fpreal l = L[idx*3];
    // Centroid
    fpreal3 c = (p0 + p1 + p2) / 3;
    // Vertex to centroid
    fpreal3 n = p1 - c;

    // |gradb|^2 = 1 / 9
    // |gradv|^2 = 4 / 9
    fpreal wsum = invmass0 / 9.0f +
                 4.0f * invmass1 / 9.0f +
                 invmass2 / 9.0f;
    if (wsum == 0.0f)
        return;

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

     // Normalize and calc gradients.
    float d = length(n);
    n *= select(1.0f / d, 0.0f, d < 1e-6f);
    fpreal3 gradv = 2.0f * n / 3.0f;
    fpreal3 gradb = gradv / -2.0f;

    fpreal C = d - restlength;
    fpreal dsum = 0, gamma = 1;

    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        dsum = dot(gradb, p0 - prev0) +
               dot(gradv, p1 - prev1) +
               dot(gradb, p2 - prev2);
        dsum *= gamma;
        gamma += 1.0f;
    }

    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);
    // Update points.
    p0 += dL * invmass0 * gradb;
    p1 += dL * invmass1 * gradv;
    p2 += dL * invmass2 * gradb;

    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    L[idx*3] += dL;
}

static void
angleUpdateXPBD(float timeinc,
                int idx,
                int ptidx,
                global const int *pts,
                global fpreal *L,
                global fpreal *P,
                global const fpreal *pprev,
                global const float *masses,
                global const int *stopped,
                float restlength,
                float kstiff,
                float kdampratio)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);
    fpreal l = L[idx*3];

    fpreal3 n1 = p1 - p0;
    fpreal e1len = length(n1);
    fpreal3 n2 = p2 - p1;
    fpreal e2len = length(n2);
    if (e1len < 1e-6f || e2len < 1e-6f)
        return;
    // Normalize
    n1 /= e1len;
    n2 /= e2len;
    fpreal d = dot(n1, n2);
    d = clamp(d, (fpreal)-1.0, (fpreal)1.0);
    if (fabs(d) >= (1.0f - 1e-6f))
        return;

    // C = acos(d) - restlen;
    // dC/dp0 = (-1/sqrt(1-d^2)) * dd/dP0;
    fpreal ds = (-1.0f / sqrt(1.0f - d * d));
    fpreal3 grad0 = ds * (n1 * d - n2) / e1len;
    fpreal3 grad2 = ds * (n1 - n2 * d) / e2len;
    fpreal3 grad1 = -grad0 - grad2;

    fpreal wsum = invmass0 * dot(grad0, grad0) +
                 invmass1 * dot(grad1, grad1) +
                 invmass2 * dot(grad2, grad2);

    if (wsum == 0.0)
        return;
    fpreal phi = acos(d);
    // restlength is in degrees.
    fpreal C = phi - radians(restlength);

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    fpreal dsum = 0, gamma = 1;

    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        dsum = dot(grad0, p0 - prev0) +
               dot(grad1, p1 - prev1) +
               dot(grad2, p2 - prev2);
        dsum *= gamma;
        gamma += 1.0f;
    }

    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);
    // Update points.
    p0 += dL * invmass0 * grad0;
    p1 += dL * invmass1 * grad1;
    p2 += dL * invmass2 * grad2;
    L[idx*3] += dL;

    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
}

static void
stretchShearUpdateXPBD(float timeinc,
                       int idx,
                       int ptidx,
                       global const int *pts,
                       global fpreal *Ls,
                       global fpreal *P,
                       global const fpreal *pprev,
                       global const float *masses,
                       global const int *stopped,
                       global fpreal *orient,
                       global const fpreal *orientprev,
                       global const float *inertias,
                       float restlength,
                       float kstiff,
                       float kdampratio)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    quat q0 = vload4(pt0, orient);
    float invmassp0 = invMass(masses, stopped, pt0);
    float invmassp1 = invMass(masses, stopped, pt1);
    float invmassq0 = invInertia(inertias, stopped, pt0);
    fpreal3 L = vload3(idx, Ls);

    fpreal3 d3;    //third director d3 = q0 * e_3 * q0_conjugate
    d3.x = 2.0f * (q0.x * q0.z + q0.w * q0.y);
    d3.y = 2.0f * (q0.y * q0.z - q0.w * q0.x);
    d3.z = q0.w * q0.w - q0.x * q0.x - q0.y * q0.y + q0.z * q0.z;

    // If restlength is zero we'll get NAN's, but if we just return when restlength==0,
    // we won't get any stretch stiffness keeping the points in place.  We could enforce
    // in constraint creation, but users could always override / scale restlength.
    // So punt and enforce a minimum restlength at constraint solve time.
    restlength = max(restlength, 1e-6f);

    fpreal3 gradp = -1 / restlength;
    // ||gradp|| =  1/restlength^2
    // ||gradq0|| = 4;
    fpreal wsum = (invmassp0 + invmassp1) / (restlength * restlength) +
                  invmassq0 * 4;
    if (wsum == 0.0f)
        return;

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    // Vector-valued constraint function.
    fpreal3 C = (p1 - p0) / restlength - d3;

    fpreal3 dsum = 0;
    fpreal gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prevp0 = vload3(pt0, pprev);
        fpreal3 prevp1 = vload3(pt1, pprev);
        quat prevq0 = vload4(pt0, orientprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        // Damping for linear part of constraint on points.
        dsum = gradp * ((p0 - prevp0) - (p1 - prevp1));

        // Damping for orientation part of constraint.
        // dq/dt with timeinc factored out (reapplied in gamma)
        quat dq0_dt = (q0 - prevq0);
        // dC/dt = dC/dq0 * dq0/dt = dq0/dt * e3 * q0.conjugate
        quat e3_qconj =  q0.yxwz * (quat)(1, -1, 1, 1);
        // Ignore scalar part.
        dq0_dt = qmultiply(dq0_dt, e3_qconj);
        dsum -= 2.0f * dq0_dt.xyz;

        dsum *= gamma;
        gamma += 1.0f;
    }

    fpreal3 dL = (-C - alpha * L - dsum) / (gamma * wsum + alpha);

    // Update points.
    p0 += invmassp0 * dL * gradp;
    p1 -= invmassp1 * dL * gradp;

    // Compute q*e_3.conjugate (cheaper than quaternion product)
    quat q_e3_conj =  q0.yxwz * (quat)(-1, 1, -1, 1);

    // Update orientation.
    // gradq0^T * dL = dL * q * e_3.conjugate
    q0 -= (2.0f * invmassq0) * qmultiply((quat)(dL, 0), q_e3_conj);
    q0 = normalize(q0);
    L += dL;

    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore4(q0, pt0, orient);
    vstore3(L, idx, Ls);
}

static void
bendTwistUpdateXPBD(float timeinc,
                    int idx,
                    int ptidx,
                    global const int *pts,
                    global fpreal *Ls,
                    global fpreal *orient,
                    global const fpreal *orientprev,
                    global const float *inertias,
                    global const int *stopped,
                    global const float *restvectors,
                    float kstiff,
                    float kdampratio)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    quat q0 = vload4(pt0, orient);
    quat q1 = vload4(pt1, orient);
    quat q0conj = qconjugate(q0);

    float invmassq0 = invInertia(inertias, stopped, pt0);
    float invmassq1 = invInertia(inertias, stopped, pt1);
    fpreal3 L = vload3(idx, Ls);

    quat restvector = vload4f(idx, restvectors);

    quat omega = qmultiply(q0conj, q1);   //darboux vector
    omega = qcloser(omega, restvector);

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;
    // Vector constraint function:
    // Zero in w since discrete Darboux vector does not have
    // vanishing scalar part.
    fpreal3 C = omega.xyz;
    // ||gradq0|| = ||gradq1|| = 1
    fpreal wsum = invmassq0 + invmassq1;
    if (wsum == 0.0f)
        return;

    fpreal3 dsum = 0;
    fpreal gamma = 1;
    //printf("kdamp = %g\n", kdamp);
    if (kdampratio > 0)
    {
        // Compute damping terms.
        quat prevq0 = vload4(pt0, orientprev);
        quat prevq1 = vload4(pt1, orientprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        // Angular velocities with timeinc factored out
        // (reapplied in gamma)
        // w = (q0 * prevq0_conj) * (2 / timeinc)
        quat w0 = qmultiply(q0, qconjugate(prevq0));// * 2.0f;
        quat w1 = qmultiply(q1, qconjugate(prevq1));// * 2.0f;
        // Zero out scalar part.
        w0.w = w1.w = 0;

        // dq/dt = 1/2 * w * q
        quat dq0_dt = qmultiply(w0, q0);// * 0.5f;
        quat dq1_dt = qmultiply(w1, q1);// * 0.5f;

        // dC/dt = dC/dq0 * dq0/dt = -q1_conj * 1/2 * w * q
        // dC/dt = dC/dq1 * dq1/dt = q0_conj * 1/2 * w * q
        quat dsum4 = qmultiply(q0conj, dq1_dt) -
                    qmultiply(qconjugate(q1), dq0_dt);
        // Ignore scalar part.
        dsum = dsum4.xyz * gamma;
        gamma += 1.0f;
    }

    fpreal3 dL = (-C - alpha * L - dsum) / (gamma * wsum + alpha);
    // gradq0^T * dL = -q1 * dL
    // gradq1^T * dL = q0 * dL
    quat dl4 = (quat)(dL, 0);
    quat dq0 = -invmassq0 * qmultiply(q1, dl4);
    quat dq1 = invmassq1 * qmultiply(q0, dl4);

    q0 = normalize(q0 + dq0);
    q1 = normalize(q1 + dq1);
    L += dL;

    vstore4(q0, pt0, orient);
    vstore4(q1, pt1, orient);
    vstore3(L, idx, Ls);
}

// Constrain orient on single point to match rest vector.
static void
bendTwistOrientUpdateXPBD(float timeinc,
                          int idx,
                          int ptidx,
                          global const int *pts,
                          global fpreal *Ls,
                          global fpreal *orient,
                          global const fpreal *orientprev,
                          global const float *inertias,
                          global const int *stopped,
                          global const float *restvectors,
                          float kstiff,
                          float kdampratio)
{
    int pt1 = pts[ptidx];
    // We're trying to match restvector as the first orientation.
    quat q0 = vload4f(idx, restvectors);
    quat q1 = vload4(pt1, orient);
    quat q0conj = qconjugate(q0);

    float invmassq1 = invInertia(inertias, stopped, pt1);
    if (invmassq1 == 0.0f)
        return;
    fpreal3 L = vload3(idx, Ls);

    // Darboux vector should be fully aligned with q0 (i.e. rest orient).
    quat restvector = (quat)(0, 0, 0, 1);

    quat omega = qmultiply(q0conj, q1);   //Darboux vector
    omega = qcloser(omega, restvector);

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;
    // Vector constraint function:
    // Zero in w since discrete Darboux vector does not have
    // vanishing scalar part.
    fpreal3 C = omega.xyz;
    // ||gradq0|| = ||gradq1|| = 1
    fpreal wsum = invmassq1;

    fpreal3 dsum = 0;
    fpreal gamma = 1;
    //printf("kdamp = %g\n", kdamp);
    if (kdampratio > 0)
    {
        // Compute damping terms.
        quat prevq1 = vload4(pt1, orientprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        // Angular velocities with timeinc factored out
        // (reapplied in gamma)
        // w = (q0 * prevq0_conj) * (2 / timeinc)
        quat w1 = qmultiply(q1, qconjugate(prevq1));// * 2.0f;
        // Zero out scalar part.
        w1.w = 0;

        // dq/dt = 1/2 * w * q
        quat dq1_dt = qmultiply(w1, q1);// * 0.5f;

        // dC/dt = dC/dq1 * dq1/dt = q0_conj * 1/2 * w * q
        quat dsum4 = qmultiply(q0conj, dq1_dt);
        // Ignore scalar part.
        dsum = dsum4.xyz * gamma;
        gamma += 1.0f;
    }

    fpreal3 dL = (-C - alpha * L - dsum) / (gamma * wsum + alpha);
    // gradq1^T * dL = q0 * dL
    quat dl4 = (quat)(dL, 0);
    quat dq1 = invmassq1 * qmultiply(q0, dl4);

    q1 = normalize(q1 + dq1);
    L += dL;

    vstore4(q1, pt1, orient);
    vstore3(L, idx, Ls);
}

static void
pressureUpdatePts(  int npts,
                    global const int *pts,
                    float timeinc,
                    global fpreal * P,
                    global const fpreal * pprevs,
                    global const float * mass,
                    global const int *stopped,
                    float restlength,                    
                    global const float *pressuregradient,
                    global const float *volume,
                    float kstiff,
                    float kstiffcompress,
                    float kdampratio,
                    global fpreal * L,
                    int reduce
                )
{
    size_t lid = 0;
    size_t lsize = 1;
#ifdef __opencl_c_work_group_collective_functions
    if (reduce)
    {
        lid = get_local_id(0);
        lsize = get_local_size(0);
    }
#endif
    
    // Accumulate volume and gradient sums.
    accum_t vold = 0, wsumd = 0, dsumd = 0;
    for (size_t idx = lid; idx < npts; idx += lsize)
    {
        int pt = pts[idx];
        fpreal3 grad = vload3f(pt, pressuregradient);
        float invmass = invMass(mass, stopped, pt);
        wsumd += invmass * dot(grad, grad);
        vold += volume[pt];
        fpreal3 p = vload3(pt, P);
        fpreal3 pprev = vload3(pt, pprevs);
        dsumd += dot(grad, p - pprev);
    }

#ifdef __opencl_c_work_group_collective_functions
    if (reduce)
    {
        vold = work_group_reduce_add(vold);
        wsumd = work_group_reduce_add(wsumd);
        dsumd = work_group_reduce_add(dsumd);
    }
#endif

    fpreal vol = (fpreal)vold;
    fpreal wsum = (fpreal)wsumd;
    fpreal dsum = (fpreal)dsumd;
    fpreal gamma = 1;

    // Check if we should use compression stiffness value, which also
    // implies we need to use a separate Lagrange multiplier in L.
    int loff = inCompressBand(vol, restlength);
    kstiff = select(kstiff, kstiffcompress, loff);
    if (kstiff == 0.0f)
        return;
    fpreal l = L[loff];

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    // Constraint value.
    fpreal C = vol - restlength;

    // More damping terms.
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;
        dsum *= gamma;
        gamma += 1.0f;
    }

    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);

    // Apply per-point correction
    for (size_t idx = lid; idx < npts; idx += lsize)
    {
        int pt = pts[idx];
        fpreal3 grad = vload3f(pt, pressuregradient);
        fpreal3 p = vload3(pt, P);
        float invmass = invMass(mass, stopped, pt);
        p += invmass * dL * grad;
        vstore3(p, pt, P);
    }

    if (lid == 0)
        L[loff] += dL;
}

static void
pressureUpdateXPBD(float timeinc,
                 int idx,
                 global const int *pts_index,
                 global const int *pts,
                 global fpreal *L,
                 global fpreal *P,
                 global const fpreal *pprevs,
                 global const float *masses,
                 global const int *stopped,
                 float restlength,
                 float kstiff,
                 float kdampratio,
                 float kstiffcompress,
                 global const float *pressuregradient,
                 global const float *volume)
{
    int ptidx = pts_index[idx];
    int npts = pts_index[idx + 1] - ptidx;
    // We can only use device-side enqueue when *not* using SINGLE_WORKGROUP,
    // since we can't assume we'll be done writing to the points by the next
    // constraint coloring update (in fact we likely won't be.)
    // We also only have enough precision in our accumulator to ensure order
    // independence if it's 64-bit and fpreal is 32-bit.
#if defined(__opencl_c_work_group_collective_functions) && defined(__opencl_c_device_enqueue) && \
    !defined(SINGLE_WORKGROUP) && !defined(SINGLE_WORKGROUP_SPANS) && \
    ACCUM_PREC==64 && FPREAL_PREC==32

    // Arbitrary cutoff, seems reasonable on RTX 2080 Super.
    int lsize = 64;
    if (npts > 100)
    {
        // We need to check this enqueue fails, which it easily can past 100 pressure
        // objects or so due to the small on-device queue.  If so, fall through and process
        // the constraint in this work item as if it had a small number of points.
        // NOTE: the summations in pressureUpdatePts have to be order independent for this
        // approach to remain deterministic, since we can't necessarily predict the order
        // of the enqueue failures as its dependent on how fast the GPU queue empties.
        if (enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                   ndrange_1D(lsize, lsize),
                   ^{
                        pressureUpdatePts(npts, &pts[ptidx], timeinc,
                                          P, pprevs, masses, stopped,
                                          restlength, pressuregradient, volume,
                                          kstiff, kstiffcompress, kdampratio, 
                                          &L[idx * 3], 1);
                    }) == CLK_SUCCESS)
            return;
    }
#endif
    pressureUpdatePts(npts, &pts[ptidx], timeinc,
                      P, pprevs, masses, stopped,
                      restlength, pressuregradient, volume,
                      kstiff, kstiffcompress, kdampratio, 
                      &L[idx * 3], 0);    
}

static void
tetVolumeUpdateXPBD(float timeinc,
                    int idx,
                    int ptidx,
                    global const int *pts,
                    global fpreal *L,
                    global fpreal *P,
                    global const fpreal *pprev,
                    DPPARMIN
                    global const float *masses,
                    global const int *stopped,
                    float restlength,
                    float kstiff,
                    float kdampratio,
                    float kstiffcompress,
                    int loff)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    int pt3 = pts[ptidx + 3];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    fpreal3 p3 = vload3(pt3, P);

    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);
    float invmass3 = invMass(masses, stopped, pt3);

    // Calculate volume and gradients with material coordinates.
    fpreal3 d1 = p1 - p0;
    fpreal3 d2 = p2 - p0;
    fpreal3 d3 = p3 - p0;
    fpreal3 grad1 = cross(d3, d2) / 6;
    fpreal3 grad2 = cross(d1, d3) / 6;
    fpreal3 grad3 = cross(d2, d1) / 6;
    fpreal3 grad0 = -(grad1 + grad2 + grad3);

    fpreal wsum =    invmass0 * dot(grad0, grad0) +
                     invmass1 * dot(grad1, grad1) +
                     invmass2 * dot(grad2, grad2) +
                     invmass3 * dot(grad3, grad3);
    if (wsum == 0)
        return;

    fpreal volume = dot(cross(d2, d1), d3) / 6;
    // Check if we should use compression stiffness value, which also
    // implies we need to use a separate Lagrange multiplier in L.
    int comp = inCompressBand(volume, restlength);
    kstiff = select(kstiff, kstiffcompress, comp);
    loff += comp;
    loff = min(loff, 2);

    if (kstiff == 0.0f)
        return;
    fpreal l = L[idx * 3 + loff];

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    // Constraint value.
    fpreal C = volume - restlength;

    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal3 prev3 = vload3(pt3, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        dsum = dot(grad0, p0 - prev0) +
               dot(grad1, p1 - prev1) +
               dot(grad2, p2 - prev2) +
               dot(grad3, p3 - prev3);
        dsum *= gamma;
        gamma += 1.0f;
    }

    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);

#ifdef JACOBI
    updatedP(dL * invmass0 * grad0, pt0, dP, dPw);
    updatedP(dL * invmass1 * grad1, pt1, dP, dPw);
    updatedP(dL * invmass2 * grad2, pt2, dP, dPw);
    updatedP(dL * invmass3 * grad3, pt3, dP, dPw);
#else
    // Update points.
    p0 += dL * invmass0 * grad0;
    p1 += dL * invmass1 * grad1;
    p2 += dL * invmass2 * grad2;
    p3 += dL * invmass3 * grad3;

    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    vstore3(p3, pt3, P);
    L[idx * 3 + loff] += dL;
#endif
}

// Calc the I4 invariant for determing inversion
// along the w direction.
static float
calcI4(const mat3 F, const float3 w)
{
    // Check for any inversion at all before expensive check for
    // inversion in the w direction.
    float I4 = 1;
    if (det3(F) < 0.0f)
    {
        mat3 U, Sigma, V, Vt, S;
        // TODO-Do we need the rotation-variant form of SVD from DD?
        svd3(F, U, Sigma, V);
        // Get the stretch matrix.
        // S = V * Sigma * V^T
        transpose3(V, Vt);
        mat3mul(Sigma, Vt, U);
        mat3mul(V, U, S);
        // I4 = w^T (S w)
        // S reprents the stretch in *material space*, so we
        // scale the original materialw by that, then dot
        // with the non-scaled to check for an inversion (i.e. < 0)
        float3 Sw = mat3vecmul(S, w);
        I4 = dot(w, Sw);
    }
    return I4;
}


static void
tetFiberUpdateXPBD(float timeinc,
                   int idx,
                   int ptidx,
                   global const int *pts,
                   global fpreal *L,
                   global fpreal *P,
                   global const fpreal *pprev,
                   global const float *masses,
                   global const int *stopped,
                   float restlength,
                   global const float *restvector,
                   global const float *restmatrix,
                   float kstiff,
                   float kdampratio,
                   uint flags)

{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    int pt3 = pts[ptidx + 3];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    fpreal3 p3 = vload3(pt3, P);

    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);
    float invmass3 = invMass(masses, stopped, pt3);

    fpreal l = L[idx * 3];

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    // The alpha term should really be 1 / (stiffness * restvolume)
    // since our FEM energy should be integrated over the entire
    // linear element.
    if (flags & NORMSTIFFNESS)
        alpha /= restlength;
    alpha /= timeinc * timeinc;
    float gradscale = 1;
    fpreal psi = 0;

    // Constraint value.
    // Ds = shape space basis, Dm = material space.
    mat3 Ds, Dminv, Ht;
    // Ds = | p0-p3 p1-p3 p2-p3 |
    mat3fromcols(p0 - p3, p1 - p3, p2 - p3, Ds);
    fpreal4 rv = vload4f(idx, restvector); 
    // Dm^-1 is stored in restmatrix.
    if (restmatrix)
        mat3load(idx, restmatrix, Dminv);
    if (restmatrix && squaredNorm3(Dminv) > 0)
    {
        // Anisotropic ARAP from HOBAK and 8.4.2 in Dynamic Deformables.
        // F = Ds * Dm^-1
        mat3 F;
        mat3mul(Ds, Dminv, F);
        fpreal fiberscale = rv.w;
        fpreal3 w = rv.xyz;
        // Calc same values as old constraint code
        // to unify gradient calculation.
        fpreal3 wTDminvT = mat3Tvecmul(Dminv, w);
        fpreal3 FwT = mat3Tvecmul(F, w);
        
        const fpreal I5 = dot(FwT, FwT);
        // TODO - handle I5 == 0
        const fpreal sqrtI5inv = native_rsqrt(I5);
        // I4 invariant, the sign of which indicates inversion along
        // fiber direction.
        fpreal I4 = calcI4(F, w);
        int SI4 = I4 > 0 ? 1 : -1;
        // Fiber energy.
        // psi = 0.5 * (sqrt(I5) - SI4 * fiberscale)^2
        fpreal diff = sqrt(I5) - SI4 * fiberscale;
        psi = 0.5 * diff * diff;
        // If we're not near the singularity, just use the generic
        // formula, else use the reflected value at 2.
        const fpreal dPsidI5 = (1.0  - SI4 * fiberscale * sqrtI5inv);
        gradscale = fabs(I4) > 1e-4 ? dPsidI5 : -(1.0 - fiberscale / 2.0);
        // Piola-Kirchhoff stress tensor (see comment below).
        outerprod3(wTDminvT, FwT, Ht);
    }
    else
    {
        // Pre-multiplied w^T * Dm^-T stored in restvector.
        fpreal3 wTDminvT = rv.xyz;
        // F = Ds * Dm^-1
        // FwT = (F w)^T = (w^T * Dm^-T) * Ds^T
        fpreal3 FwT = mat3Tvecmul(Ds, wTDminvT);
        // psi = 0.5 * w^T * F^T * F * w
        psi = 0.5 * dot(FwT, FwT);
        // Take the square root for "linear" response and correct gradscale.
        // The "normalized stiffness" constraints (i.e. the correct ones) always
        // use linear response.
        if (flags & (LINEARENERGY | NORMSTIFFNESS))
        {
         
            psi = sqrt(2 * psi);
            gradscale = 1.0f / psi;
        }
        // Piola-Kirchhoff stress tensor.
        // P = F * w * w^T
        // First three gradients are columns of H, or rows of H^T.
        // H = P * Dm^-T
        // H = Ds * Dm^-1 * w * w^T * Dm^-T
        // Ht = Dm^-1 * w * w^T * Dm^-T * Ds^T
        // Ht =  Dm^-1 * w * (F w)^T
        // Ht =  (w^T * Dm^-T)^T * (F w)^T
        outerprod3(wTDminvT, FwT, Ht);
    }

    fpreal3 grad0 = gradscale * Ht[0];
    fpreal3 grad1 = gradscale * Ht[1];
    fpreal3 grad2 = gradscale * Ht[2];
    fpreal3 grad3 = -grad0 - grad1 - grad2;

    fpreal wsum =    invmass0 * dot(grad0, grad0) +
                     invmass1 * dot(grad1, grad1) +
                     invmass2 * dot(grad2, grad2) +
                     invmass3 * dot(grad3, grad3);
    if (wsum == 0.0)
        return;

    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal3 prev3 = vload3(pt3, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        // We also need to fix up beta term with rest volume, as with alpha.
        if (flags & NORMSTIFFNESS)
            beta *= restlength;
        gamma = alpha * beta / timeinc;

        dsum = dot(grad0, p0 - prev0) +
               dot(grad1, p1 - prev1) +
               dot(grad2, p2 - prev2) +
               dot(grad3, p3 - prev3);
        dsum *= gamma;
        gamma += 1.0f;
    }

    fpreal C = psi;
    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);
    // Update points.
    p0 += dL * invmass0 * grad0;
    p1 += dL * invmass1 * grad1;
    p2 += dL * invmass2 * grad2;
    p3 += dL * invmass3 * grad3;

    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    vstore3(p3, pt3, P);
    L[idx * 3] += dL;
}

static void
triAreaUpdateXPBD(float timeinc,
                    int idx,
                    int ptidx,
                    global const int *pts,
                    global fpreal *L,
                    global fpreal *P,
                    global const fpreal *pprev,
                    DPPARMIN
                    global const float *masses,
                    global const int *stopped,
                    float restlength,
                    float kstiff,
                    float kdampratio,
                    float kstiffcompress)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);

    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);

    // Calculate area and gradients with material coordinates.
    fpreal3 d1 = p1 - p0;
    fpreal3 d2 = p2 - p0;
    fpreal3 n = cross(d2, d1);
    fpreal area = length(n) * 0.5f;
    fpreal3 grad1 = cross(n, d2) / (4 * area);
    fpreal3 grad2 = cross(n,  d1) / (-4 * area);
    fpreal3 grad0 = -grad1 - grad2;

    fpreal wsum =    invmass0 * dot(grad0, grad0) +
                    invmass1 * dot(grad1, grad1) +
                    invmass2 * dot(grad2, grad2);
    if (wsum == 0)
        return;

    // Check if we should use compression stiffness value, which also
    // implies we need to use a separate Lagrange multiplier in L.
    int loff = inCompressBand(area, restlength);
    kstiff = select(kstiff, kstiffcompress, loff);
    if (kstiff == 0.0f)
        return;
    fpreal l = L[idx * 3 + loff];

    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    // Constraint value.
    fpreal C = area - restlength;

    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        dsum = dot(grad0, p0 - prev0) +
               dot(grad1, p1 - prev1) +
               dot(grad2, p2 - prev2);
        dsum *= gamma;
        gamma += 1.0f;
    }

    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);

#ifdef JACOBI
    updatedP(dL * invmass0 * grad0, pt0, dP, dPw);
    updatedP(dL * invmass1 * grad1, pt1, dP, dPw);
    updatedP(dL * invmass2 * grad2, pt2, dP, dPw);
#else
    // Update points.
    p0 += dL * invmass0 * grad0;
    p1 += dL * invmass1 * grad1;
    p2 += dL * invmass2 * grad2;
    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    L[idx * 3 + loff] += dL;
#endif
}

// Return a rotation from triangle space to world space in xform,
// where the triangle normal is the z-axis.
// This MUST match fromTriangleSpaceXform in pbd_constraints.h.
// Returns the area of the triangle.
static fpreal
fromTriangleSpaceXform(fpreal3 p0, fpreal3 p1, fpreal3 p2, mat3 xform)
{
    // Find rotation to triangle space.
    fpreal3 e0 = p1 - p0;
    fpreal3 e1 = p2 - p0;
    fpreal3 n = cross(e1, e0);
    xform[2] = normalize(n);

    xform[1] = normalize(cross(e0, xform[2]));
    xform[0] = cross(xform[1], xform[2]);
    return length(n) * 0.5f;
}


static void
triARAPUpdateXPBD(float timeinc,
                int idx,
                int ptidx,
                global const int *pts,
                global fpreal *L,
                global fpreal *P,
                global const fpreal *pprev,
                DPPARMIN
                global const float *masses,
                global const int *stopped,
                float restlength,
                global float *restvector,
                float kstiff,
                float kdampratio,
                float kstiffcompress,
                uint flags)

{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);

    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);

    // Dm^-1 is stored in restvector.
    mat2 Dminv = vload4f(idx, restvector);

    // Get transform from triangle space and area.
    mat3 xform;
    fpreal area = fromTriangleSpaceXform(p0, p1, p2, xform);
    // Zero area triangle means the xform is garbage (or on CPU, NANs!).
    if (area == 0.0f)
        return;
    // Check if we should use compression stiffness value, which also
    // implies we need to use a separate Lagrange multiplier in L.
    int loff = inCompressBand(area, restlength);
    kstiff = select(kstiff, kstiffcompress, loff);
    if (kstiff == 0.0f)
        return;
    fpreal l = L[idx * 3 + loff];

    // Get triangle space coords.
    // It's faster to multiply by transpose since it uses row dot products.
    fpreal2 P0 = mat3Tvec2mul(xform, p0);
    fpreal2 P1 = mat3Tvec2mul(xform, p1);
    fpreal2 P2 = mat3Tvec2mul(xform, p2);

    // Ds = | P0-P2 P1-P2 |
    mat2 Ds = mat2fromcols(P0 - P2, P1 - P2);
    // F = Ds * Dm^-1
    mat2 F = mat2mul(Ds, Dminv);

    // Compute R with closed form solution for rotation in theta.
    // See http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    fpreal2 m = (fpreal2)(F.x + F.w, F.z - F.y);
    m = normalize(m);
    // m.x is cos(theta), m.y is sin(theta)
    mat2 R = (mat2)(m.x, -m.y,
                    m.y,  m.x);

    // d = F - R
    mat2 d = F - R;

    // psi = ||F - R||^2
    fpreal psi = squaredNorm2(d);
    fpreal gradscale = 2;
    // Take the square root for "linear" response and correct gradscale.
    // The "normalized stiffness" constraints (i.e. the correct ones) always
    // use linear response.
    if (flags & (LINEARENERGY | NORMSTIFFNESS))
    {
        psi = sqrt(psi);
        gradscale = 1 / psi;
    }
    if (psi < 1e-6f)
        return;
    // Pk^T = d^T
    // H = Pk * Dm^-T
    // H^T = Dm^-1 * Pk^T = Dm^-1 * d^T
    mat2 Ht = mat2mul(Dminv, transpose2(d));

    // grad0,1 are columns of H, or rows of H^T
    fpreal3 grad0 = (fpreal3)(Ht.lo * gradscale, 0);
    fpreal3 grad1 = (fpreal3)(Ht.hi * gradscale, 0);
    fpreal3 grad2 = -grad0 - grad1;

    // Rotate back to world space;
    // It's faster to multiply by transpose since it uses row dot products.
    mat3 xformT;
    transpose3(xform, xformT);
    grad0 = mat3Tvecmul(xformT, grad0);
    grad1 = mat3Tvecmul(xformT, grad1);
    grad2 = mat3Tvecmul(xformT, grad2);

    fpreal wsum =   invmass0 * dot(grad0, grad0) +
                    invmass1 * dot(grad1, grad1) +
                    invmass2 * dot(grad2, grad2);
    if (wsum == 0.0)
        return;
    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    // The alpha term should really be 1 / (stiffness * restvolume) since our FEM energy
    // should be integrated over the entire linear element.
    if (flags & NORMSTIFFNESS)
        alpha /= restlength;
    alpha /= timeinc * timeinc;
    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        // We also need to fix up beta term with rest volume, as with alpha.
        if (flags & NORMSTIFFNESS)
            beta *= restlength;
        gamma = alpha * beta / timeinc;

        dsum = dot(grad0, p0 - prev0) +
               dot(grad1, p1 - prev1) +
               dot(grad2, p2 - prev2);
        dsum *= gamma;
        gamma += 1.0f;
    }
    fpreal C =  psi;
    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);

#ifdef JACOBI
    updatedP(dL * invmass0 * grad0, pt0, dP, dPw);
    updatedP(dL * invmass1 * grad1, pt1, dP, dPw);
    updatedP(dL * invmass2 * grad2, pt2, dP, dPw);
#else
    p0 += dL * invmass0 * grad0;
    p1 += dL * invmass1 * grad1;
    p2 += dL * invmass2 * grad2;
    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    L[idx * 3 + loff] += dL;
#endif
}

static void
tetARAPUpdateXPBD(float timeinc,
                int idx,
                int ptidx,
                global const int *pts,
                global fpreal *L,
                global fpreal *P,
                global const fpreal *pprev,
                DPPARMIN
                global const float *masses,
                global const int *stopped,
                float restlength,
                global float *restvector,
                global const float *restmatrix,
                float kstiff,
                float kdampratio,
                uint flags)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    int pt3 = pts[ptidx + 3];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    fpreal3 p3 = vload3(pt3, P);

    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);
    float invmass3 = invMass(masses, stopped, pt3);

    mat3 F, Ds, Dminv;
    // Dm^-1 is stored in restmatrix.
    mat3load(idx, restmatrix, Dminv);
    // Ds = | p0-p3 p1-p3 p2-p3 |
    mat3fromcols(p0 - p3, p1 - p3, p2 - p3, Ds);
    // F = Ds * Dm^-1
    mat3mul(Ds, Dminv, F);

    mat3 R, d;
    // Previous rotation is stored in restvector.
    quat q = vload4f(idx, restvector);
    // We can use low iterations since we ran one pass at 20 in initRotations.
    q = extractRotation(F, q, 3);
    // Store current rotation.
    vstore4f(q, idx, restvector);

    // d = F - R
    // Use transposed version since we switched non-transposed
    // to match VEX.
    qtomat3T(q, R);
    d[0] = F[0] - R[0];
    d[1] = F[1] - R[1];
    d[2] = F[2] - R[2];

    // psi = ||F - R||^2
    fpreal psi = squaredNorm3(d);
    fpreal gradscale = 2;
    // Take the square root for "linear" response and correct gradscale.
    // The "normalized stiffness" constraints (i.e. the correct ones) always
    // use linear response.
    if (flags & (LINEARENERGY | NORMSTIFFNESS))
    {
        psi = sqrt(psi);
        gradscale = 1 / psi;
    }
    if (psi < 1e-6f)
        return;
    // Pk^T = d^T
    // H = Pk * Dm^-T
    // H^T = Dm^-1 * Pk^T = Dm^-1 * d^T
    mat3 dt, Ht;
    transpose3(d, dt);
    mat3mul(Dminv, dt, Ht);

    // grad0,1,2 are columns of H, or rows of H^T
    fpreal3 grad0 = gradscale * Ht[0];
    fpreal3 grad1 = gradscale * Ht[1];
    fpreal3 grad2 = gradscale * Ht[2];
    fpreal3 grad3 = (-grad0 - grad1 - grad2);

    fpreal wsum =   invmass0 * dot(grad0, grad0) +
                    invmass1 * dot(grad1, grad1) +
                    invmass2 * dot(grad2, grad2) +
                    invmass3 * dot(grad3, grad3);
    if (wsum == 0.0)
        return;
    // XPBD term
    fpreal alpha = 1.0f / kstiff;
    // The alpha term should really be 1 / (stiffness * restvolume) since our FEM energy
    // should be integrated over the entire linear element.
    if (flags & NORMSTIFFNESS)
        alpha /= restlength;
    alpha /= timeinc * timeinc;
    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal3 prev3 = vload3(pt3, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        // We also need to fix up beta term with rest volume, as with alpha.
        if (flags & NORMSTIFFNESS)
            beta *= restlength;
        gamma = alpha * beta / timeinc;

        dsum = dot(grad0, p0 - prev0) +
               dot(grad1, p1 - prev1) +
               dot(grad2, p2 - prev2) +
               dot(grad3, p3 - prev3);
        dsum *= gamma;
        gamma += 1.0f;
    }
    fpreal C =  psi;
    // Change in Lagrange multiplier.
    fpreal dL = (-C  - alpha * L[idx * 3] - dsum) / (gamma * wsum + alpha);
#ifdef JACOBI
    updatedP(dL * invmass0 * grad0, pt0, dP, dPw);
    updatedP(dL * invmass1 * grad1, pt1, dP, dPw);
    updatedP(dL * invmass2 * grad2, pt2, dP, dPw);
    updatedP(dL * invmass3 * grad3, pt3, dP, dPw);
#else
    p0 += dL * invmass0 * grad0;
    p1 += dL * invmass1 * grad1;
    p2 += dL * invmass2 * grad2;
    p3 += dL * invmass3 * grad3;
    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    vstore3(p3, pt3, P);
    L[idx * 3] += dL;
#endif
}

static int
tetARAPCoupledUpdateXPBD(float timeinc,
                int idx,
                int ptidx,
                global const int *pts,
                global fpreal *L,
                global fpreal *P,
                global const fpreal *pprev,
                DPPARMIN
                global const float *masses,
                global const int *stopped,
                float restlength,
                global float *restvector,
                global const float *restmatrix,
                float skstiff,
                float vkstiff,
                float kdampratio,
                uint flags)
{
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    int pt3 = pts[ptidx + 3];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    fpreal3 p3 = vload3(pt3, P);

    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);
    float invmass3 = invMass(masses, stopped, pt3);

    mat3 F, Ds, Dminv;
    // Dm^-1 is stored in restmatrix.
    mat3load(idx, restmatrix, Dminv);
    // Ds = | p0-p3 p1-p3 p2-p3 |
    mat3fromcols(p0 - p3, p1 - p3, p2 - p3, Ds);
    // F = Ds * Dm^-1
    mat3mul(Ds, Dminv, F);

    mat3 R, d;
    // Previous rotation is stored in restvector.
    quat q = vload4f(idx, restvector);
    // We can use low iterations since we ran one pass at 20 in initRotations.
    q = extractRotation(F, q, 3);
    // Store current rotation.
    vstore4f(q, idx, restvector);

    // d = F - R
    // Use transposed version since we switched non-transposed
    // to match VEX.
    qtomat3T(q, R);
    d[0] = F[0] - R[0];
    d[1] = F[1] - R[1];
    d[2] = F[2] - R[2];
    // psi = ||F - R||^2
    fpreal psi = squaredNorm3(d);
    fpreal gradscale = 2;
    // Take the square root for "linear" response and correct gradscale.
    // The "normalized stiffness" constraints (i.e. the correct ones) always
    // use linear response.
    if (flags & (LINEARENERGY | NORMSTIFFNESS))
    {
        psi = sqrt(psi);
        gradscale = 1 / psi;
    }
    if (psi < 1e-6f)
        return 0;
    fpreal sC =  psi;
    // Pk^T = d^T
    // H = Pk * Dm^-T
    // H^T = Dm^-1 * Pk^T = Dm^-1 * d^T
    mat3 dt, Ht;
    transpose3(d, dt);
    mat3mul(Dminv, dt, Ht);

    // grad0,1,2 are columns of H, or rows of H^T
    fpreal3 sgrad0 = gradscale * Ht[0];
    fpreal3 sgrad1 = gradscale * Ht[1];
    fpreal3 sgrad2 = gradscale * Ht[2];
    fpreal3 sgrad3 = (-sgrad0 - sgrad1 - sgrad2);
    fpreal ssum =   invmass0 * dot(sgrad0, sgrad0) +
                    invmass1 * dot(sgrad1, sgrad1) +
                    invmass2 * dot(sgrad2, sgrad2) +
                    invmass3 * dot(sgrad3, sgrad3);
    if (ssum == 0.0)
        return 0;

    // Calculate volume and gradients with material coordinates.
    fpreal3 d1 = p1 - p0;
    fpreal3 d2 = p2 - p0;
    fpreal3 d3 = p3 - p0;
    fpreal3 vgrad1 = cross(d3, d2) / 6;
    fpreal3 vgrad2 = cross(d1, d3) / 6;
    fpreal3 vgrad3 = cross(d2, d1) / 6;
    fpreal3 vgrad0 = -(vgrad1 + vgrad2 + vgrad3);
    fpreal volume = dot(cross(d2, d1), d3) / 6;
    fpreal vC = volume - restlength;

    fpreal vsum =    invmass0 * dot(vgrad0, vgrad0) +
                     invmass1 * dot(vgrad1, vgrad1) +
                     invmass2 * dot(vgrad2, vgrad2) +
                     invmass3 * dot(vgrad3, vgrad3);
    if (vsum == 0)
        return 0;

    // Coupled sum.
    fpreal svsum =    invmass0 * dot(sgrad0, vgrad0) +
                      invmass1 * dot(sgrad1, vgrad1) +
                      invmass2 * dot(sgrad2, vgrad2) +
                      invmass3 * dot(sgrad3, vgrad3);

    // XPBD terms
    fpreal salpha = 1.0f / skstiff;
    fpreal valpha = 1.0f / vkstiff; 
    // The alpha term should really be 1 / (stiffness * restvolume) since our FEM energy
    // should be integrated over the entire linear element.
    // Really so should valpha, but it's almost always effectively zero, and we don't
    // do in tetVolumeUpdate, so we don't here.
    if (flags & NORMSTIFFNESS)
        salpha /= restlength;
    salpha /= timeinc * timeinc;
    valpha /= timeinc * timeinc;
    fpreal sdsum = 0, sgamma = 1;
    fpreal vdsum = 0, vgamma = 1;

    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal3 prev3 = vload3(pt3, pprev);
        fpreal sbeta = skstiff * kdampratio * timeinc * timeinc;
        fpreal vbeta = vkstiff * kdampratio * timeinc * timeinc;
        // We also need to fix up beta term with rest volume, as with alpha.
        if (flags & NORMSTIFFNESS)
            sbeta *= restlength;
        sgamma = salpha * sbeta / timeinc;
        vgamma = valpha * vbeta / timeinc;

        sdsum = dot(sgrad0, p0 - prev0) +
               dot(sgrad1, p1 - prev1) +
               dot(sgrad2, p2 - prev2) +
               dot(sgrad3, p3 - prev3);
        vdsum = dot(vgrad0, p0 - prev0) +
               dot(vgrad1, p1 - prev1) +
               dot(vgrad2, p2 - prev2) +
               dot(vgrad3, p3 - prev3);
        sdsum *= sgamma;
        vdsum *= vgamma;
        sgamma += 1.0f;
        vgamma += 1.0f;
    }

    fpreal sL = L[idx * 3 + 0];
    // Volume L is stored at offset 2.
    fpreal vL = L[idx * 3 + 2];

    // 2x2 symmetric matrix solve for both Lagrange multipliers at once.
    // Faster and uses less registers if written out manually.
    // Shape matrix component
    fpreal Axx = sgamma * ssum + salpha;
    // Volume matrix component
    fpreal Ayy = vgamma * vsum + valpha;
    // Coupled matrix component
    // Axy = Ayx = svsum
    fpreal det = Axx * Ayy - svsum * svsum;
    if (det < 1e-8f)
        return -1;
    det = 1.0f / det;
    // Shape RHS
    fpreal sb = -sC - salpha * sL - sdsum;
    // Volume RHS
    fpreal vb = -vC - valpha * vL - vdsum;
    // First component of solution is shape dL.
    fpreal sdL = det * (Ayy * sb - svsum * vb);
    // Second component of solution is volume dL.
    fpreal vdL = det * (Axx * vb - svsum * sb);

#ifdef JACOBI
    updatedP(invmass0 * (sdL * sgrad0 + vdL * vgrad0), pt0, dP, dPw);
    updatedP(invmass1 * (sdL * sgrad1 + vdL * vgrad1), pt1, dP, dPw);
    updatedP(invmass2 * (sdL * sgrad2 + vdL * vgrad2), pt2, dP, dPw);
    updatedP(invmass3 * (sdL * sgrad3 + vdL * vgrad3), pt3, dP, dPw);
#else
    p0 += invmass0 * (sdL * sgrad0 + vdL * vgrad0);
    p1 += invmass1 * (sdL * sgrad1 + vdL * vgrad1);
    p2 += invmass2 * (sdL * sgrad2 + vdL * vgrad2);
    p3 += invmass3 * (sdL * sgrad3 + vdL * vgrad3);
    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    vstore3(p3, pt3, P);
    L[idx * 3 + 0] += sdL;
    L[idx * 3 + 2] += vdL;
#endif
    return 1;
}

static void
pointPrimUpdateXPBD(float timeinc,
                   int idx,
                   global const int *pts_index,
                   global const int *pts,
                   global fpreal *L,
                   global fpreal *P,
                   global const fpreal *pprev,
                   DPPARMIN
                   global const float *masses,
                   global const int *stopped,
                   float restlength,
                   global const float *restvector,
                   float kstiff,
                   float kdampratio,
                   float kstiffcompress)
{
    int ptidx = pts_index[idx];
    int npts = pts_index[idx + 1] - ptidx;
    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    float invmass0 = invMass(masses, stopped, pt0);
    float invmass1 = invMass(masses, stopped, pt1);
    float invmass2 = invMass(masses, stopped, pt2);

    int pt3, pt4;
    fpreal3 p3 = 0, p4 = 0;
    float invmass3 = 0, invmass4 = 0;
    fpreal w1 = 0, w2 = 0, w3 = 0, w4 = 0;

    fpreal4 rv = vload4f(idx, restvector);
    fpreal u = rv.x;
    fpreal v = rv.y;

    if (npts == 3)
    {
        w1 = 1 - u;
        w2 = u;
    }
    else if (npts == 4)
    {
        pt3 = pts[ptidx + 3];
        p3 = vload3(pt3, P);
        invmass3 = invMass(masses, stopped, pt3);
        // Triangle interpolation weights.
        w1 = (1 - u - v);
        w2 = u;
        w3 = v;
    }
    else // npts == 5
    {
        // Point on quad.
        // Load final point info.
        pt3 = pts[ptidx + 3];
        pt4 = pts[ptidx + 4];
        p3 = vload3(pt3, P);
        p4 = vload3(pt4, P);
        invmass3 = invMass(masses, stopped, pt3);
        invmass4 = invMass(masses, stopped, pt4);
        // Quad interpolation weights.
        fpreal u1 = 1 - u;
        fpreal v1 = 1 - v;
        w1 = (u1 * v1);
        w2 = (u1 * v);
        w3 = (u * v);
        w4 = (u * v1);
    }
    // Vector from barycentric point on primitive to individual point.
    fpreal3 n = w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4 - p0;

    fpreal d = length(n);
    if (d < 1e-6f)
        return;

    // Check if we should use compression stiffness value, which also
    // implies we need to use a separate Lagrange multiplier in L.
    int loff = inCompressBand(d, restlength);
    kstiff = select(kstiff, kstiffcompress, loff);
    if (kstiff == 0.0f)
        return;

    // Constraint calc.
    fpreal C = d - restlength;
    n /= d;
    fpreal3 grad0 = -n;
    fpreal3 grad1 = w1 * n;
    fpreal3 grad2 = w2 * n;
    fpreal3 grad3 = w3 * n;
    fpreal3 grad4 = w4 * n;

    fpreal wsum =    invmass0 * dot(grad0, grad0) +
                     invmass1 * dot(grad1, grad1) +
                     invmass2 * dot(grad2, grad2) +
                     invmass3 * dot(grad3, grad3) +
                     invmass4 * dot(grad3, grad4);

    if (wsum == 0.0)
        return;

    fpreal l = L[idx * 3 + loff];

    // XPBD term.
    fpreal alpha = 1.0f / kstiff;
    alpha /= timeinc * timeinc;

    fpreal dsum = 0, gamma = 1;
    if (kdampratio > 0)
    {
        // Compute damping terms.
        fpreal3 prev0 = vload3(pt0, pprev);
        fpreal3 prev1 = vload3(pt1, pprev);
        fpreal3 prev2 = vload3(pt2, pprev);
        fpreal beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;

        dsum = dot(grad0, p0 - prev0) +
               dot(grad1, p1 - prev1) +
               dot(grad2, p2 - prev2);

        if (npts >= 4)
            dsum += dot(grad3, p3 - vload3(pt3, pprev));
        if (npts == 5)
            dsum += dot(grad4, p4 - vload3(pt4, pprev));
        dsum *= gamma;
        gamma += 1.0f;
    }

    fpreal dL = (-C  - alpha * l - dsum) / (gamma * wsum + alpha);

#ifdef JACOBI
    updatedP(dL * invmass0 * grad0, pt0, dP, dPw);
    updatedP(dL * invmass1 * grad1, pt1, dP, dPw);
    updatedP(dL * invmass2 * grad2, pt2, dP, dPw);
    if (npts >= 4)
        updatedP(dL * invmass3 * grad3, pt3, dP, dPw);
    if (npts >= 5)
        updatedP(dL * invmass4 * grad4, pt4, dP, dPw);
#else
    // Update points.
    p0 += dL * invmass0 * grad0;
    p1 += dL * invmass1 * grad1;
    p2 += dL * invmass2 * grad2;
    p3 += dL * invmass3 * grad3;
    p4 += dL * invmass4 * grad4;

    vstore3(p0, pt0, P);
    vstore3(p1, pt1, P);
    vstore3(p2, pt2, P);
    if (npts >= 4)
        vstore3(p3, pt3, P);
    if (npts == 5)
        vstore3(p4, pt4, P);
    L[idx * 3 + loff] += dL;
#endif
}

// 64-bit accum_t quaternion and matrix types for shape matching.
#if ACCUM_PREC==64
typedef quatd quata;
#define qrotatea( q,  v ) qrotated( q, v )
#define extractRotationa( A,  q,  maxiter ) extractRotationd( A,  q,  maxiter )
typedef mat3d mat3a;
#else
typedef quat quata;
#define qrotatea( q,  v ) qrotate( q, v )
#define extractRotationa( A,  q,  maxiter ) extractRotation( A,  q,  maxiter )
typedef mat3 mat3a;
#endif

static void
shapeMatchPts(  int npts,
                global const int *pts,
                float timeinc,
                global fpreal * P,
                global const fpreal * pprevs,
                global const fpreal * rest_in,
                global const float * mass,
                global const int *stopped,                    
                float kstiff,
                float kdampratio,
                global float * restvector,
                global fpreal * L,
                int reduce
             )
{
    size_t lid = 0;
    size_t lsize = 1;
    // eps is 1 / "infinite" mass used for pinned points to move the center of mass
    // to the average of the pin locations.
    const float eps = 1e-8;
#ifdef __opencl_c_work_group_collective_functions
    if (reduce)
    {
        lid = get_local_id(0);
        lsize = get_local_size(0);
    }
#endif
    
    // Find rest and current centers of mass.
    accum3_t restcmd = 0, cmd = 0;
    accum_t totmass = 0;
    for (size_t idx = lid; idx < npts; idx += lsize)
    {
        int pt = pts[idx];
        float m = 1.0f /  (invMass(mass, stopped, pt) + eps);
        restcmd += toaccum3(m * vload3(pt, rest_in));
        cmd += toaccum3(m * vload3(pt, P));
        totmass += m;
    }
#ifdef __opencl_c_work_group_collective_functions
    if (reduce)
    {
        restcmd = work_group_reduce_add3(restcmd);
        cmd = work_group_reduce_add3(cmd);
        totmass = work_group_reduce_add(totmass);
    }
#endif

    restcmd /= totmass;
    cmd /= totmass;
    fpreal3 restcm = tofpreal3(restcmd);
    fpreal3 cm = tofpreal3(cmd);

    // accum_t version of mat3
    mat3a A;
    A[0] = 0; A[1] = 0; A[2] = 0;
    // Calc Apq matrix.
    for (size_t idx = lid; idx < npts; idx += lsize)
    {
        int pt = pts[idx];
        accum3_t q = toaccum3(vload3(pt, rest_in)) - restcmd;
        accum3_t p = toaccum3(vload3(pt, P)) - cmd;
        float m = 1.0f /  (invMass(mass, stopped, pt) + eps);
        p *= m;
        // outerproduct
        A[0] += p.x * q;
        A[1] += p.y * q;
        A[2] += p.z * q;
    }

#ifdef __opencl_c_work_group_collective_functions
    if (reduce)
    {
        A[0] = work_group_reduce_add3(A[0]);
        A[1] = work_group_reduce_add3(A[1]);
        A[2] = work_group_reduce_add3(A[2]);
    }        
#endif

    // Find nearest rotation.
    float4 Rf = vload4(0, restvector);
    // Also use accum_t version of quat.
    quata R = (quata)(Rf.x, Rf.y, Rf.z, Rf.w);
    R = extractRotationa(A, R, 10);

    accum_t wsum = 0, dsum = 0, C = 0;
    // Compute C and damping term if needed.
    for (size_t idx = lid; idx < npts; idx += lsize)
    {
        int pt = pts[idx];
        accum3_t pos = toaccum3(vload3(pt, P));
        accum3_t q = toaccum3(vload3(pt, rest_in)) - restcmd;
        accum3_t goal = cmd + qrotatea(R, q);
        accum3_t d =  goal - pos;
        C += dot(d, d);
        accum3_t grad = -2 * d;
        float invmass = invMass(mass, stopped, pt);
        wsum += invmass * dot(grad, grad); 
        // Compute damping term.
        if (kdampratio > 0)
            dsum += dot(grad, pos - toaccum3(vload3(pt, pprevs)));
    }

#ifdef __opencl_c_work_group_collective_functions
    if (reduce)
    {
        wsum = work_group_reduce_add(wsum);
        dsum = work_group_reduce_add(dsum);
        C = work_group_reduce_add(C);
    }        
#endif

    accum_t gamma = 1;

    accum_t l = *L;    
    accum_t alpha = 1.0 / kstiff;
    alpha /= timeinc * timeinc;
    // More damping terms.
    if (kdampratio > 0)
    {
        accum_t beta = kstiff * kdampratio * timeinc * timeinc;
        gamma = alpha * beta / timeinc;
        dsum *= gamma;
        gamma += 1.0f;
    }

    accum_t dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha);

    // Apply per-point correction
    for (size_t idx = lid; idx < npts; idx += lsize)
    {
        int pt = pts[idx];
        accum3_t pos = toaccum3(vload3(pt, P));
        accum3_t q = toaccum3(vload3(pt, rest_in)) - restcmd;
        accum3_t goal = cmd + qrotatea(R, q);
        accum3_t d =  goal - pos;
        accum3_t grad = -2 * d;
        float invmass = invMass(mass, stopped, pt);
        pos += invmass * dL * grad; 
        vstore3(tofpreal3(pos), pt, P);        
    }
    if (lid == 0)
    {
        *L += dL;
        vstore4((float4)(R.x, R.y, R.z, R.w), 0, restvector);
    }
}

static void
shapeMatchUpdateXPBD(float timeinc,
                     int idx,
                     global const int * pts_index, 
                     global const int *pts,
                     global fpreal *L,
                     global fpreal *P,
                     global const fpreal *pprev,
                     global const fpreal *rest,
                     global const float *mass,
                     global const int *stopped,
                     global float *restvector,
                     float kstiff,
                     float kdampratio)

{
    int ptidx = pts_index[idx];
    int npts = pts_index[idx + 1] - ptidx;
    // We can only use device-side enqueue when *not* using SINGLE_WORKGROUP,
    // since we can't assume we'll be done writing to the points by the next
    // constraint coloring update (in fact we likely won't be.)
    // We also only have enough precision in our accumulator to ensure order
    // independence if it's 64-bit and fpreal is 32-bit.
#if defined(__opencl_c_work_group_collective_functions) && defined(__opencl_c_device_enqueue) && \
    !defined(SINGLE_WORKGROUP) && !defined(SINGLE_WORKGROUP_SPANS) && \
    ACCUM_PREC==64 && FPREAL_PREC==32

    // Arbitrary cutoff, seems reasonable on RTX 2080 Super.
    int lsize = 64;
    if (npts > 50)
    {
        // We need to check this enqueue fails, which it easily can past 100 shape matched
        // objects or so due to the small on-device queue.  If so, fall through and process
        // the constraint in this work item as if it had a small number of points.
        // NOTE: the summations in shapeMatchPts have to be order independent for this
        // approach to remain deterministic, since we can't necessarily predict the order
        // of the enqueue failures as its dependent on how fast the GPU queue empties.
        if (enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                   ndrange_1D(lsize, lsize),
                   ^{
                        shapeMatchPts(npts, &pts[ptidx], timeinc,
                                      P, pprev, rest, mass, stopped,
                                      kstiff, kdampratio, 
                                      &restvector[idx * 4], &L[idx * 3], 1);
                    }) == CLK_SUCCESS)
        return;
    }        
#endif
    shapeMatchPts(npts, &pts[ptidx], timeinc,
                  P, pprev, rest, mass, stopped,
                  kstiff, kdampratio, 
                  &restvector[idx * 4], &L[idx * 3], 0);    
}


// Project p onto the line defined by orig + dir.
// Dir must be normalized.
static fpreal3
projectToLine(fpreal3 p, fpreal3 orig, fpreal3 dir)
{
    return orig + dir * dot(p - orig, dir);
}

// Project p to dist above the plane defined by orig + dir.
// Dir must be normalized.
static fpreal3
projectAbovePlane(fpreal3 p, fpreal3 orig, fpreal3 dir, fpreal dist)
{
    return p + dir * (dist - dot(p - orig, dir));
}

#ifndef SINGLE_WORKGROUP
#if defined(__H_GPU__) && defined(__H_AMD__)
// More efficient max workgroup size on AMD hardware.
__attribute__((reqd_work_group_size(64, 1, 1)))
#endif
#endif
kernel void
constraintUpdate(
#ifdef SINGLE_WORKGROUP
#ifdef SINGLE_WORKGROUP_SPANS
		 int startcolor,
#endif
                 int ncolors,
                 global const int *color_offsets,
                 global const int *color_lengths,
#else
                 int color_offset,
                 int color_length,
#endif
                 float timeinc,
                 int type_length,
                 global const int *type,
                 int pts_length,
                 global const int *pts_index,
                 global const int *pts,
                 int restlen_length,
                 global const float *restlengths,
                 int stiffness_length,
                 global const float *stiffness,
                 int dampingratio_length,
                 global const float *dampingratio,
#ifdef HAS_compressstiffness
                 int compressstiffness_length,
                 global float * compressstiffnesses,
#endif
                 int L_length,
                 global fpreal *L,
#ifdef HAS_restvector
                 int restvector_length,
                 global float * restvector ,
#endif
#ifdef HAS_restdir
                 int restdir_length,
                 global const float * restdir,
#endif
#ifdef HAS_restmatrix
                 int restmatrix_length,
                 global const float * restmatrix,
#endif
#ifdef HAS_rest
                 int rest_length,
                 global fpreal *rest,
#endif                 
                 int P_length,
                 global fpreal *P,
                 int pprev_length,
                 global const fpreal *pprev,
#ifdef HAS_dP
                 int dP_length,
                 global fpreal * dP ,
#endif
#ifdef HAS_dPw
                 int dPw_length,
                 global fpreal * dPw ,
#endif
                 int mass_length,
                 global const float *mass
#ifdef HAS_stopped
                 , int stopped_length,
                 global const int *stopped_in
#endif
#ifdef HAS_orient
                 , int orient_length,
                 global fpreal * orient
#endif
#ifdef HAS_orientprevious
                 , int orientprevious_length,
                 global const fpreal * orientprevious
#endif
#ifdef HAS_inertia
                 , int inertia_length,
                 global const float * inertia
#endif
#ifdef HAS_pressuregradient
                 , int pressuregradient_length,
                 global float * pressuregradient
#endif
#ifdef HAS_volume
                 , int volume_length,
                 global float * volume
#endif
#ifdef HAS_allstopped
                 , int allstopped_length,
                 global int *allstopped
#endif
                 )
{
    // All of our constraints are XPBD-based, with their updates a function
    // of timeinc.  If zero, we can't/won't do anything.
    if (timeinc == 0.0f)
        return;
#ifdef SINGLE_WORKGROUP
#define SKIPWORKITEM continue
#ifdef SINGLE_WORKGROUP_SPANS
   for(int i = startcolor; i < startcolor+ncolors; i++)
#else
   for(int i = 0; i < ncolors; i++)
#endif
    {
        int color_length = color_lengths[i];
        int color_offset = color_offsets[i];
        if (i > 0)
            barrier(CLK_GLOBAL_MEM_FENCE);
#ifdef SINGLE_WORKGROUP_ALWAYS
	// Our SKIPWORKITEM is a continue, so we
	// nee dto do our update in the start.
	color_offset -= get_global_size(0);
	color_length += get_global_size(0);
	while (1)
	{
	    color_offset += get_global_size(0);
	    color_length -= get_global_size(0);
	    if (color_length <= 0)
		break;
#endif

#else
#define SKIPWORKITEM return
    {
#endif

    int idx = get_global_id(0);
    if (idx >= color_length)
        SKIPWORKITEM;
    idx += color_offset;
#ifdef HAS_allstopped
    if (allstopped[idx])
        SKIPWORKITEM;
#endif
    int ctype = type[idx];

    int ptidx = pts_index[idx];
    float restlen = restlengths[idx];
    float kstiff = stiffness[idx];
    float kdampratio = dampingratio[idx];
    // Optional compression stiffness values.
#ifdef HAS_compressstiffness
    float kstiffcompress = compressstiffnesses[idx];
    // Compression Stiffness defaults to -1, which means use regular stiffness.
    kstiffcompress = select(kstiff, kstiffcompress, kstiffcompress >= 0.0f);
#else
    float kstiffcompress = kstiff;
#endif
    if (kstiff == 0.0f && kstiffcompress == 0.0f)
        SKIPWORKITEM;

#ifdef HAS_stopped
    global const int *stopped = stopped_in;
#else
    global const int *stopped = 0;
#endif

// Constraint dispatch.
#if defined(CONSTRAINT_distance)
    if (ctype == DISTANCE)
        distanceUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, DPPARM mass, stopped, restlen, kstiff, kdampratio, kstiffcompress);
#endif


#if defined(CONSTRAINT_pin) && defined(HAS_restvector)
    if (ctype == PIN)
        distancePosUpdateXPBD(timeinc, idx, pts[ptidx], vload4f(idx, restvector).xyz, vload3(pts[ptidx], P), L, P, pprev, DPPARM mass, stopped, restlen, kstiff, kdampratio, kstiffcompress);
#endif


#if defined(CONSTRAINT_distanceline) && defined(HAS_restdir)
    if (ctype == DISTANCELINE)
    {
        int pt1 = pts[ptidx];
        fpreal3 p1 = vload3(pt1, P);
        // Project the current position onto the line and treat as a distance constraint to that position.
        fpreal3 p0 = projectToLine(p1, vload4f(idx, restvector).xyz, vload3f(idx, restdir));
        distancePosUpdateXPBD(timeinc, idx, pt1, p0, p1, L, P, pprev, DPPARM mass, stopped, restlen, kstiff, kdampratio, kstiffcompress);
    }
#endif


#if defined(CONSTRAINT_distanceplane) && defined(HAS_restdir)
    if (ctype == DISTANCEPLANE)
    {
        int pt1 = pts[ptidx];
        fpreal3 p1 = vload3(pt1, P);
        fpreal3 dir = vload3f(idx, restdir);
        // Project the current position to a distance of restlength above the plane, then treat as a distance constraint
        // to that position with a restlength of zero.  Override the stiffness with compressStiffness if below the plane,
        // giving "sidedness" for the plane constraint, rather than just a spring to the projected point.
        fpreal3 p0 = projectAbovePlane(p1, vload4f(idx, restvector).xyz, dir, restlen);
        kstiff = select(kstiff, kstiffcompress, inCompressBand(dot(p1 - p0, dir), 0));
        distancePosUpdateXPBD(timeinc, idx, pt1, p0, p1, L, P, pprev, DPPARM mass, stopped, 0, kstiff, kdampratio, kstiffcompress);
    }
#endif


#if defined(CONSTRAINT_triarap) && defined(HAS_restvector)
    if (isTriARAP(ctype))
        triARAPUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, DPPARM mass, stopped, restlen, restvector, kstiff, kdampratio, kstiffcompress, FEMFlags(ctype));
#endif


#if defined(CONSTRAINT_tetarap) && defined(HAS_restvector) && defined(HAS_restmatrix)
    const float volstiff = 1e35f;
    if (isTetARAP(ctype))
    {
        int coupledsolve = -1;
        if (isTetARAPVol(ctype))
            coupledsolve = tetARAPCoupledUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, DPPARM mass, stopped, restlen, restvector, restmatrix, kstiff, volstiff, kdampratio, FEMFlags(ctype));
        // Result < 0 means coupled linear solve failed (or not isTetARAPVol), try again with uncoupled.
        if (coupledsolve < 0)
        {
            tetARAPUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, DPPARM mass, stopped, restlen, restvector, restmatrix, kstiff, kdampratio, FEMFlags(ctype));
            if (isTetARAPVol(ctype))
                tetVolumeUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, DPPARM mass, stopped, restlen, volstiff, kdampratio, volstiff, 2);
        }
    }
#endif


#if defined(CONSTRAINT_triarea)
    if (ctype == TRIAREA)
        triAreaUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, DPPARM mass, stopped, restlen, kstiff, kdampratio, kstiffcompress);
#endif


#if defined(CONSTRAINT_tetvolume)
    if (ctype == TETVOLUME)
        tetVolumeUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, DPPARM mass, stopped, restlen, kstiff, kdampratio, kstiffcompress, 0);
#endif


#if defined(CONSTRAINT_ptprim) && defined(HAS_restvector)
    if (ctype == PTPRIM)
        pointPrimUpdateXPBD(timeinc, idx, pts_index, pts, L, P, pprev, DPPARM mass, stopped, restlen, restvector, kstiff, kdampratio, kstiffcompress);
#endif

// The following constraint types do NOT support Jacobi updates.
// WARNING: if you add Jacobi smoothing to a constraint type, you MUST remove
// it from the list of types without smoothing in the hasSmoothing function at
// the bottom of $SHS/vex/include/pbd_constraints.h.

#ifndef JACOBI

#if defined(CONSTRAINT_bend)
    if (ctype == BEND)
        dihedralUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, mass, stopped, restlen, kstiff, kdampratio);
#endif


#if defined(CONSTRAINT_trianglebend)
    if (ctype == TRIANGLEBEND)
        triangleBendUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, mass, stopped, restlen, kstiff, kdampratio);
#endif


#if defined(CONSTRAINT_angle)
    if (ctype == ANGLE)
        angleUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, mass, stopped, restlen, kstiff, kdampratio);
#endif


#if defined(CONSTRAINT_tetfiber) && defined(HAS_restvector)
    global const float *rm = 0;
#if defined(HAS_restmatrix)
    rm = restmatrix;
#endif
    if (isTetFiber(ctype))
        tetFiberUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, mass, stopped, restlen, restvector, rm, kstiff, kdampratio, FEMFlags(ctype));
#endif


#if defined(CONSTRAINT_stretchshear) && defined(HAS_orient) && defined(HAS_orientprevious) && defined(HAS_inertia)
   if (ctype == STRETCHSHEAR)
        stretchShearUpdateXPBD(timeinc, idx, ptidx, pts, L, P, pprev, mass, stopped, orient, orientprevious, inertia, restlen, kstiff, kdampratio);
#endif


#if defined(CONSTRAINT_bendtwist) && defined(HAS_orient) && defined(HAS_orientprevious) && defined(HAS_inertia)
    if (ctype == BENDTWIST)
        bendTwistUpdateXPBD(timeinc, idx, ptidx, pts, L, orient, orientprevious, inertia, stopped, restvector, kstiff, kdampratio);
#endif


#if defined(CONSTRAINT_pinorient) && defined(HAS_orient) && defined(HAS_orientprevious) && defined(HAS_inertia)
    if (ctype == PINORIENT)
        bendTwistOrientUpdateXPBD(timeinc, idx, ptidx, pts, L, orient, orientprevious, inertia, stopped, restvector, kstiff, kdampratio);
#endif


#if defined(CONSTRAINT_pressure) && defined(HAS_pressuregradient) && defined(HAS_volume)
    if (ctype == PRESSURE)
        pressureUpdateXPBD(timeinc, idx, pts_index, pts, L, P, pprev, mass, stopped, restlen,
                           kstiff, kdampratio, kstiffcompress, pressuregradient, volume);
#endif

#if defined(CONSTRAINT_shapematch) && defined(HAS_rest)
    if (ctype == SHAPEMATCH)
        shapeMatchUpdateXPBD(timeinc, idx, pts_index, pts, L, P, pprev, rest, mass, stopped, restvector, kstiff, kdampratio);
#endif

#endif // JACOBI
#ifdef SINGLE_WORKGROUP
#ifdef SINGLE_WORKGROUP_ALWAYS
    }
#endif
#endif
    }
}
