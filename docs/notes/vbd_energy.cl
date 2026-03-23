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
 * NAME:    vbd_energy.cl ( CE Library, OpenCL)
 *
 * COMMENTS:
 *    VBD energy for deformable materials.
 */

// Just for compilation testing.
#ifdef TEST_VBD_COMPILE
#define HAS_fallback
#define HAS_fiberstiffness
#define HAS_fiberscale
#define HAS_restvector
#define HAS_restmatrix
#define HAS_hittypes
#define HAS_pnext
// #define HAS_collisionstiffnessscale
// #define USE_DOUBLE
#endif

#ifdef USE_DOUBLE
// We want to be able to define the USE_DOUBLE flag to run
// a double-precision solve on single-precision inputs.
// But OpenCL SOP/DOPs always #define fpreal*, which
// makes defining USE_DOUBLE a no-op, so we undef them all
// here so they get re-defined as we want in typedefines.h.
#undef fpreal
#undef fpreal2
#undef fpreal3
#undef fpreal4
#undef fpreal8
#undef fpreal16
#undef FPREAL_PREC
#endif

#include <platform.h>
#include <typedefines.h>
#include <matrix.h>
#include <quaternion.h>
#include <svd3.h>
#include <pbd_types.h>
#include <reduce.h>

static void
zeros9(fpreal *H)
{
    for (int i=0; i < 81; i++)
        H[i] = 0.0f;
}

static void
eye9(fpreal *H, const fpreal scale)
{
    for (int i=0; i < 81; i++)
        H[i] = (i % 10 == 0) ? scale : 0;
}

static void
outer9(fpreal *H, const fpreal *t, const fpreal scale)
{
    for (int i=0; i < 9; i++)
        for (int j=0; j < 9; j++)
            H[i * 9 + j] = scale * t[i] * t[j];
}

static void
addouter9(fpreal *H, const fpreal *t, const fpreal scale)
{
    for (int i=0; i < 9; i++)
        for (int j=0; j < 9; j++)
            H[i * 9 + j] += scale * t[i] * t[j];
}

static void
add9(fpreal *H, const fpreal *J)
{
    for (int i=0; i < 81; i++)
        H[i] += J[i];
}

// Column-wise flattening of T to 9 fpreals in *t.
static void
mat3tovec9(const mat3 T, fpreal *t)
{
    t[0] = T[0].x;
    t[1] = T[1].x;
    t[2] = T[2].x;

    t[3] = T[0].y;
    t[4] = T[1].y;
    t[5] = T[2].y;

    t[6] = T[0].z;
    t[7] = T[1].z;
    t[8] = T[2].z;
}

static void
computeMuLambda(const fpreal E, const fpreal v, fpreal *mu, fpreal *lambda)
{
    *mu = E / (2 * (1 + v));
    *lambda = E * v / ((1 + v) * (1 - 2 * v));
}

static void
partialJpartialF(const mat3 F, mat3 pJpF)
{
    // Eqn. 19 from Section 4.2 in "Stable Neo-Hookean Flesh Simulation"
    mat3 Ft;
    transpose3(F, Ft);
    fpreal3 F12 = cross(Ft[1], Ft[2]);
    fpreal3 F20 = cross(Ft[2], Ft[0]);
    fpreal3 F01 = cross(Ft[0], Ft[1]);
    mat3fromcols(F12, F20, F01, pJpF);
}

// Eqn. 29 from Section 4.5 in "Stable Neo-Hookean Flesh Simulation"
// ifdef scale to a double where possible to avoid losing precision
// when calling from the SNH Hessian code.
static void
crossProduct(const mat3 F,
             int col,
#if defined(cl_khr_fp64) && !defined(NO_DOUBLE_SUPPORT)
             double scale,
#else
             float scale,
#endif                          
             fpreal fhat[])
{
    if (col == 0)
    {
        fhat[0 + 0 * 3] = 0;
        fhat[0 + 1 * 3] = scale * -F[2].s0;
        fhat[0 + 2 * 3] = scale * F[1].s0;
        fhat[1 + 0 * 3] = scale * F[2].s0;
        fhat[1 + 1 * 3] = 0;
        fhat[1 + 2 * 3] = scale * -F[0].s0;
        fhat[2 + 0 * 3] = scale * -F[1].s0;
        fhat[2 + 1 * 3] = scale * F[0].s0;
        fhat[2 + 2 * 3] = 0;
    }
    if (col == 1)
    {
        fhat[0 + 0 * 3] = 0;
        fhat[0 + 1 * 3] = scale * -F[2].s1;
        fhat[0 + 2 * 3] = scale * F[1].s1;
        fhat[1 + 0 * 3] = scale * F[2].s1;
        fhat[1 + 1 * 3] = 0;
        fhat[1 + 2 * 3] = scale * -F[0].s1;
        fhat[2 + 0 * 3] = scale * -F[1].s1;
        fhat[2 + 1 * 3] = scale * F[0].s1;
        fhat[2 + 2 * 3] = 0;
    }
    if (col == 2)
    {
        fhat[0 + 0 * 3] = 0;
        fhat[0 + 1 * 3] = scale * -F[2].s2;
        fhat[0 + 2 * 3] = scale * F[1].s2;
        fhat[1 + 0 * 3] = scale * F[2].s2;
        fhat[1 + 1 * 3] = 0;
        fhat[1 + 2 * 3] = scale * -F[0].s2;
        fhat[2 + 0 * 3] = scale * -F[1].s2;
        fhat[2 + 1 * 3] = scale * F[0].s2;
        fhat[2 + 2 * 3] = 0;
    }
}

static void
SNH_PK1Hessian(const mat3 F, fpreal mu, fpreal lambda, mat3 PK1, fpreal H[])
{
    // Reparamaterizing to be consistent with linear, see Section 3.4 
    // of "Stable Neo-Hookean Flesh Simulation", end of first paragraph.
    fpreal det = det3(F);
    mat3 pJpF;
    partialJpartialF(F, pJpF);
    // Use double precision for scaling factors if possible,
    // else at high volume stiffness/shape stiffness ratios they
    // won't cancel out even when F==I.
#if defined(cl_khr_fp64) && !defined(NO_DOUBLE_SUPPORT)
    // Reparameterizing to be consistent with linear, see Section 3.4 
    // of "Stable Neo-Hookean Flesh Simulation", end of first paragraph.
    double lambdamu = (double)lambda + (double)mu;
    double alpha = 1.0 + mu / lambdamu;
    double Jminus1 = det3(F) - alpha;
    // PK1 = mu * F + lambda * Jminus1 * pJpF
    double scale = lambdamu * Jminus1;
#else        
    float lambdamu = lambda + mu;
    float alpha = 1.0f + mu / lambdamu;
    float Jminus1 = det3(F) - alpha;
    // PK1 = mu * F + lambda * Jminus1 * pJpF
    float scale = lambdamu * Jminus1;
#endif
    mat3lincomb2(PK1, F, mu, pJpF, scale);

    // Eqn. 29 from Section 4.5 in "Stable Neo-Hookean Flesh Simulation"
    // f0hat = crossProduct(F, 0) * scale
    // f1hat = crossProduct(F, 1) * scale
    // f2hat = crossProduct(F, 2) * scale
    fpreal f0hat[9], f1hat[9], f2hat[9];
    crossProduct(F, 0, scale, f0hat);
    crossProduct(F, 1, scale, f1hat);
    crossProduct(F, 2, scale, f2hat);

    // mu * np.eye(9) + lambda * np.outer(pjpf, pjpf) + hessJ
    eye9(H, mu);
    fpreal pjpf[9];
    mat3tovec9(pJpF, pjpf);
    addouter9(H, pjpf, lambdamu);
    // add the fractal cross-product 
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    {
            // mat3tovec9 expands column-first,
            // so compute fidx that way.
            int fidx = i + j * 3;
            H[i * 9 + j + 3]        += -f2hat[fidx];
            H[(i + 3) * 9 + j]      +=  f2hat[fidx];

            H[i * 9 + j + 6]        +=  f1hat[fidx];
            H[(i + 6) * 9 + j]      += -f1hat[fidx];

            H[(i + 3) * 9 + j + 6]  += -f0hat[fidx];
            H[(i + 6) * 9 + j + 3]  +=  f0hat[fidx];
    }
}

// Calc the I4 invariant for determing inversion
// along the w direction.
static fpreal
calcI4(const mat3 F, const fpreal3 w)
{
    // Check for any inversion at all before expensive check for
    // inversion in the w direction.
    fpreal I4 = 1;
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
        fpreal3 Sw = mat3vecmul(S, w);
        I4 = dot(w, Sw);
    }
    return I4;
}

// Project to a Symmetric Positive Definite using absolute
// eigenvalue filtering.  For justification see:
// "Stabler Neo-Hookean Simulation:
// "Absolute Eigenvalue Filtering for Projected Newton"
static void
projectToSPD(mat3 H)
{
    // Enforce symmetry possibly lost to floating point.
    mat3makesym(H);
    // Nothing to do if already SPD.
    if (mat3isPD(H))
        return;
    mat3 L, LH;
    // H = L * abs(eigh(H)) * L.T
    // eigen_analysis returns L.T, so use qtomat3T to get L.
    qtomat3T(svd3_jacobi_eigen_analysis(H), L);
    // Abs and clamp below to ensure no (near-)zero eigenvalues.
    const fpreal3 eps = 1e-6;
    fpreal3 eigs = max(fabs(diag3(H)), eps);
    mat3diag(eigs, H);
    mat3mul(L, H, LH);
    mat3mulT(LH, L, H);
    // Re-enforce symmetry possibly lost to floating point.
    mat3makesym(H);
}

// Fiber PK1 and Hessian that add to the output matrices, i.e.
// PK1 and H are in/out parameters.
static void
Fiber_PK1HessianAdd(const mat3 F, fpreal3 w, fpreal fiberscale, fpreal mu, 
                    mat3 PK1, fpreal H[])
{
    // Anisotropic ARAP which is the same as our fiber constraint
    // when fiberscale is zero.
    fpreal lenw = length(w);
    if (lenw < 1e-9)
        return;
    w /= lenw;
    fpreal3 FwT = mat3Tvecmul(F, w);
#if defined(cl_khr_fp64) && !defined(NO_DOUBLE_SUPPORT)
    const double I5 = dot(FwT, FwT);
    if (I5 < 1e-9)
        return;
    const double sqrtI5inv = native_rsqrt(I5);
    double I4 = calcI4(F, w);
    int SI4 = I4 > 0 ? 1 : -1;
    const double dPsidI5 = mu * (1.0  - SI4 * fiberscale * sqrtI5inv);
#else
    const fpreal I5 = dot(FwT, FwT);
    if (I5 < 1e-9)
        return;
    const fpreal sqrtI5inv = native_rsqrt(I5);
    fpreal I4 = calcI4(F, w);
    int SI4 = I4 > 0 ? 1 : -1;
    const fpreal dPsidI5 = mu * (1.0f  - SI4 * fiberscale * sqrtI5inv);
#endif
    mat3 A, FA;
    // A = w^T w
    outerprod3(w, w, A);
    // FA = F @ A
    mat3mul(F, A, FA);
    // dPsi/dI5
    // If we're not near the singularity, just return the generic
    // formula, else use the reflected value at 2.
    if (fabs(I4) > 1e-4)
        mat3lincomb2(PK1, PK1, 1.0f, FA, dPsidI5);
    else
        mat3lincomb2(PK1, PK1, 1.0f, FA, mu * (1.0 - fiberscale / 2.0));

    // Anisotropic Hessian from DD, which is
    // dPsi/dI5 * H_IV + d^2psi/d^2I5 * g_IV * g_IV^T
    // where H_IV = kron(A, I) and g_IV = vec(FA).
    // First add in scaled Kronecker product of A x I
    // equivalent to H += dPsidI5 * np.kron(A, np.identity(3))
    fpreal v[9];
    mat3tovec9(A, v);
    for (int y = 0; y < 3; y++)
    for (int x = 0; x < 3; x++)
    {
        const int x3 = 3 * x;
        const int y3 = 3 * y;
        // mat3tovec9 expands column-first,
        // so compute aidx that way.
        const int aidx = x + (y * 3);
        for (int i = 0; i < 3; i++)
            H[(x3 + i) * 9 + y3 + i] += dPsidI5 * v[aidx];
    }

    // Add second part of Hessian sum if fiberscale is present:
    if (fiberscale > 0.0f)
    {
        // d^2psi/d^2I5 * g_IV * g_IV^T
        mat3tovec9(FA, v);
        addouter9(H, v, mu * SI4 * fiberscale * sqrtI5inv * sqrtI5inv * sqrtI5inv);
    }
}

static void
Fiber_PK1Hessian(const mat3 F, fpreal3 w, fpreal fiberscale, fpreal mu,
                 mat3 PK1, fpreal H[])
{
    mat3zero(PK1);
    zeros9(H);
    Fiber_PK1HessianAdd(F, w, mu, fiberscale, PK1, H);
}

static int
tetUpdateVBD(const int type, const int ptidx, const int prim, const int vtxorder,
                global const int *primpts, global const float *P, global const float *pprev, 
                const fpreal restlength, global const float *restvector,
                global const float *restmatrix,
                const fpreal mu, const fpreal lambda,
                const fpreal fiberscale, const fpreal kfiber, const fpreal kdamp,
                fpreal3 *f, mat3 H)
{
    fpreal3 p0 = vload3f(primpts[ptidx + 0], P);
    fpreal3 p1 = vload3f(primpts[ptidx + 1], P);
    fpreal3 p2 = vload3f(primpts[ptidx + 2], P);
    fpreal3 p3 = vload3f(primpts[ptidx + 3], P);

    mat3 F, Ds, Dminv, PK1;
    fpreal dPdF[81];
    // Rest volume is stored in restlength.
    fpreal V = restlength;
    // Dm^-1 is stored in restmatrix.
    mat3load(prim, restmatrix, Dminv);
    // Ds = | p0-p3 p1-p3 p2-p3 |
    mat3fromcols(p0 - p3, p1 - p3, p2 - p3, Ds);
    // F = Ds * Dm^-1
    mat3mul(Ds, Dminv, F);
    // Guard against malformed Dminv input.
    if (squaredNorm3(F) < 1e-9)
        return 0;
    // Calc SNH energy PK1 (dPsiDF) and hessian (d2Psi/d2F).
    if (type == TETARAPNORMVOL)
    {
        SNH_PK1Hessian(F, mu, lambda, PK1, dPdF);
        if (kfiber > 0)
            Fiber_PK1HessianAdd(F, vload4f(prim, restvector).xyz, fiberscale, kfiber, PK1, dPdF);
    }
    if (type == TETFIBERNORM)
        Fiber_PK1Hessian(F, vload4f(prim, restvector).xyz, fiberscale, mu, PK1, dPdF);

    fpreal3 Dmrows[4];
    Dmrows[0] = Dminv[0];
    Dmrows[1] = Dminv[1];
    Dmrows[2] = Dminv[2];
    Dmrows[3] = -Dminv[0] - Dminv[1] - Dminv[2];
    fpreal3 Dmrow = Dmrows[vtxorder];
    // G = Pk * Dm^-T
    // G^T = Dm^-1 * Pk^T = Dm^-1 * d^T
    // gradients are PK1^T * dmrow
    fpreal3 fl = 0;
    mat3 J;
    fl -= mat3Tvecmul(PK1, Dmrow);
    
    // Now the Hessian with regard to positions x.
    // This is the multiplied out equivalent of using
    // ComputePFPx from Dynamic Deformables (but in numpy syntax):
    // dFdx = ComputePFPx(Dminv)[:, vtx*3:vtx*3+3]
    // i.e. choosing the columns of pFpx that correspond to this vertex.
    // then Jacobian is J = (dFdx.T @ dPdF) @ dFdx,
    // i.e. a tensor "chain rule"
    // so dFdx.shape = (9, 3), dPdf.shape = (9, 9), J.shape = (3, 3).
    // We re-use entries where we can for speed and to enforce symmetry.
    fpreal m1 = Dmrow.x, m2 = Dmrow.y, m3 = Dmrow.z;
    J[0].s0 = m1 * (dPdF[0 * 9 + 0] * m1 + dPdF[3 * 9 + 0] * m2 + dPdF[6 * 9 + 0] * m3) + m2 * (dPdF[0 * 9 + 3] * m1 + dPdF[3 * 9 + 3] * m2 + dPdF[6 * 9 + 3] * m3) + m3 * (dPdF[0 * 9 + 6] * m1 + dPdF[3 * 9 + 6] * m2 + dPdF[6 * 9 + 6] * m3);
    J[0].s1 = m1 * (dPdF[1 * 9 + 0] * m1 + dPdF[4 * 9 + 0] * m2 + dPdF[7 * 9 + 0] * m3) + m2 * (dPdF[1 * 9 + 3] * m1 + dPdF[4 * 9 + 3] * m2 + dPdF[7 * 9 + 3] * m3) + m3 * (dPdF[1 * 9 + 6] * m1 + dPdF[4 * 9 + 6] * m2 + dPdF[7 * 9 + 6] * m3);
    J[0].s2 = m1 * (dPdF[2 * 9 + 0] * m1 + dPdF[5 * 9 + 0] * m2 + dPdF[8 * 9 + 0] * m3) + m2 * (dPdF[2 * 9 + 3] * m1 + dPdF[5 * 9 + 3] * m2 + dPdF[8 * 9 + 3] * m3) + m3 * (dPdF[2 * 9 + 6] * m1 + dPdF[5 * 9 + 6] * m2 + dPdF[8 * 9 + 6] * m3);
    J[1].s0 = J[0].s1;
    J[1].s1 = m1 * (dPdF[1 * 9 + 1] * m1 + dPdF[4 * 9 + 1] * m2 + dPdF[7 * 9 + 1] * m3) + m2 * (dPdF[1 * 9 + 4] * m1 + dPdF[4 * 9 + 4] * m2 + dPdF[7 * 9 + 4] * m3) + m3 * (dPdF[1 * 9 + 7] * m1 + dPdF[4 * 9 + 7] * m2 + dPdF[7 * 9 + 7] * m3);
    J[1].s2 = m1 * (dPdF[2 * 9 + 1] * m1 + dPdF[5 * 9 + 1] * m2 + dPdF[8 * 9 + 1] * m3) + m2 * (dPdF[2 * 9 + 4] * m1 + dPdF[5 * 9 + 4] * m2 + dPdF[8 * 9 + 4] * m3) + m3 * (dPdF[2 * 9 + 7] * m1 + dPdF[5 * 9 + 7] * m2 + dPdF[8 * 9 + 7] * m3);
    J[2].s0 = J[0].s2;
    J[2].s1 = J[1].s2;
    J[2].s2 = m1 * (dPdF[2 * 9 + 2] * m1 + dPdF[5 * 9 + 2] * m2 + dPdF[8 * 9 + 2] * m3) + m2 * (dPdF[2 * 9 + 5] * m1 + dPdF[5 * 9 + 5] * m2 + dPdF[8 * 9 + 5] * m3) + m3 * (dPdF[2 * 9 + 8] * m1 + dPdF[5 * 9 + 8] * m2 + dPdF[8 * 9 + 8] * m3);

    // Scale force and Hessian by volume now, before damping,
    // so we can avoid potentially very large (E) * very small (disp)
    // floating point multiplies.
    fl *= V;
    mat3scaleip(J, V);

    // High volume stiffness can lead to negative eigenvalues,
    // so project each term back to Symmetric Positive Definite,
    // which will make the entire Hessian SPD, since the rest
    // of the energies (so far) are all SPD.
    projectToSPD(J);

    if (kdamp > 0.0f)
    {
        // kdamp is pre-multiplied by 1 / dt.
        int pt0 = primpts[ptidx + vtxorder];
        fpreal3 disp = vload3f(pt0, P) - vload3f(pt0, pprev);
        fl -= mat3vecmul(J, kdamp * disp);
        mat3scaleip(J, 1 + kdamp);
    }

    mat3add(H, J, H);
    *f += fl;
    return 1;

}

// Spring update, where p0 is the point being solved for.
static int
springUpdateVBD(const fpreal3 p0, const fpreal3 p1, const fpreal3 p0prev,
                const fpreal restlength,
                const fpreal kstiff, fpreal kdamp, fpreal3 *f, mat3 H)
{
    // Rest volume is stored in restlength.
    fpreal l0 = restlength;
    fpreal3 pq = p0 - p1;
    fpreal l = length(pq);
    if (l > 1e-6)
    {
        mat3 G, I;
        mat3identity(I);

        // E = 1/2 * ks * (l - l0)^2
        // f = -ks * (l - l0) * n
        // H = ks * outer(n, n) + ks * (1 - l0 / l) * (I - outer(n, n))
        fpreal3 n = pq / l;
        fpreal scale = kstiff * (l - l0);
        *f -= scale * n;
        outerprod3(n, n, G);
        mat3scaleip(G, kstiff);
        // As described in section 13.4 of Dynamic Deformables
        // and Section 3.1 of Choi, "Stable but Responsive Cloth",
        // the full Hessian can become indefinite under compression
        // so we only add the second term under expansion, i.e. l > l0.
        if (l > l0)
            mat3lcombine(scale / l, I, l0 / l, G, G);

        if (kdamp > 0)
        {
            // kdamp is pre-multiplied by 1 / dt.
            fpreal3 disp = p0 - p0prev;
            *f -= kdamp * mat3vecmul(G, disp);
            mat3scaleip(G, 1 + kdamp);
        }
        mat3add(H, G, H);
        return 1;
    }
    return 0;
}

static void
frictionUpdate(fpreal mu, fpreal lambda, mat32 T, fpreal2 u, fpreal epsU,
               fpreal3 *f, mat3 H)
{
    // Friction
    fpreal uNorm = length(u);
    if (uNorm <= 0)
        return;
    // IPC friction 
    // https://github.com/ipc-sim/ipc-toolkit/blob/main/src/ipc/friction/smooth_friction_mollifier.cpp
    // fpreal fcoeff = (ldPtan >= mus * ldPnml) ? fkin : 1;
#if 1
    fpreal f1_SF_over_x = (uNorm > epsU) ? 1 / uNorm : (-uNorm / epsU + 2) / epsU;
#else
    fpreal uNorm_eps = uNorm / epsU;
    fpreal f1_SF_over_x =  (uNorm > epsU) ? 1 : (2 * uNorm_eps  - uNorm_eps * uNorm_eps);
    u /= uNorm;
#endif
    fpreal3 Tu = (fpreal3)(dot(T[0], u), dot(T[1], u), dot(T[2], u));
    fpreal scale = mu * lambda * f1_SF_over_x;
    // printf("u = (%g, %g), Tu = (%g, %g, %g), scale = %g\n", u.x, u.y, Tu.x, Tu.y, Tu.z, scale);
    // printf("uNorm = %g, epsU = %g, f1_SF_over_x = %g\n", uNorm, epsU, 1/uNorm, f1_SF_over_x);
    *f -= scale * Tu;
    // H += scale * (T @ T.T)
    if (H)
    {
        H[0] += scale * (fpreal3)(dot(T[0], T[0]), dot(T[0], T[1]), dot(T[0], T[2]));
        H[1] += scale * (fpreal3)(dot(T[1], T[0]), dot(T[1], T[1]), dot(T[1], T[2]));
        H[2] += scale * (fpreal3)(dot(T[2], T[0]), dot(T[2], T[1]), dot(T[2], T[2]));
    }
}

// From UT_Vector3<T>::getFrameOfReference.
// Assumes z is normalized.
#if 0
static void
getFrameOfReference(const fpreal3 z, fpreal3 *x, fpreal3 *y)
{
    if (fabs(z.x) < 0.6f)
        *y = (fpreal3)(1, 0, 0);
    else if (fabs(z.z) < 0.6f)
        *y = (fpreal3)(0, 1, 0);
    else
        *y = (fpreal3)(0, 0, 1);
    *x = cross(*y, z);
    *y = cross(z, *x);
}
#else
// see Building an Orthonormal Basis, Revisited.
// TODO - replace the version in vbd_energy.cl with this
// TODO - should go somewhere global along with closestpttriangle
static void 
getFrameOfReference(const fpreal3 n, fpreal3 *b1, fpreal3 *b2)
{
    fpreal s  = n.z >= 0 ? 1 : -1;
    fpreal a = -1.0f / (s + n.z);
    fpreal b = n.x * n.y * a;
    *b1 = (fpreal3)(1.0f + s * n.x * n.x * a, s * b, -s * n.x);
    *b2 = (fpreal3)(b, s + n.y * n.y * a, -n.y);
}
#endif
// TODO - move to a shared include
// From "Real-Time Collision Detection" by Ericson.
// with modifications to return closest vertex or edge
// barycentric coords.
// Based on VEX version from pbd_constraints.h.
static fpreal3
closestpttriangle(const fpreal3 p, const fpreal3 a,
                  const fpreal3 b, const fpreal3 c,
                  int *vert, int *edge, fpreal3 *uv)
{
    *vert = *edge = -1;
    // Check if P in vertex region outside A
    fpreal3 ab = b - a;
    fpreal3 ac = c - a;
    fpreal3 ap = p - a;
    fpreal d1 = dot(ab, ap);
    fpreal d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        *vert = 0;
        *uv = (fpreal3)(1, 0, 0);
        return a; // barycentric coordinates (1,0,0)
    }

    // Check if P in vertex region outside B
    fpreal3 bp = p - b;
    fpreal d3 = dot(ab, bp);
    fpreal d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        *vert = 1;
        *uv = (fpreal3)(0, 1, 0);
        return b; // barycentric coordinates (0,1,0)
    }

    // Check if P in edge region of AB, if so return projection of P onto AB
    fpreal vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        *edge = 0;
        fpreal v = d1 / (d1 - d3);
        *uv = (fpreal3)(1-v, v, 0);
        return a + v * ab; // barycentric coordinates (1-v,v,0)
    }

    // Check if P in vertex region outside C
    fpreal3 cp = p - c;
    fpreal d5 = dot(ab, cp);
    fpreal d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        *vert = 2;
        *uv = (fpreal3)(0, 0, 1);
        return c; // barycentric coordinates (0,0,1)
    }

    // Check if P in edge region of AC, if so return projection of P onto AC
    fpreal vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        *edge = 2;
        fpreal w = d2 / (d2 - d6);
        *uv = (fpreal3)(1-w, 0, w);
        return a + w * ac; // barycentric coordinates (1-w,0,w)
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    fpreal va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        *edge = 1;
        fpreal w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        *uv = (fpreal3)(0, 1-w, w);
        return b + w * (c - b); // barycentric coordinates (0,1-w,w)
    }

    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    fpreal denom = 1.0f / (va + vb + vc);
    fpreal v = vb * denom;
    fpreal w = vc * denom;
    *uv = (fpreal3)(1-v-w, v, w);
    return a + ab * v + ac * w; // = u*a + v*b + w*c, u = va * denom = 1.0f-v-w
}

// Based on ipctk's edge_edge_closest_point, except
// currently does not update and returns zero on
// singular (parallel) edges.
static int
closestptedges(const fpreal3 ea0, const fpreal3 ea1,
               const fpreal3 eb0, const fpreal3 eb1,
               fpreal2 *uv)
{
    fpreal3 eb_to_ea = ea0 - eb0;
    fpreal3 ea = ea1 - ea0;
    fpreal3 eb = eb1 - eb0;

    fpreal dotebea = dot(eb, ea);
    mat2 coef = (mat2)(dot(ea, ea), -dotebea, -dotebea, dot(eb, eb));

    fpreal2 rhs;
    rhs.s0 = -dot(eb_to_ea, ea);
    rhs.s1 = dot(eb_to_ea, eb);

    mat2 coefInv;
    if (!mat2inv(coef, &coefInv))
        return 0;
    *uv = mat2vecmul(coefInv, rhs);
    return 1;
}

#define USE_HITUV
static int
collisionUpdate(fpreal kcol, const global float *P,
                const global float *pprev, const global float *pscale,
#ifdef HAS_collisionstiffnessscale
                const global float *colstiffscale,
#endif                
                int type, const int vtx,
                fpreal2 hituv, fpreal3 hitnml, const global int *pts,
                const fpreal mus, const fpreal muk,
                fpreal3 *f, mat3 H, fpreal *outrad, fpreal *outdepth)
{
    // Point triangle update.
    fpreal3 p = vload3f(pts[0], P);
    fpreal3 a = vload3f(pts[1], P);
    fpreal3 b = vload3f(pts[2], P);
    fpreal3 c = vload3f(pts[3], P);
    fpreal3 triuv, diff, n;
    // bary holds how much of the force/hessian should be applied for the point
    // given its vertex order.
    fpreal bary[4];
    fpreal hitrad, prad = pscale[pts[0]];
    fpreal arad = pscale[pts[1]], brad = pscale[pts[2]], crad = pscale[pts[3]];

    // Update closest point and normal.
    if (type == 0)
    {
#ifdef USE_HITUV
        // Re-use the initial hit information.

        // point-tri barys
        // vtx 1-3 force direction is reversed.
        bary[0] = 1;
        bary[1] = -(1 - hituv.x - hituv.y);
        bary[2] = -hituv.x;
        bary[3] = -hituv.y;
        fpreal3 closept = a + (b - a) * hituv.x + (c - a) * hituv.y;
        diff = p - closept;
        // Add in triangle thickness.
        fpreal trirad = arad + (brad - arad) * hituv.x + (crad - arad) * hituv.y;
        n = hitnml;
#else
        // Re-calculate the hit information.
        // Point-triangle
        int vert, edge;
        fpreal3 closept = closestpttriangle(p, a, b, c, &vert, &edge, &triuv);
        diff = p - closept;
        fpreal trirad = arad * triuv.x + brad * triuv.y + crad * triuv.z;
        // point-tri barys
        // vtx 1-3 force direction is reversed.
        bary[0] = 1;
        bary[1] = -triuv.x;
        bary[2] = -triuv.y;
        bary[3] = -triuv.z;
        n = normalize(cross(a - b, c - b));        
#endif
        hitrad = prad + trirad;
    }
    else
    {
        // Edge-edge
        // NOTE this currently does not update
        // hituv for singular (parallel) edges        
#ifndef USE_HITUV
        closestptedges(p, a, b, c, &hituv);
#endif
        diff = mix(p, a, hituv.x) - mix(b, c, hituv.y);
        hitrad = mix(prad, arad, hituv.x) + mix(brad, crad, hituv.y);
        // vtx 2 and 3 force direction is reversed.
        bary[0] = 1 - hituv.x;
        bary[1] = hituv.x;
        bary[2] = -(1 - hituv.y);
        bary[3] = -hituv.y;
    }
    fpreal depth = dot(diff, n);
    fpreal d = hitrad - depth;
    if (d <= 0)
        return 0;
    if (outrad)
        *outrad = hitrad;
    if (outdepth)
        *outdepth = depth;
#ifdef HAS_collisionstiffnessscale
// For now the first point determine's the collision
// stiffness, which makes computing adaptive stiffness
// much easier.
#if 0
    kcol *= (colstiffscale[pts[0]] + colstiffscale[pts[1]] +
             colstiffscale[pts[2]] + colstiffscale[pts[3]]) / 4;
#else
    kcol *= (vtx == 0 && colstiffscale) ? colstiffscale[pts[0]] : 1;
#endif    
#endif

    // Add gradient and Hessian of d to f and H.
    // Collision force magnitude.
    fpreal lambda = kcol * d;
    // Collision force along collision normal.
    fpreal3 cf = lambda * n;
    // Collision Hessian.
    mat3 cH;
    if (H)
    {
        outerprod3(n, n, cH);
        mat3scaleip(cH, kcol);
    }    
    // Friction handling.
    if (mus > 0)
    {
        fpreal3 dx0 = p - vload3f(pts[0], pprev);
        fpreal3 dx1 = a - vload3f(pts[1], pprev);
        fpreal3 dx2 = b - vload3f(pts[2], pprev);
        fpreal3 dx3 = c - vload3f(pts[3], pprev);
        const fpreal epsU = muk;//mus * ldPnml;//0.01;
        fpreal3 dx, b0, b1;
        fpreal enl = 1;
        if (type == 0)
        {
            // Point-triangle
            // Dx is dif between points displacement and triangle's
            // interoplated displacement.
#ifdef USE_HITUV
            dx = dx0 - (dx1 + hituv.x * (dx2 - dx1) + hituv.y * (dx3 - dx1));
#else
            dx = dx0 - (dx1 * triuv.x + dx2 * triuv.y + dx3 * triuv.z);
#endif
            // Basis is triangle edge and normal.
            b0 = normalize(b - a);
            b1 = cross(b0, n);
        }
        else
        {
            // Edge-edge
            // Dx is diff between lerp of closest points' displacements.
            dx = mix(dx0, dx1, hituv.x) - mix(dx2, dx3, hituv.y);
            // Create a basis between the two edges, which might not work if
            // they are parallel (enl -> 0).
            b0 = normalize(a - p); 
            fpreal3 en = cross(b0, c - b);
            enl = length(en);
            b1 = cross(en/enl, b0);
        }
        // Skip friction update for parallel edges with indefinite normal.
        if (enl > 1e-5)
        {
            // T is 3x2 matrix with b0, b1 as its columns
            mat32 T;
            T[0] = (fpreal2)(b0.s0, b1.s0);
            T[1] = (fpreal2)(b0.s1, b1.s1);
            T[2] = (fpreal2)(b0.s2, b1.s2);
            // u = T.T @ dx
            fpreal2 u = (fpreal2)(dot(b0, dx), dot(b1, dx));
            // NOTE: first parameter is dynamic friction(?)
            frictionUpdate(mus, lambda, T, u, epsU, &cf, cH);
        }
    }
    // Add in force and hessian scaled by barycentric contribution.
    *f += bary[vtx] * cf;
    if (H)
        mat3lcombine(bary[vtx] * bary[vtx], cH, 1, H, H);

    return 1;
}

static fpreal
groundUpdate(const fpreal kcol, const fpreal3 p,
             const fpreal3 pprev, const fpreal rad,
             const fpreal3 dir, const fpreal3 origin,
             const fpreal mus, const fpreal muk,
             fpreal3 *f, mat3 H)
{
    fpreal3 n = normalize(dir);
    fpreal d = rad - dot(p - origin, n);
    if (d <= 0)
        return 0;
    // Add gradient and Hessian of d to f and H.
    // Collision force magnitude.
    fpreal lambda = kcol * d;
    // Update force along collision normal.
    *f += lambda * n;
    // Update Hessian.
    if (H)
    {
        mat3 N;
        outerprod3(n, n, N);
        mat3lcombine(kcol, N, 1, H, H);
    }

    // Friction handling.
    if (mus > 0)
    {
        fpreal3 dx = p - pprev;
        // fpreal   ldPnml = fabs(dot(dx, n));
        const fpreal epsU = muk;//mus * ldPnml;//0.01;
        //fpreal epsU = epsV * dt;
        // Get basis vectors from normal.
        fpreal3 b0, b1;
        getFrameOfReference(n, &b0, &b1);
        // T is 3x2 matrix with b0, b1 as its columns
        mat32 T;
        T[0] = (fpreal2)(b0.s0, b1.s0);
        T[1] = (fpreal2)(b0.s1, b1.s1);
        T[2] = (fpreal2)(b0.s2, b1.s2);
        // u = T @ dx
        fpreal2 u = (fpreal2)(dot(b0, dx), dot(b1, dx));
        // NOTE: first parameter is dynamic friction(?)
        frictionUpdate(mus, lambda, T, u, epsU, f, H);
    }
    return d;
}

static fpreal
findMaxDepthAndGrad(const int idx,
                    const fpreal kcol,
                    const global float *P,
                    const global float *pscale,
                    const fpreal hitradratio,
                    const global int * hittypes,
                    const global float * hituvs,
                    const global float * hitnmls,
                    const global int * hitpts,
                    const global int * hitidx_index,
                    const global int * hitidx,
                    const global int * hitvtx,
                    int doground,
                    fpreal3 origin,
                    fpreal3 dir, 
                    fpreal3 *cf)
{
    int pthitidx = hitidx_index[idx];
    int nhits =  hitidx_index[idx + 1] - pthitidx;
    fpreal maxdepth = 0;
    for(int i = 0; i < nhits; i++)
    {
        int vtx = hitvtx[pthitidx + i];
        // Only consider points in pt/tri contacts.
        if (vtx != 0)
            continue;
        int hit = hitidx[pthitidx + i];
        int hittype = hittypes[hit];
        if (hittype != 0)
            continue;
        fpreal2 uvs = vload2f(hit, hituvs);
        fpreal3 hitnml = vload3f(hit, hitnmls);
        fpreal hitrad, depth;
        if (collisionUpdate(kcol, P, 0, pscale,
#ifdef HAS_collisionstiffnessscale
                        0,
#endif        
                        hittype, vtx, uvs, hitnml, &hitpts[hit * 4],
                        0, 0, cf, 0, &hitrad, &depth))
        {
            hitrad *= hitradratio;
            maxdepth = max(hitrad - depth, maxdepth);
        }
    }
    fpreal3 p = vload3f(idx, P);
    if (doground)
    {
        fpreal hitrad = pscale[idx];
        // Add ground penalty force.
        fpreal d = groundUpdate(kcol, p, 0, hitrad,
                               dir, origin, 0, 0, cf, 0);
        if (d > 0)
        {
            // d = rad - depth.
            fpreal depth = hitrad - d;
            hitrad *= hitradratio;
            maxdepth = max(hitrad - depth, maxdepth);
        }
    }

    return maxdepth;
}

#ifndef BLOCKSIZE
#define BLOCKSIZE 16
#endif

// Each thread requires 4x3 + 1 fpreals for material calc.
#define LMEMSIZE 16
#define LMEMCOLOFF LMEMSIZE-3

__attribute__((reqd_work_group_size(BLOCKSIZE, 1, 1)))
kernel void
solveVBD( 
    int worksets_begin,
    int worksets_length,
    float dt,
    int coloredpt_length,
    const global int * coloredpt,
    int P_length,
    global float * P,
#ifdef HAS_pnext
    int pnext_length,
    global float * pnext,
#endif
    int pprevious_length,
    const global float * pprevious,
    int inertial_length,
    const global float * inertial,
    int mass_length,
    const global float * mass,
    int pscale_length,
    const global float * pscale,
#ifdef HAS_fallback
    int fallback_length,
    global int * fallback,
#endif    
    int type_length,
    const global int * type_hash,
    int stiffness_length,
    const global float * stiffness,
    int dampingratio_length,
    const global float * dampingratio,
#ifdef HAS_volumestiffness
    int volumestiffness_length,
    const global float * volumestiffness,
#endif
#ifdef HAS_volumestiffnessscale
    int volumestiffnessscale_length,
    const global float * volumestiffnessscale,
#endif
#ifdef HAS_fiberscale
    int fiberscale_length,
    const global float * fiberscale_in,
#endif
#ifdef HAS_fiberstiffness
    int fiberstiffness_length,
    const global float * fiberstiffness,
#endif
    int restlength_length,
    const global float * restlength,
#ifdef HAS_restvector
    int restvector_length,
    const global float * restvector,
#endif
#ifdef HAS_restmatrix
    int restmatrix_length,
    const global float * restmatrix,
#endif
#ifdef HAS_hittypes
    int hittypes_length,
    const global int * hittypes_index,
    const global int * hittypes,
    int hituvs_length,
    const global int * hituvs_index,
    const global float * hituvs,
    int hitnmls_length,
    const global int * hitnmls_index,
    const global float * hitnmls,
    int hitpts_length,
    const global int * hitpts_index,
    const global int * hitpts,
    int hits_length,
    const global int * hits_index,
    const global int * hits,
    int hitvtx_length,
    const global int * hitvtx_index,
    const global int * hitvtx,
#endif
    float3 gravity,
    int doground,
    float3 origin,
    float3 dir,
    float kcol,
#ifdef HAS_collisionstiffnessscale
    int collisionstiffnessscale_length,
    const global float *collisionstiffnessscale,
#endif
    float mus, float muk,
    int ptprims_length,
    const global int * ptprims_index,
    const global int * ptprims,
    int primpts_length,
    const global int * primpts_index,
    const global int * primpts,
    int ptvertices_length,
    const global int * ptvertices_index,
    const global int * ptvertices,
    int vtxprimindex_length,
    const global int * vtxprimindex
    )
{
    // Each thread needs LMEMSIZE fpreals.
    __local accum_t buf[LMEMSIZE * BLOCKSIZE];
    int base = P_length * BLOCKSIZE;
    int docopy = 0;
#ifdef HAS_pnext
    // We add base (npoints * BLOCKSIZE) to all the copy-pass workgroup
    // offsets when coloring to indicate when to do the copy vs regular solve.
    // Only relevant when self-collisions are enabled, when we have pnext.
    docopy = (worksets_begin >= base);
#endif
    int idx = get_global_id(0);
    if (idx >= worksets_length)
        return;
    idx += worksets_begin - docopy * base;
    int lsize = get_local_size(0);
    // Each group processes one point.
    idx /= lsize;
    // Get the actual point number from the colored set for this workset.
    idx = coloredpt[idx];
    if (mass[idx] == 0.0f)
        return;
    int tid = get_local_id(0);
    // You'd think this is way too expensive to do per-group, but
    // it's actually slower to device-enqueue the copy even on NVIDIA :(
#ifdef HAS_pnext
    if (docopy)
    {
        // Copy from pnext to P.
        if (tid == 0)
            vstore3(vload3(idx, pnext), idx, P);
        return;
    }
#endif
    int primidx = ptprims_index[idx];
    int nprims = ptprims_index[idx + 1] - primidx;
    int vtxidx = ptvertices_index[idx];    

    fpreal3 f = 0;
    fpreal kdamp = 0;
    mat3 H;
    mat3zero(H);
    // Each thread computes f and H for one or more primitives.
    for(int i = tid; i < nprims; i += lsize)
    {
        int prim = ptprims[primidx + i];
        fpreal kstiff = stiffness[prim];
        if (kstiff < 1e-6f)
            continue;
        int ptidx = primpts_index[prim];
        int vtxorder = vtxprimindex[ptvertices[vtxidx + i]];
        int type = type_hash[prim];
        fpreal rl = restlength[prim];
#ifndef QUASISTATIC
        // Premultiply by 1/dt.
        kdamp = dampingratio[prim] / dt;
#endif

#ifdef HAS_restvector

#ifdef HAS_restmatrix
        // Tet materials.
        fpreal kvol = kstiff, fiberscale = 0, kfiber = 0;
#ifdef HAS_volumestiffness
        kvol = volumestiffness[prim];
#ifdef HAS_volumestiffnessscale
        kvol *= volumestiffnessscale[prim];
#endif        
#endif
#ifdef HAS_fiberscale
        fiberscale = fiberscale_in[prim];
#endif
#ifdef HAS_fiberstiffness
        kfiber = fiberstiffness[prim];
#endif
        if (type == TETARAPNORMVOL || type == TETFIBERNORM)
            tetUpdateVBD(type, ptidx, prim, vtxorder, primpts, P, pprevious,
                        rl, restvector, restmatrix,
                        kstiff, kvol, fiberscale, kfiber, kdamp,
                        &f, H);
#endif
        // Soft pins, pinned point goes first.
        if (type == PIN)
        {
            int pt0 = primpts[ptidx];
            springUpdateVBD(vload3f(pt0, P),
                            vload4f(prim, restvector).xyz,
                            vload3f(pt0, pprevious),
                            rl, kstiff, kdamp, &f, H);
        }
#endif
        // Springs, first point position is the current one.
        if (type == DISTANCE)
        {
            int pt0 = primpts[ptidx + vtxorder];
            int pt1 = primpts[ptidx + (1 - vtxorder)];
            springUpdateVBD(vload3f(pt0, P),
                            vload3f(pt1, P),
                            vload3f(pt0, pprevious),
                            rl, kstiff, kdamp, &f, H);
        }

    }

    int colhit = 0, nhits = 0;
#ifdef HAS_hittypes
    // Each thread updates f and H for one or more contacts.
    int hitidx = hits_index[idx];
    nhits = hits_index[idx + 1] - hitidx;
    for(int i = tid; i < nhits; i += lsize)
    {
        int hit = hits[hitidx + i];
        int vtx = hitvtx[hitidx + i];
        int hittype = hittypes[hit];
        fpreal2 uvs = vload2f(hit, hituvs);
        fpreal3 hitnml = vload3f(hit, hitnmls);
        colhit += collisionUpdate(kcol, P, pprevious, pscale,
#ifdef HAS_collisionstiffnessscale
                                  collisionstiffnessscale,
#endif
                                  hittype, vtx, uvs, hitnml, &hitpts[hit * 4],
                                  mus, muk, &f, H, 0, 0);
    }
#endif

    // Store each thread's f, H, and colhit into local memory.
    __local accum_t *lbuf = buf + tid * LMEMSIZE;
    vstore3(toaccum3(f), 0, lbuf);
    vstore3(toaccum3(H[0]), 1, lbuf);
    vstore3(toaccum3(H[1]), 2, lbuf);
    vstore3(toaccum3(H[2]), 3, lbuf);
    lbuf[LMEMCOLOFF] = (accum_t) colhit;

    // Wait for all local memory writes to complete.
    barrier(CLK_LOCAL_MEM_FENCE);

    // For low valence/hit points, nprims/nhits can be less than the workgroup size.
    int nvals = min(max(nprims, nhits), lsize);
    // Do binary reduction of f, H, and colhit in local memory,
    // similar to reduceLocal function in reduce.h.
    for(int offset = lsize / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset && tid + offset < nvals)
        {
            __local accum_t *src = buf + (tid + offset) * LMEMSIZE;
            vstore3(vload3(0, lbuf) + vload3(0, src), 0, lbuf);
            vstore3(vload3(1, lbuf) + vload3(1, src), 1, lbuf);
            vstore3(vload3(2, lbuf) + vload3(2, src), 2, lbuf);
            vstore3(vload3(3, lbuf) + vload3(3, src), 3, lbuf);
            lbuf[LMEMCOLOFF] += src[LMEMCOLOFF];
        }    
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Once-per-point processing done by thread 0.
    if (tid == 0)
    {
        // Reload results of reduction.
        f = tofpreal3(vload3(0, lbuf));
        H[0] = tofpreal3(vload3(1, lbuf));
        H[1] = tofpreal3(vload3(2, lbuf));
        H[2] = tofpreal3(vload3(3, lbuf));
        colhit = (int)lbuf[LMEMCOLOFF];

        fpreal3 p = vload3f(idx, P);
        int groundhit = 0;
        if (doground)
        {
            // Add ground penalty force and Hessian.
            fpreal3 pprev = vload3f(idx, pprevious);
            groundhit = (groundUpdate(kcol, p, pprev, pscale[idx],
                                      asfpreal3(dir), asfpreal3(origin),
                                      mus, muk, &f, H) > 0);
        }

#ifdef HAS_fallback
        fallback[idx] |= (colhit > 0) || groundhit;
#endif            

        // Add inertial terms once on the single thread.
#ifdef QUASISTATIC
        // For quasistatic we add external forces,
        // in this case just mass * gravity.
        f += mass[idx] * asfpreal3(gravity);
#else
        // For dynamic we add the inertial terms.
        fpreal m_dt2 = mass[idx] / (dt * dt);        
        f += m_dt2 * (vload3f(idx, inertial) - p);
        // H += mass * I / dt^2
        H[0].s0 += m_dt2;
        H[1].s1 += m_dt2;
        H[2].s2 += m_dt2;
#endif

        // dx = H^-1 * f;
#if 0
        mat3 Hinv;
        // If H is almost singular, i.e. |det(H)| <= 1e-5
        // then skip this update.
        if (mat3invtol(H, Hinv, 1e-5) == 0)
            return;
        fpreal3 dx = mat3vecmul(Hinv, f);
#else
        // We've ensured that H is SPD so we can use LDLT solve.
        fpreal3 dx = mat3solve_LDLT(H, f);
#endif  
        p += dx;
#ifdef HAS_pnext
        vstore3f(p, idx, pnext);
#else
        vstore3f(p, idx, P);
#endif
    }
}

static float
getAcceleratorOmega(int order, float rho, float prevOmega)
{
    if (order == 1)
        return 1;
    if (order == 2)        
        return  2 / (2 - (rho * rho));
    return 4.0 / (4.0 - (rho * rho) * prevOmega);
}

kernel void
updateOmega( 
            int omega_length,
            global fpreal * restrict omega,
            int iter_length,
            global int * restrict iter,
            float rho,
            int offset
)
{
    int idx = get_global_id(0);
    if (idx >= omega_length)
        return;
    // For extreme initial deformations, the first few iterations
    // can overshoot, so we allow offsetting the iteration count.
    // Especially useful for quasistatic.
    int i = max(*iter - offset, 1);
    *omega = getAcceleratorOmega(i, rho, *omega);
    *iter += 1;
}

kernel void
copyToPrevIter( 
               int ppreviter_length,
               global float * restrict ppreviter,
               int P_length,
               const global float * restrict P
)
{
    int idx = get_global_id(0);
    if (idx >= ppreviter_length)
        return;
    vstore3(vload3(idx, P), idx, ppreviter);
}

kernel void
applyOmega( 
           int P_length,
           global float * restrict P,
           int mass_length,
           const global float * restrict mass,
           int fallback_length,
           const global int * restrict fallback,
           int ppreviter_length,
           const global float * restrict ppreviter,
           int plastiter_length,
           global float * restrict plastiter,
           int omega_length,
           const global float * restrict omega
)
{
    int idx = get_global_id(0);
    if (idx >= P_length)
        return;
    if (mass[idx] == 0.0 || fallback[idx])
        return;
    float3 p = vload3(idx, P);
    float3 plast = vload3(idx, plastiter);
    vstore3(*omega * (p - plast) + plast, idx, P);
    vstore3(vload3(idx, ppreviter), idx, plastiter);
}

kernel void
updateAdaptiveCollisionStiffness(int P_length,
                        const global float *P,
                        int mass_length,
                        const global float *mass,
                        int pscale_length,
                        const global float *pscale,
                        float hitradratio,
                        float stiffmult,
                        float maxstiffscale,
                        int maxdepth_length,
                        global float *maxdepth_inout,
                        int colstiffscale_length,
                        global float *colstiffscale_inout,
                        int hittypes_length,
                        const global int * hittypes_index,
                        const global int * hittypes,
                        int hituvs_length,
                        const global int * hituvs_index,
                        const global float * hituvs,
                        int hitnmls_length,
                        const global int * hitnmls_index,
                        const global float * hitnmls,
                        int hitpts_length,
                        const global int * hitpts_index,
                        const global int * hitpts,
                        int hitidx_length,
                        const global int * hitidx_index,
                        const global int * hitidx,
                        int hitvtx_length,
                        const global int * hitvtx_index,
                        const global int * hitvtx,
                        int doground,
                        float3 origin,
                        float3 dir
)
{
    int idx = get_global_id(0);
    if (idx >= P_length)
        return;        
    if (mass[idx] == 0.0f)
        return;
    fpreal prevmaxdepth = maxdepth_inout[idx];
    int pthitidx = hitidx_index[idx];
    int nhits =  hitidx_index[idx + 1] - pthitidx;
    // We don't actually use the collision energy, so fake kcol.
    fpreal kcol = 1;
    fpreal3 cf = 0;
    fpreal maxdepth = findMaxDepthAndGrad(idx, kcol, P, pscale, hitradratio,
                                         hittypes, hituvs, hitnmls, hitpts,
                                         hitidx_index, hitidx, hitvtx,
                                         doground,
                                         asfpreal3(origin), asfpreal3(dir),
                                         &cf);
    fpreal scale = colstiffscale_inout[idx];
    if (prevmaxdepth > 0 && maxdepth > 0 && maxdepth > prevmaxdepth)
        scale = min(scale * stiffmult, (fpreal)maxstiffscale);
    // Store mindist to compare next iteration.
    maxdepth_inout[idx] = maxdepth;
    colstiffscale_inout[idx] = scale;
}

kernel void
initAdaptiveStiffness( 
    float dt,
    int P_length,
    global float * P,
    int pprevious_length,
    const global float * pprevious,
    int inertial_length,
    const global float * inertial,
    int mass_length,
    const global float * mass,
    int pscale_length,
    const global float * pscale,
    const float hitradratio,
    const int estimateInitialStiffness,
    const float maxinitialstiffscale,
    int collisionstiffnessscale_length,
    global float * collisionstiffnessscale,
    int maxdepth_length,
    global float *maxdepth,

    int type_length,
    const global int * type_hash,
    int stiffness_length,
    const global float * stiffness,
    int dampingratio_length,
    const global float * dampingratio,
#ifdef HAS_volumestiffness
    int volumestiffness_length,
    const global float * volumestiffness,
#endif
#ifdef HAS_fiberscale
    int fiberscale_length,
    const global float * fiberscale_in,
#endif
#ifdef HAS_fiberstiffness
    int fiberstiffness_length,
    const global float * fiberstiffness,
#endif
    int restlength_length,
    const global float * restlength,
#ifdef HAS_restvector
    int restvector_length,
    const global float * restvector,
#endif
#ifdef HAS_restmatrix
    int restmatrix_length,
    const global float * restmatrix,
#endif

    int hittypes_length,
    const global int * hittypes_index,
    const global int * hittypes,
    int hituvs_length,
    const global int * hituvs_index,
    const global float * hituvs,
    int hitnmls_length,
    const global int * hitnmls_index,
    const global float * hitnmls,
    int hitpts_length,
    const global int * hitpts_index,
    const global int * hitpts,
    int hits_length,
    const global int * hitidx_index,
    const global int * hitidx,
    int hitvtx_length,
    const global int * hitvtx_index,
    const global int * hitvtx,

    float3 gravity,
    int doground,
    float3 origin,
    float3 dir,
    float kcol,
    float mus,
    float muk,
    int ptprims_length,
    const global int * ptprims_index,
    const global int * ptprims,
    int primpts_length,
    const global int * primpts_index,
    const global int * primpts,
    int ptvertices_length,
    const global int * ptvertices_index,
    const global int * ptvertices,
    int vtxprimindex_length,
    const global int * vtxprimindex
    )
{
    int idx = get_global_id(0);
    if (idx >= P_length)
        return;        
    if (mass[idx] == 0.0f)
        return;
    int pthitidx = hitidx_index[idx];
    int nhits = hitidx_index[idx + 1] - pthitidx;
    // Find max depth and collision energy gradient.
    fpreal3 cf = 0;
    maxdepth[idx] = findMaxDepthAndGrad(idx, kcol, P, pscale, hitradratio,
                                        hittypes, hituvs, hitnmls, hitpts,
                                        hitidx_index, hitidx, hitvtx,
                                        doground,
                                        asfpreal3(origin), asfpreal3(dir),
                                        &cf);
    if (!estimateInitialStiffness)
        return;

    // No additional stiffness if no collision energy.
    fpreal cfnormsqr = dot(cf, cf);
    if (cfnormsqr < 1e-6)
        return;

    // Now get material gradient.
    int primidx = ptprims_index[idx];
    int nprims = ptprims_index[idx + 1] - primidx;
    int vtxidx = ptvertices_index[idx];    

    fpreal3 f = 0;
    // Premultiply by 1/dt, in case we ever actually need damping here.
    fpreal kdamp = 0 / dt;
    mat3 H;
    mat3zero(H);
    for(int i = 0; i < nprims; i ++)
    {
        int prim = ptprims[primidx + i];
        fpreal kstiff = stiffness[prim];
        if (kstiff < 1e-6f)
            continue;
        int ptidx = primpts_index[prim];
        int vtxorder = vtxprimindex[ptvertices[vtxidx + i]];
        int type = type_hash[prim];
        fpreal rl = restlength[prim];

#ifdef HAS_restvector

#ifdef HAS_restmatrix
        // Tet materials.
        fpreal kvol = kstiff, fiberscale = 0, kfiber = 0;
#ifdef HAS_volumestiffness
        kvol = volumestiffness[prim];
#endif
#ifdef HAS_fiberscale
        fiberscale = fiberscale_in[prim];
#endif
#ifdef HAS_fiberstiffness
        kfiber = fiberstiffness[prim];
#endif
        if (type == TETARAPNORMVOL || type == TETFIBERNORM)
            tetUpdateVBD(type, ptidx, prim, vtxorder, primpts, P, pprevious,
                        rl, restvector, restmatrix,
                        kstiff, kvol, fiberscale, kfiber, kdamp,
                        &f, H);
#endif
        // Soft pins, pinned point goes first.
        if (type == PIN)
        {
            int pt0 = primpts[ptidx];
            springUpdateVBD(vload3f(pt0, P),
                            vload4f(prim, restvector).xyz,
                            vload3f(pt0, pprevious),
                            rl, kstiff, kdamp, &f, H);
        }
#endif
        // Springs, first point position is the current one.
        if (type == DISTANCE)
        {
            int pt0 = primpts[ptidx + vtxorder];
            int pt1 = primpts[ptidx + (1 - vtxorder)];
            springUpdateVBD(vload3f(pt0, P),
                            vload3f(pt1, P),
                            vload3f(pt0, pprevious),
                            rl, kstiff, kdamp, &f, H);
        }

    }

    // Add inertial terms once on the single thread.
#ifdef QUASISTATIC
    // For quasistatic we add external forces,
    // in this case just mass * gravity.
    f += mass[idx] * asfpreal3(gravity);
#else
    // For dynamic we add the inertial terms.
    fpreal m_dt2 = mass[idx] / (dt * dt);        
    f += m_dt2 * (vload3f(idx, inertial) - vload3f(idx, P));
    // H += mass * I / dt^2
    H[0].s0 += m_dt2;
    H[1].s1 += m_dt2;
    H[2].s2 += m_dt2;
#endif

    fpreal kappa = collisionstiffnessscale[idx];
    kappa = max(kappa, -dot(cf, f) / cfnormsqr);
    collisionstiffnessscale[idx] = min(maxinitialstiffscale, (float)kappa);
        // kappa[idx] = max(0.0f, -dot(cf, f) / cfnormsqr);
        // kappa[idx] = -dot(cf, f) / cfnormsqr;

    // minimize approximated next search direction
    //            Eigen::VectorXd HInvGc, HInvGE;
    //            linSysSolver->solve(g_c, HInvGc);
    //            linSysSolver->solve(g_E, HInvGE);
    //            kappa = HInvGE.dot(HInvGc) / HInvGc.squaredNorm();
    //            if(kappa < 0.0) {
    //                kappa = g_c.dot(g_E) / g_c.squaredNorm();
    //            }

}

kernel void
updateAdaptiveStiffness( 
    int P_length,
    global float * restrict P,
    int type_length,
    global int * restrict type,
    int restmatrix_length,
    global float * restrict restmatrix,
    int primpts_length,
    global int * restrict primpts_index,
    global int * restrict primpts,
    int volumestiffness_length,
    global float * restrict volumestiffness,
    int volumestiffnessscale_length,
    global float * restrict volumestiffnessscale,
    float  beta
)
{
    int idx = get_global_id(0);
    if (idx >= volumestiffnessscale_length)
        return;
    // For now only handles Tet Volume constraints
    // and volumestiffnessscale.
    if (type[idx] != TETARAPNORMVOL)
        return;
    if (volumestiffnessscale[idx] >= 1)
        return;
    int ptidx = primpts_index[idx];
    fpreal3 p0 = vload3f(primpts[ptidx + 0], P);
    fpreal3 p1 = vload3f(primpts[ptidx + 1], P);
    fpreal3 p2 = vload3f(primpts[ptidx + 2], P);
    fpreal3 p3 = vload3f(primpts[ptidx + 3], P);    
    
    mat3 F, Ds, Dminv;
    mat3load(idx, restmatrix, Dminv);
    // Ds = | p0-p3 p1-p3 p2-p3 |
    mat3fromcols(p0 - p3, p1 - p3, p2 - p3, Ds);
    // F = Ds * Dm^-1
    mat3mul(Ds, Dminv, F);

    // Treat change in volume as constraint error.
    fpreal C = det3(F) - 1;
    fpreal scale = volumestiffnessscale[idx];
    fpreal maxstiff = volumestiffness[idx];
    // Nowhere to go plus NaNs lurking below.
    if (maxstiff < 1e-9)
        return;
    fpreal curstiff = maxstiff * scale;
    // Bump current stiffness by scaled amount of constraint error.
    curstiff += beta * fabs(C);
    // Recalc scale and clamp to 1.
    scale = curstiff / maxstiff;
    volumestiffnessscale[idx] = min((fpreal)1, scale);
}
