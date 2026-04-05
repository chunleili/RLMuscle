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
 * NAME:    pbd_util.cl ( CE Library, OpenCL)
 *
 * COMMENTS:
 *    PBD utility and collisions functions
 */

#include <platform.h>
#include <typedefines.h>
#include <matrix.h>
#include <quaternion.h>
#include <pbd_types.h>


kernel void
computePressureValues(int P_length,
                    global fpreal *P,
                    int volumepts_length,
                    global int *volumepts_index,
                    global int *volumepts,
                    int pressuregradient_length,
                    global float *pressuregradient,
                    int volume_length,
                    global float *volume
#ifdef HAS_pressurescale
                    , int pressurescale_length,
                    global const float *pressurescale
#endif
#if defined(__opencl_c_work_group_collective_functions) && defined(__opencl_c_device_enqueue)
                    , int type_length,
                    global const int *type,
                    int pprevious_length,
                    global fpreal * pprevious,
                    int mass_length,
                    global float * mass,
#ifdef HAS_stopped
                    int stopped_length,
                    global const int *stopped_in,
#endif
#ifdef HAS_restvector
                    int restvector_length,
                    global float * restvector,
#endif
                    int pts_length,
                    global int * pts_index,
                    global int * pts
#endif
)
{
    // Compute pressure gradient and volume for each point.
    int idx = get_global_id(0);
    if (idx < pressuregradient_length)
    {   int ptidx = volumepts_index[idx];
        int ntris = (volumepts_index[idx + 1] - ptidx) / 2;
        fpreal3 grad = 0;
        fpreal trivol = 0;
        fpreal3 p0 = vload3(idx, P);
        for(int i = 0; i < ntris; i++)
        {
            int pt1 = volumepts[ptidx + i * 2];
            int pt2 = volumepts[ptidx + i * 2 + 1];
            fpreal3 p1 = vload3(pt1, P);
            fpreal3 p2 = vload3(pt2, P);

            fpreal3 n = cross(p1, p2);
            grad += n;
            // Point with minimum index in triangle holds volume for that triangle.
            trivol += select((fpreal)0.0, dot(n, p0), (exint)(idx < pt1 && idx < pt2));
        }
#ifdef HAS_pressurescale
        grad *= pressurescale[idx];
#endif
        vstore3f(grad, idx, pressuregradient);
        volume[idx] = (float)trivol / 6;
    }
}

static fpreal3
applyFriction(fpreal3 pi, fpreal3 pprevi, fpreal3 hitdp, fpreal3 hitnml,
              float mus, float muk, float frscale)
{
    fpreal3  dpi = pi - pprevi;
    fpreal3  dp = dpi - hitdp;
    fpreal3  dPnml = dot(dp, hitnml) * hitnml;
    fpreal3  dPtan = dp - dPnml;
    fpreal   ldPtan = length(dPtan);
    fpreal   ldPnml = length(dPnml);

    // ldPnml is our approximate normal force, so we scale our tangent
    // by the ratio, clamping at 1.
    fpreal fkin = muk * ldPnml;
    fkin = select(fkin / ldPtan, (fpreal)1, (exint)(fkin >= ldPtan));

    // Check if we lie in the in the static cone, if so bring to
    // a full stop.
    fpreal fcoeff = select((fpreal)1, fkin, (exint)(ldPtan >= mus * ldPnml));

    fcoeff *= -frscale;
    pi += fcoeff * dPtan;
    return pi;
}

kernel void
surfaceCollision(float timeinc,
                 int P_length,
                 global fpreal *P_inout,
                 int pprevious_length,
                 global fpreal *pprevious_in,
                 int mass_length,
                 global float *mass,
#ifdef HAS_stopped
                 int stopped_length,
                 global const int *stopped,
#endif
                 int hitnum_len,
                 global int *hitnum_in,
                 int hitpos_len,
                 global fpreal *hitpos_in,
                 int hitnml_len,
                 global float *hitnml_in,
                 int hitv_len,
                 global float *hitv_in,
                 float frscale,
                 float mus, float muk
#ifdef HAS_friction
		 , int friction_length,
		 global float *friction_in
#endif
#ifdef HAS_dynamicfriction
		 , int dynamicfriction_length,
		 global float *dynamicfriction_in
#endif
		 )
{
    int idx = get_global_id(0);
    if (idx >= P_length)
        return;

    if (hitnum_in[idx] <= 0 || mass[idx] == 0.0f
#ifdef HAS_stopped
        || stopped[idx] & 1
#endif
        )
        return;

#ifdef HAS_friction
    mus *= friction_in[idx];
#endif
#ifdef HAS_dynamicfriction
    muk *= dynamicfriction_in[idx];
#endif

    fpreal3  hitdp = vload3f(idx, hitv_in);
    hitdp *= timeinc;

    fpreal3  pi = vload3(idx, P_inout);
    fpreal3  pprevi = vload3(idx, pprevious_in);
    fpreal3  hitnml = vload3f(idx, hitnml_in);

    pi = applyFriction(pi, pprevi, hitdp, hitnml, mus, muk, frscale);

    // Assign result
    vstore3(pi, idx, P_inout);
}


kernel void projectCollisions(
                 int P_length,
                 global fpreal * P ,
                 int hitpos_length,
                 global fpreal * hitpos ,
                 int hitnml_length,
                 global float * hitnml ,
                 int hitnum_length,
                 global int * hitnum,
                 int mass_length,
                 global float *mass
#ifdef HAS_stopped
                 , int stopped_length,
                 global const int *stopped
#endif
)
{
    int idx = get_global_id(0);
    if (idx >= P_length)
        return;
    if (hitnum[idx] <= 0 || mass[idx] == 0.0f
#ifdef HAS_stopped
        || stopped[idx] & 1
#endif
        )
        return;
    fpreal3 p = vload3(idx, P);
    fpreal3 hitp = vload3(idx, hitpos);
    fpreal3 hitn = vload3f(idx, hitnml);

    fpreal C = dot(p - hitp, hitn);
    p -= min(C, (fpreal)0) * hitn;
    vstore3(p, idx, P);
}

kernel void
groundPlaneCollision(int P_length,
                 global fpreal *P_inout,
                 int pprevious_length,
                 global fpreal *pprevious_in,
                 int mass_length,
                 global float *mass,
#ifdef HAS_stopped
                 int stopped_length,
                 global const int *stopped,
#endif
#ifdef HAS_fallback
                 int fallback_length,
                 global int * fallback ,
#endif
#ifdef HAS_phase
                 int phase_length,
                 global int * phase ,
#endif
                 fpreal3 origin,
                 fpreal3 dir,
                 int pscale_length,
                 global const float *pscale,
                 float frscale,
                 float mus, float muk
#ifdef HAS_friction
                 , int friction_length,
                 global float *friction_in
#endif
#ifdef HAS_dynamicfriction
                 , int dynamicfriction_length,
                 global float *dynamicfriction_in
#endif
         )
{
    int idx = get_global_id(0);
    if (idx >= P_length)
        return;

    if (mass[idx] == 0.0f
#ifdef HAS_stopped
        || stopped[idx] & 1
#endif
        )
        return;

#ifdef HAS_friction
    mus *= friction_in[idx];
#endif
#ifdef HAS_dynamicfriction
    muk *= dynamicfriction_in[idx];
#endif

    fpreal3 p = vload3(idx, P_inout);

    fpreal rad = pscale[idx];
    dir = normalize(dir);
    fpreal dist = dot(p - origin, dir);
    if (dist >= rad)
        return;

    fpreal3  pprev = vload3(idx, pprevious_in);
    // Apply any friction, unless fluid particle
#ifdef HAS_phase
    if (!phase[idx])
#endif    
        p = applyFriction(p, pprev, (fpreal3)0, dir, mus, muk, frscale);

    // Project above plane.
    p += dir * (rad - dist);
    // Assign result
    vstore3(p, idx, P_inout);
#ifdef HAS_fallback
    fallback[idx] = 1;
#endif
}


// Extract the rotation from the deformation gradient
// provided from the restmatrix and current point positions.
// This is typicaly called at the start of a timestep with
// reasonably high iterations (e.g. 20).
// At the moment this re-loads the current restvector to warm-start
// with, but it's possible we should restart from {0, 0, 0, 1}
// each timestep for better atomicity.
kernel void
initRotations(int type_length,
              global const int *type,
              int pts_length,
              global const int *pts_index,
              global const int *pts,
              int P_length,
              global const fpreal *P,
              int restmatrix_length,
              global const float * restmatrix,
              int iterations,
              int restvector_length,
              global float * restvector
              )
{
    int idx = get_global_id(0);
    if (idx >= type_length)
        return;
    const int ctype = type[idx];
    if (!isTetARAP(ctype))
        return;
    int ptidx = pts_index[idx];

    int pt0 = pts[ptidx];
    int pt1 = pts[ptidx + 1];
    int pt2 = pts[ptidx + 2];
    int pt3 = pts[ptidx + 3];
    fpreal3 p0 = vload3(pt0, P);
    fpreal3 p1 = vload3(pt1, P);
    fpreal3 p2 = vload3(pt2, P);
    fpreal3 p3 = vload3(pt3, P);

    mat3 F, Ds, Dminv;
    // Dm^-1 is stored in restmatrix.
    mat3load(idx, restmatrix, Dminv);
    // Ds = | p0-p3 p1-p3 p2-p3 |
    mat3fromcols(p0 - p3, p1 - p3, p2 - p3, Ds);
    // F = Ds * Dm^-1
    mat3mul(Ds, Dminv, F);

    quat q = vload4f(idx, restvector);
    q = extractRotation(F, q, iterations);
    vstore4f(q, idx, restvector);
}

// This should match computeStress in
// $SHS/vex/include/pbd_constraints.h
kernel void
computeStress(
             float dt,
             int L_length,
             global float * Lin ,
             int type_length,
             global const int * type ,
             int     normalize ,
             int stress_length,
             global float * stress
            )
{
    int idx = get_global_id(0);
    if (idx >= stress_length)
        return;
    float3 L = vload3(idx, Lin);
    L.z = select(L.z, 0.0f, isTetARAPVol(type[idx]));
    stress[idx] = length(L);
    if (!normalize)
        return;
    float scale = 1.0f / select(dt * dt, dt, isNonLinearARAP(type[idx]));
    stress[idx] *= scale;
}

kernel void integrate(
                 float timeinc,
                 int P_length,
                 global float * P ,
                 int v_length,
                 global float * vin ,
                 int mass_length,
                 global float * mass
#ifdef HAS_stopped
                 , int stopped_length,
                 global int * stopped
#endif
#ifdef HAS_fallback
                 , int fallback_length,
                 global int * fallback
#endif
#ifdef HAS_pprevious
                 , int pprevious_length,
                 global float * pprevious
#endif
#ifdef HAS_plast
                 , int plast_length,
                 global float * plastin
#endif
#ifdef HAS_vprevious
                 , int vprevious_length,
                 global float * vprevious
#endif
#ifdef HAS_vlast
                 , int vlast_length,
                 global float * vlastin
#endif
#ifdef HAS_orient
                 , int orient_length,
                 global float * orient
#endif
#ifdef HAS_orientprevious
                 , int orientprevious_length,
                 global float * orientprevious
#endif
#ifdef HAS_orientlast
                 , int orientlast_length,
                 global float * orientlast
#endif
#ifdef HAS_w
                 , int w_length,
                 global float * win
#endif
#ifdef HAS_wprevious
                 , int wprevious_length,
                 global float * wprevious
#endif
#ifdef HAS_wlast
                 , int wlast_length,
                 global float * wlastin
#endif
#ifdef HAS_inertia
                 , int inertia_length,
                 global float * inertia
#endif
)
{
    int idx = get_global_id(0);
    if (idx >= P_length)
        return;

    if (mass[idx] > 0
#ifdef HAS_stopped
     && !(stopped[idx] & 1)
#endif
        )
    {
        float3 p = vload3(idx, P);
        float3 v = vload3(idx, vin);
#if defined(HAS_pprevious) && defined(HAS_plast) && defined(HAS_vprevious) && defined(HAS_vlast)
        // 2nd order
        float3 pprev = vload3(idx, pprevious);
        float3 plast = vload3(idx, plastin);
        float3 vprev = vload3(idx, vprevious);
        float3 vlast = vload3(idx, vlastin);

        // dv/dt represents force and drag from POP Solver.
        float3 dvdt = v - vprev;
        // vprevious is v at start of timestep, vlast is previous timestep.
        // BDF2 integration.
        // For some reason, moving the 3 divisor under
        // each vector value gives better precision similar to VEX.
        float3 newv = 4 * vprev/3 - vlast/3 + 2 * dvdt/3;
        float3 p2 = 4 * pprev/3 - plast/3 + 2 * timeinc * newv/3;

        // Handle fallback.
#ifdef HAS_fallback
        p = select(p2, p + timeinc * v, (int3)-fallback[idx]);
#else
        p = p2;
#endif

#else
        // 1st order
        p += timeinc * v;
#endif
        vstore3(p, idx, P);
    }

#if defined(HAS_orient) && defined(HAS_inertia)
    if (inertia[idx] > 0
#ifdef HAS_stopped
            && !(stopped[idx] & 2)
#endif
        )
    {
#ifdef HAS_w
        quat q = vload4(idx, orient);
        float3 w = vload3(idx, win);
        quat wq1 = (quat)(w, 1);
        quat wq0 = (quat)(w, 0);
#if defined(HAS_orientprevious) && defined(HAS_orientlast) && defined(HAS_wprevious) && defined(HAS_wlast)
        // 2nd order
        // dwdt represents torque and spin drag from external forces.
        float3 wprev = vload3(idx, wprevious);
        float3 wlast = vload3(idx, wlastin);
        float3 dwdt = w - wprev;
        // wprevious is w at start of timestep, wlast is previous timestep.
        // BDF2 integration.
        w = 4 * wprev/3 - wlast/3 + 2 * dwdt/3;
        quat dqdt = 0.5f * qmultiply(wq0, q);
        quat q2 = 4 * vload4(idx,orientprevious)/3 - vload4(idx, orientlast)/3 + 2 * timeinc * dqdt/3;
        // Handle fallback.
#ifdef HAS_fallback
        q = select(q2, q + (timeinc / 2) * qmultiply(wq1, q), (int4)-fallback[idx]);
#else
        q = q2;
#endif

#else
        // 1st order
        q += (timeinc / 2) * qmultiply(wq1, q);
#endif
        vstore4(normalize(q), idx, orient);
#endif
    }
#endif
}


kernel void calcV(
                 float timeinc,
                 int P_length,
                 global float * P ,
#ifdef HAS_isgrain
                 int isgrain_length,
                 global int *isgrain,
#endif
#ifdef HAS_phase
                 int phase_length,
                 global int *phase,
#endif
#ifdef HAS_hitnum
                 int hitnum_length,
                 global int *hitnum,
#endif
                 int pprevious_length,
                 global float * pprevious ,
#ifdef HAS_plast
                 int plast_length,
                 global float * plastin ,
#endif
                 int v_length,
                 global float * v,
#ifdef HAS_vprevious
                 int vprevious_length,
                 global float * vprevious,
#endif
                 int accelfallback,
                 float maxaccel,
                 int limitaccel
#ifdef HAS_fallback
                 , int fallback_length,
                 global int * fallback_inout
#endif
#ifdef HAS_orient
                 , int orient_length,
                 global float * orient
#endif
#ifdef HAS_orientprevious
                 , int orientprevious_length,
                 global float * orientprevious
#endif
#ifdef HAS_orientlast
                 , int orientlast_length,
                 global float * orientlast
#endif
#ifdef HAS_w
                 , int w_length,
                 global float * wout
#endif
)
{
    int idx = get_global_id(0);
    if (idx >= v_length)
        return;
    float3 p = vload3(idx, P);
    float3 pprev = vload3(idx, pprevious);

#ifdef HAS_plast
    // 2nd order
    float3 plast = vload3(idx, plastin);
    float3 vel = ((2 * p + (p + plast)) - 4 * pprev) / (2 * timeinc);

#if defined(HAS_fallback) && defined(HAS_vprevious)
    int fallback = 0;
    if (accelfallback)
    {
        // TODO - this isn't right for fluids since nothing clears fallback
        // if collisions are off.
        fallback = fallback_inout[idx];
        float3 accel = (vel - vload3(idx, vprevious)) / timeinc;
        fallback = (length(accel) > maxaccel) && (fallback
#ifdef HAS_hitnum
                                                  || hitnum[idx]
#endif                                                  
#ifdef HAS_isgrain
                                                  || isgrain[idx]
#endif                                                  
#ifdef HAS_phase
                                                  || (phase[idx] > 0)
#endif                                                  
                                                  );
        // Update the fallback attribute.
        fallback_inout[idx] = fallback;
        vel = select(vel, (p - pprev) / timeinc, (int3)-fallback);
    }
#endif

#else
    // 1st order
    float3 vel = (p - pprev) / timeinc;
#endif

    // Limit acceleration if necessary
    if (limitaccel)
    {
        float3 origv = vload3(idx, v);
        float3 dv = vel - origv;
        float accel = length(dv);
        maxaccel *= timeinc;
        if (accel > maxaccel)
            vel = origv + dv * maxaccel / accel;
    }

    vstore3(vel, idx, v);

#if defined(HAS_orient) && defined(HAS_orientprevious) && defined(HAS_w)

    quat q = vload4(idx, orient);
    quat qconj = q * (float4)(-1, -1, -1, 1);
    quat qprev = vload4(idx, orientprevious);
#ifdef HAS_orientlast
    quat qlast = vload4(idx, orientlast);
    // The result of solving BDF2 for w:
    // q = 4/3 * q_prev - 1/3 * q_last + 2/3 * t * dqdt
    // q = 4/3 * q_prev - 1/3 * q_last + 1/3 * t * w * q
    // w = (3 * q - 4 * q_prev + q_last) * conj(q) / t
    q = 3 * q - 4 * qprev + qlast;
    q = qmultiply(q, qconj);
    float3 w = q.xyz / timeinc;

#ifdef HAS_fallback
    q = vload4(idx, orient) - qprev;
    q = qmultiply(q, qconj);
    float3 w1 = q.xyz * (2 / timeinc);
    w = select(w, w1, (int3)-fallback);
#endif

#else
    // The result of solving BDF1 for w:
    // q = q_prev + t * dqdt
    // q = q_prev + t * 1/2 * w * q
    // w = (q - qprev) * conj(q) * (2 / t)
    q -= qprev;
    q = qmultiply(q, qconj);
    float3 w = q.xyz * (2 / timeinc);
#endif
    vstore3(w, idx, wout);
#endif
}

// Mostly stolen from mpm.cl.
kernel void
integrateForces(
        float timeinc,
        int P_length,
        global float* restrict P,
        int v_length,
        global float* restrict v,
        int mass_length,
        global float* restrict mass,
#ifdef HAS_targetv
        int targetv_length,
        global float* restrict targetv,
#endif
#ifdef HAS_airresist
        int airresist_length,
        global float* restrict airresist,
#endif
#ifdef HAS_force
        int force_length,
        global float* restrict force,
#endif
        int ignoremass)
{
    int idx = get_global_id(0);
    if (idx >= v_length)
        return;

    float3 p_v = vload3(idx, v);

    float p_mass = 1.0f;
    if (!ignoremass)
        p_mass = mass[idx];

    // air resistance
    float p_airresist = 0.0f;
#ifdef HAS_airresist
    p_airresist += airresist[idx];
#endif

    float3 p_targetv = (float3)(0.0f);
#ifdef HAS_targetv
    p_targetv += vload3(idx, targetv);
#endif

    if (p_airresist > 0.0f)
    {
        if (!ignoremass)
            p_airresist /= p_mass;

        p_v -= p_targetv; // go to relative vel
        float scale = 1.0f / (1.0f + p_airresist * length(p_v) * timeinc); // quadratic only
        p_v *= scale;
        p_v += p_targetv; // restore world vel
    }

#ifdef HAS_force
    // integrate forces
    float3 accel = vload3(idx, force);
    if (!ignoremass)
        accel /= p_mass;
    p_v += timeinc * accel;
#endif

    vstore3(p_v, idx, v);
}
