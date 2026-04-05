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
 * NAME:    pbd_constraints.h
 *
 * COMMENTS:
 *    Functions for creating and manipulating PBD constraints.
 */

#ifndef __pbd_constraints_h
#define __pbd_constraints_h

#include <math.h>

int
compareIntArrays(const int a[], b[])
{
    if (len(a) != len(b))
        return 0;
    foreach(int idx; int i; a)
    {
        if (i != b[idx])
            return 0;
    }
    return 1;
}

int []
sortRemoveDuplicates(const int a[])
{
    int b[];
    resize(b, len(a));
    int sorted[] = sort(a);
    int last = 0, first = 1;
    int idx = 0;
    foreach(int c; sorted)
    {
        if (!first && c == last)
            continue;
        b[idx] = c;
        idx++;
        first = 0;
        last = c;
    }
    resize(b, idx);
    return b;
}

int []
calcConstraintTopology(const int constraintgeo, pointgeo)
{
    int topo[] = attribdataid(constraintgeo, "meta", "topology");
    append(topo, attribdataid(constraintgeo, "meta", "primitivelist"));
    append(topo, attribdataid(pointgeo, 'point', 'id'));
    append(topo, attribdataid(pointgeo, 'point', 'weld'));
    append(topo, attribdataid(pointgeo, 'point', 'branchweld'));
    return topo;
}

string
constraintAlias(const string type)
{
    // Handle alias conversion.
    if (type == "attach")
        return "pin";
    if (type == "stitch" || type == "branchstitch")
        return "distance";
    if (type == "attachnormal")
        return "distanceline";
    return type;
}

int
constraintHash(const string type)
{
    return random_shash(constraintAlias(type));
}

int
isTriARAP(const string type)
{
    return type == "triarap" || type == "triarapnl" || type == "triarapnorm";
}

int isTetARAPVol(const string type)
{
    return type == "tetarapvol" || type == "tetarapnlvol" || type == "tetarapnormvol";
}

int
isTetARAP(const string type)
{
    return type == "tetarap" || type == "tetarapnl" || type == "tetarapnorm" || isTetARAPVol(type);
}

int
isNonLinearARAP(const string type)
{
    return type == "triarapnl" || type == "tetarapnl" || type == "tetarapnlvol";
}

int
isTetFiber(const string type)
{
    return type == "tetfiber" || type == "tetfibernorm";
}

float
computeDistanceRestLength(const int geo; const int p0, p1)
{
    return distance(vector(point(geo, "P", p0)), vector(point(geo, "P", p1)));
}

void
createDistanceConstraint(const int geo; const int ptnum; const string srcgrp;
                         const int outgeo; const string outgrp)
{
    int nbrs[] = neighbours(geo, ptnum);
    foreach(int n; nbrs)
    {
        if (n <= ptnum || !inpointgroup(geo, srcgrp, n))
            continue;
        int prim = addprim(outgeo, "polyline", array(ptnum, n));
        setprimgroup(outgeo, outgrp, prim, 1);
        setprimattrib(outgeo, "restlength", prim, computeDistanceRestLength(geo, ptnum, n));
    }
}

void
createAttachConstraint(const int geo, attachgeo; const int ptnum, srcidx; const string attachgrp;
                       const int useclosestpt, useclosestprim, maxdistcheck; const float maxdist;
                       const int outgeo; const string outgrp)
{
    int targetprim = -1, targetpt = -1;
    vector targetuv, targetpos;
    float dist;

    vector pos = point(geo, "P", ptnum);

    if (useclosestpt)
    {
        if (useclosestprim)
        {
            if (maxdistcheck)
                dist = xyzdist(attachgeo, attachgrp, pos, targetprim, targetuv, maxdist);
            else
                dist = xyzdist(attachgeo, attachgrp, pos, targetprim, targetuv);
        }
        else
        {
            if (maxdistcheck)
                targetpt = nearpoint(attachgeo, attachgrp, pos, maxdist);
            else
                targetpt = nearpoint(attachgeo, attachgrp, pos);
        }
    }
    else
    {
        int tgtpts[] = expandpointgroup(attachgeo, attachgrp);
        if (srcidx < len(tgtpts))
            targetpt = tgtpts[srcidx];
    }

    if (targetprim < 0 && targetpt < 0)
        return;

    int prim = addprim(outgeo, "sphere", ptnum);
    matrix3 t = 0.01;
    setprimintrinsic(outgeo, "transform", prim, t);
    setprimgroup(outgeo, outgrp, prim, 1);

    if (targetprim >= 0)
    {
        setprimattrib(outgeo, "target_prim", prim, targetprim);
        setprimattrib(outgeo, "target_uv", prim, targetuv);
        targetpos = primuv(attachgeo, "P", targetprim, targetuv);
    }
    else if (targetpt >= 0)
    {
        setprimattrib(outgeo, "target_pt", prim, targetpt);
        targetpos = point(attachgeo, "P", targetpt);
        dist = distance(pos, targetpos);
    }

    setprimattrib(outgeo, "restvector", prim, (vector4)targetpos);
    setprimattrib(outgeo, "restlength", prim, dist);
}

// Returns projection of pos onto the line, assuming dir is normalized.
vector
projectToLine(const vector pos, orig, dir)
{
    return orig + dir * dot(pos - orig, dir);
}


// Returns the matrix to transform from triangle to world space.
// This MUST match toTriangleSpaceXform in pbd_constraints.cl.
matrix3
fromTriangleSpaceXform(const vector p0, p1, p2)
{
    vector e0 = p1 - p0;
    vector e1 = p2 - p0;
    vector e2 = cross(e1, e0);
    vector z  = normalize(e2);
    vector y = normalize(cross(e0, z));
    vector x = cross(y, z);
    return set(x, y, z);
}

// Rotate the vector from triangle space using the first three points of the polygon
// specified in prim.  Invert from world space to triangle if directed.
vector
fromTriangleSpace(const string path; const int prim; const vector dir; const int invert)
{
    int pts[] = primpoints(path, prim);
    vector p0 = point(path, "P", pts[0]);
    vector p1 = point(path, "P", pts[1]);
    vector p2 = point(path, "P", pts[2]);
    matrix3 xform = fromTriangleSpaceXform(p0, p1, p2);
    if (invert)
        return dir * transpose(xform);
    return dir * xform;
}

float
computeDistanceLineRestLength(const int geo; const int pt0; const vector orig, indir;
                              const string path; const int prim)
{
    vector pos = point(geo, "P", pt0);
    // Transform from triangle space if needed.
    vector dir = indir;
    if (path != "" && prim >=0)
        dir = fromTriangleSpace(path, prim, dir, 0);
    return distance(pos, projectToLine(pos, orig, dir));
}

// Create an AttachNormal constraint that stores the line direction in restdir
// in triangle space.
// NOTE: Possibly more convenient would be to store restdir in world space, but also
// store a 3x3 matrix that would transform it back to triangle space.  Then on each
// timestep we would update restdir from the current target geometry with:
// v@restdir *= 3@restmatrix * curmatrix;
// where curmatrix is the current triangle-to-world transform, then:
// 3@restmatrix = invert(curmatrix);
// But this would require storing an extra matrix per constraint.
int
createAttachNormalConstraint(const int geo; const int ptnum;
                             const string targetpath;
                             const int targetprim; const vector targetuv;
                             const int outgeo; const string outgrp)
{
    vector pos = point(geo, "P", ptnum);
    // Orig is point on target attach geometry.
    vector orig = primuv(targetpath, "P", targetprim, targetuv);
    // Dir is ray from orig to point position.  This will be close to the normal
    // for convex geometry, but might vary quite a bit for non-convex. We use
    // this instead of the normal so the rest state is stable in the latter case.
    vector dir = normalize(pos - orig);

    // Transform to triangle space; the solver will transform back to world space
    // based on current target geometry point positions.
    // Only polygons.
    if (primintrinsic(targetpath, "typeid", targetprim) != 1)
        return -1;
    // With at least three points to form a basis.
    if (len(primpoints(targetpath, targetprim)) < 3)
        return -1;

    dir = fromTriangleSpace(targetpath, targetprim, dir, /*invert=*/ 1);

    int prim = addprim(outgeo, "sphere", ptnum);
    matrix3 t = 0.01;
    setprimintrinsic(outgeo, "transform", prim, t);
    setprimattrib(outgeo, "target_prim", prim, targetprim);
    setprimattrib(outgeo, "target_uv", prim, targetuv);
    setprimattrib(outgeo, "target_path", prim, targetpath);
    // Restlength is always zero to constraint directly to line to closest point.
    setprimattrib(outgeo, "restlength", prim, 0.0f);
    setprimattrib(outgeo, "restvector", prim, (vector4) orig);
    setprimattrib(outgeo, "restdir", prim, dir);
    setprimgroup(outgeo, outgrp, prim, 1);

    return prim;
}

// Compute the target position directly from the uv coords and the primitive's
// point positions.  (We can't use primuv because we don't know the primitive
// number, and don't want to try to track it.)
vector
pointPrimTargetPos(const int geo; const int pts[]; const float u, v; const string pattrib)
{
    vector p0 = point(geo, pattrib, pts[0]);
    vector p1 = point(geo, pattrib, pts[1]);
    // Only line segments, triangles and quads.
    int npts = len(pts);
    if (npts == 2)
    {
        float w0 = (1 - u);
        float w1 = u;
        return w0 * p0 + w1 * p1;
    }
    else if (npts == 3)
    {
        vector p2 = point(geo, pattrib, pts[2]);
        float w0 = (1 - u - v);
        float w1 = u;
        float w2 = v;
        return w0 * p0 + w1 * p1 + w2 * p2;
    }
    else if (npts == 4)
    {
        vector p2 = point(geo, pattrib, pts[2]);
        vector p3 = point(geo, pattrib, pts[3]);
        float u1 = 1 - u;
        float v1 = 1 - v;
        float w0 = (u1 * v1);
        float w1 = (u1 * v);
        float w2 = (u * v);
        float w3 = (u * v1);
        return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
    }
    return 0;
}

vector
pointPrimTargetPos(const int geo; const int pts[]; const float u, v)
{
    return pointPrimTargetPos(geo, pts, u, v, "P");
}

float
computePointPrimRestLength(const int geo; const int pt; const int tgtpts[]; const vector4 restvector)
{
    int npts = len(tgtpts);
    // Only line segments, triangles, and quads.
    if (npts < 2 || npts > 4)
        return 0;
    vector pos = point(geo, "P", pt);
    vector tgtpos = pointPrimTargetPos(geo, tgtpts, restvector.x, restvector.y);
    return distance(pos, tgtpos);
}

// Map from uv's over an entire polyline to a single parametric value along the
// line segment in which the uv point falls.  Targetpts should contain all the line
// points as input, but will have the segment points as output.
void
remapPolyLineUVs(const string geo; const int targetprim; int targetpts[]; vector targetuv)
{
    vector2 uv = set(targetuv.x, 0);
    // Convert to segment based mapping
    uv = primuvconvert(geo, uv, targetprim, PRIMUV_UNIT_TO_REAL);
    int pt0 = floor(uv.x);
    // If on last point we need to back up 1.
    if (pt0 >= len(targetpts) - 1)
        pt0 -= 1;
    uv.x -= pt0;
    targetpts = array(targetpts[pt0], targetpts[pt0 + 1]);
    targetuv = set(uv.x, 0, 0);
}

void
createStitchConstraint(const int geo; const int ptnum, srcidx; const string targetgrp;
                       const int useclosestpt, useclosestprim, maxdistcheck; const float maxdist;
                       const int outgeo; const string outgrp)
{
    int targetprim = -1, targetpt = -1;
    vector targetuv, targetpos;
    float dist;
    int tgtpts[];
    vector pos = point(geo, "P", ptnum);
    if (useclosestpt)
    {
        if (useclosestprim)
        {
            if (maxdistcheck)
                dist = xyzdist(geo, targetgrp, pos, targetprim, targetuv, maxdist);
            else
                dist = xyzdist(geo, targetgrp, pos, targetprim, targetuv);
            if (targetprim < 0)
                return;
            if (primintrinsic(geo, "typeid", targetprim) != 1)
                return;
            tgtpts = primpoints(geo, targetprim);
            if (!primintrinsic(geo, "closed", targetprim)) // polyline, remap to line segment.
                remapPolyLineUVs(sprintf("opinput:%d", geo), targetprim, tgtpts, targetuv);
            // Line segment, triangle, or quad polygons only.
            int npts = len(tgtpts);
            if (npts < 2 || npts > 4)
                return;
        }
        else
        {
            if (maxdistcheck)
                targetpt = nearpoint(geo, targetgrp, pos, maxdist);
            else
                targetpt = nearpoint(geo, targetgrp, pos);
        }
    }
    else
    {
        tgtpts = expandpointgroup(geo, targetgrp);
        if (srcidx < len(tgtpts))
            targetpt = tgtpts[srcidx];
    }
    if (targetprim < 0 && targetpt < 0)
        return;

    int prim = -1;
    if (targetprim >= 0)
    {
        int pts[];
        append(pts, ptnum);
        append(pts, tgtpts);
        vector4 restvec = set(targetuv.x, targetuv.y, 0, 0);
        prim = addprim(outgeo, "polyline", pts);
        // Just to ensure these match exactly, even though we already know dist.
        dist = computePointPrimRestLength(geo, ptnum, tgtpts, restvec);
        setprimattrib(outgeo, "restlength", prim, dist);
        setprimattrib(outgeo, "restvector", prim, restvec);
        setprimattrib(outgeo, "type", prim, "ptprim");
    }
    else if (targetpt >= 0)
    {
        prim = addprim(outgeo, "polyline", array(ptnum, targetpt));
        setprimattrib(outgeo, "restlength", prim, computeDistanceRestLength(geo, ptnum, targetpt));
        setprimattrib(outgeo, "type", prim, "stitch");
    }
    setprimgroup(outgeo, outgrp, prim, 1);
}

int
oppositepoint(const int geo; const int hedge)
{
    return hedge_dstpoint(geo, hedge_next(geo, hedge));
}


int computeDihedralRestLength(const int geo;
                              const int pt0, pt1, pt2, pt3;
                              float restlength)
{
    vector p0 = point(geo, "P", pt0);
    vector p1 = point(geo, "P", pt1);
    vector p2 = point(geo, "P", pt2);
    vector p3 = point(geo, "P", pt3);

    vector e = p3 - p2;
    float elen = length(e);
    if (elen < 1e-6)
        return 0;
    float invElen = 1 / elen;

    // Find initial rest angle.
    vector n1 = cross(p3 - p0, p2 - p0);
    vector n2 = cross(p2 - p1, p3 - p1);
    float d = dot(normalize(n1), normalize(n2));
    d = clamp(d, -1, 1);
    float phi = acos(d);
    // We want to xpress phi a -PI..PI
    if (dot(cross(n1, n2), e) < 0)
        phi = -phi;
    restlength = degrees(phi);
    return 1;
}


void
createDihedralConstraint(const int geo; const int ptnum; const string srcgrp;
                         const int outgeo; const string outgrp)
{
    int prims[] = pointprims(geo, ptnum);
    foreach(int prim; prims)
    {
        int primpts[] = primpoints(geo, prim);
        // Process triangles where this point is minimum point number,
        // and all points are in src group.
        if (len(primpts) != 3)
            continue;
        if (ptnum != min(primpts))
            continue;
        if (!inpointgroup(geo, srcgrp, primpts[1]) ||
            !inpointgroup(geo, srcgrp, primpts[2]))
            continue;

        int starthedge = primhedge(geo, prim);

        if (starthedge < 0)
            return;

        // Ignore open curves as the hedge function won't give us
        // the setup we expect with them.
        if (primintrinsic(geo, 'closed', prim) == 0)
            return;

        int h = starthedge;
        int p = prim;
        int hasgrp = strlen(outgrp) > 0;
        while (1)
        {
            // Giving half edge h, add a bend polygon.
            int oh = hedge_nextequiv(geo, h);
           // Skip boundary edges, invalid hedges, and non-manifold
           // edges
            if (h != oh && oh >= 0 && h == hedge_nextequiv(geo, oh))
            {
                int op = hedge_prim(geo, oh);
                // Always build in ascending direction.
                if (op >= 0 && p < op)
                {
                    int pt0 = oppositepoint(geo, h);
                    int pt1 = oppositepoint(geo, oh);
                    int pt2 = hedge_srcpoint(geo, h);
                    int pt3 = hedge_dstpoint(geo, h);
                    if (inpointgroup(geo, srcgrp, pt0) &&
                        inpointgroup(geo, srcgrp, pt1) &&
                        inpointgroup(geo, srcgrp, pt2) &&
                        inpointgroup(geo, srcgrp, pt3))
                    {
                        float restlength;
                        if (computeDihedralRestLength(geo, pt0, pt1, pt2, pt3,
                                                      restlength))
                        {
                            int newprim = addprim(outgeo, 'polyline', array(pt0, pt1, pt2, pt3));
                            setprimattrib(outgeo, "restlength", newprim, restlength);
                            if (hasgrp)
                                setprimgroup(outgeo, outgrp, newprim, 1);
                        }
                    }
                }
            }

            int nh = hedge_next(geo, h);

            // Stop the loop when we complete the polygon
            // Closed polygons won't complete.
            if (nh == starthedge || nh < 0)
                break;
            h = nh;
        }
    }
}

void
createDihedralConstraintFromNewlyWeldedPrimitives(const int geo; const int oldgeo; const int prim;
                         const int outgeo; const string outgrp)
{
    int primpts[] = primpoints(geo, prim);

    if (len(primpts) != 3)
	return;

    int starthedge = primhedge(geo, prim);

    if (starthedge < 0)
	return;

    // Ignore open curves as the hedge function won't give us
    // the setup we expect with them.
    if (primintrinsic(geo, 'closed', prim) == 0)
	return;

    int h = starthedge;
    int p = prim;
    int hasgrp = strlen(outgrp) > 0;
    while (1)
    {
	// Giving half edge h, add a bend polygon.
	int oh = hedge_nextequiv(geo, h);
        // Skip boundary edges, invalid hedges, and non-manifold
        // edges
	if (h != oh && oh >= 0 && h == hedge_nextequiv(geo, oh))
	{
	    int op = hedge_prim(geo, oh);
	    // Always build in ascending direction.
	    if (op >= 0 && p < op)
	    {
		int pt0 = oppositepoint(geo, h);
		int pt1 = oppositepoint(geo, oh);
		int pt2 = hedge_srcpoint(geo, h);
		int pt3 = hedge_dstpoint(geo, h);

		// See if we became welded.  Because hedges are
		// linear vertex numbers, these will match in the
		// two and we can find the corresponding half edge
		// in the other geometry directly.
		// So, if our h & oh don't have matching points
		// in the old geo, we know we are a newly welded edge.
		if (hedge_srcpoint(oldgeo, h) != hedge_dstpoint(oldgeo, oh) ||
		    hedge_dstpoint(oldgeo, h) != hedge_srcpoint(oldgeo, oh))
		{
		    float restlength;
		    if (computeDihedralRestLength(geo, pt0, pt1, pt2, pt3,
						   restlength))
                    {
                        int newprim = addprim(outgeo, 'polyline', array(pt0, pt1, pt2, pt3));
                        setprimattrib(outgeo, "restlength", newprim, restlength);
                        if (hasgrp)
                            setprimgroup(outgeo, outgrp, newprim, 1);
                    }
		}
	    }
	}

	int nh = hedge_next(geo, h);

	// Stop the loop when we complete the polygon
	// Closed polygons won't complete.
	if (nh == starthedge || nh < 0)
	    break;
	h = nh;
    }
}

void
computeOrientRodlengths(const int geo; const int primnum; const string srcgrp;
                        const int outgeo)
{
    // Ignore anything but open polylines.
    if (primintrinsic(geo, "typename", primnum) != "Poly" ||
        primintrinsic(geo, "closed", primnum) == 1)
        return;
    // Only test group if it's not all points.
    int hasgrp = npointsgroup(geo, srcgrp) < npoints(geo);
    // Check if the points have an incoming orient attribute. If os, we assume
    // that it provides a stable basis in which to calculate our rod-aligned orients.
    int hasporient = haspointattrib(geo, "orient");
    vector from = {0, 0, 1};
    // Iterate over pts in vertex order.
    int pts[] = primpoints(geo, primnum);
    int npts = len(pts);
    vector4 orients[];
    float rodlens[];
    resize(orients, npts - 1);
    resize(rodlens, npts - 1);

    int lastpt = pts[npts - 1];
    int loop = 0;
    for(int i=0; i < npts - 1; i++)
    {
        vector d = point(geo, "P", pts[i + 1]) - point(geo, "P", pts[i]);
        vector to = normalize(d);
        if (hasporient)
        {
            // Transform vectors back to rest orientation.
            vector4 porient = point(geo, "orient", pts[i]);
            to = qrotate(qinvert(porient), to);
            // If to is already closely aligned along the orient direction,
            // we can end up with flipping back and forth between very small
            // negative and positive values in the x and y components across
            // frames. (This happens often when a second vellumconstraints
            // tries to "recompute" the exact same orientation as the last one).
            // This flipping leads to flipping in the output orientation, so 
            // we remove the x,y components in this case.
            if (abs(abs(to.z) - 1.0) < 1e-6)
                to = {0, 0, 1} * sign(to.z);
        }
        vector4 dq = dihedral(from, to);
        if (i == 0)
            orients[i] = dq;
        else
            orients[i] = qmultiply(dq, orients[i-1]);
        rodlens[i] = length(d);
        from = to;
        // Check if this is a loop.
        loop |= (pts[i] == lastpt);
    }

    for(int i=0; i < npts - 1; i++)
    {
        if (hasporient)
        {
            // Rotate new orientation back to current world orientation.
            vector4 porient = point(geo, "orient", pts[i]);
            orients[i] = qmultiply(porient, orients[i]);
        }

        // Make sure to do the above rotation in the orients array
        // before possibly skipping this point based on group membership,
        // otherwise if we then grab the rotation for the last point from
        // the previous vertex, it could be incorrect if the previous
        // vertex wasn't in srcgrp. (Exiting too early caused bug 97655.)
        if (hasgrp && !inpointgroup(geo, srcgrp, pts[i]))
            continue;

        // Set new orientation, possibly overwriting input orientation.
        setpointattrib(geoself(), "orient", pts[i], orients[i]);

        float inertia = point(geo, "inertia", pts[i]);
        // Don't overwrite pinned inertia
        if (inertia == 0.0)
            continue;
        setpointattrib(geoself(), "inertia", pts[i], rodlens[i]);
    }
    // Set inertia and orient for last point if not already seen as part of a loop.
    if (npts > 1 && inpointgroup(geo, srcgrp, lastpt) && !loop)
    {
        float inertia = point(geo, "inertia", lastpt);
        // Don't overwrite pinned inertia
        if (inertia != 0.0)
            setpointattrib(geoself(), "inertia", lastpt, rodlens[npts - 2]);
        // Copy previous orient so final bend/twist rest relative orient is
        // identity, which reduces flipping when final vertex orientation is
        // otherwise unconstrained.
        setpointattrib(geoself(), "orient", lastpt, orients[npts - 2]);
    }
}

vector4
computeBendTwistRestVector(const int geo; const int pt0, pt1)
{
    // Discrete Darbeaux vector is just closest rotation from
    // first orientation to second.
    vector4 q0conj = point(geo, "orient", pt0) * {-1, -1, -1, 1};
    vector4 q1 = point(geo, "orient", pt1);
    vector4 restDarbeaux = qmultiply(q0conj, q1);
    vector4 omegaplus = restDarbeaux + {0, 0, 0, 1};
    vector4 omegaminus = restDarbeaux - {0, 0, 0, 1};
    if (dot(omegaminus, omegaminus) > dot(omegaplus, omegaplus))
        restDarbeaux *= -1;
    return restDarbeaux;
}

// Create a single constraint of the specified type along the (first and only) output edge from
// the provided point, assuming we're dealing with points on polylines.
void
createOutEdgeConstraint(const int geo; const int ptnum; const string srcgrp; const string type;
                     const int outgeo; const string outgrp)
{
    int nbrs[] = neighbours(geo, ptnum);
    foreach(int n; nbrs)
    {
        // Only create if the in-group neighbor is the destination of an edge.
        if (!inpointgroup(geo, srcgrp, n) || pointhedge(geo, ptnum, n) < 0)
            continue;
        int prim = addprim(outgeo, "polyline", array(ptnum, n));
        if (type == "stretchshear")
            setprimattrib(outgeo, "restlength", prim, computeDistanceRestLength(geo, ptnum, n));
        else
            setprimattrib(outgeo, "restvector", prim, computeBendTwistRestVector(geo, ptnum, n));
        setprimgroup(outgeo, outgrp, prim, 1);
        // Only one constraint per point.
        return;
    }
}

void
createStretchShearConstraint(const int geo; const int ptnum; const string srcgrp;
                         const int outgeo; const string outgrp)
{
    createOutEdgeConstraint(geo, ptnum, srcgrp, "stretchshear", outgeo, outgrp);
}

void
createBendTwistConstraint(const int geo; const int ptnum; const string srcgrp;
                          const int outgeo; const string outgrp)
{
    createOutEdgeConstraint(geo, ptnum, srcgrp, "bendtwist", outgeo, outgrp);
}

int
findBranchBendPoints(const int geo; const int ptnum; const int weld;
                     int pt0, pt1)
{
    int nbrs[] = neighbours(geo, weld);
    foreach(int n; nbrs)
    {
        // Check if there's an edge pointing from n to the weld.
        if (pointhedge(geo, n, weld) >= 0)
        {
            // We need to create a constraint from n to this point also.
            pt0 = n;
            pt1 = ptnum;
            return 1;
        }
    }
    // If only one neighbor and there's no edge with weld as the destination,
    // then punt and make the constraint between the weld and ptnum
    if (len(nbrs) == 1)
    {
        pt0 = weld;
        pt1 = ptnum;
        return 1;
    }
    return 0;
}

void
createBranchWeldConstraints(const int geo; const int ptnum;
                            const string srcgrp; const float maxbranchangle;
                            const int outgeo; const string outbendgrp)
{
    int weld = point(geo, "branchweld", ptnum);
    if (weld < 0 || !inpointgroup(geo, srcgrp, weld))
        return;

    // Stitch points together with stiff distance constraint.
    // TODO - do we need to parameterize this somehow?
    int cprim = addprim(outgeo, "polyline", array(weld, ptnum));
    setprimattrib(outgeo, "restlength", cprim, 0);
    setprimattrib(outgeo, "stiffness", cprim, 1e20);
    setprimattrib(outgeo, "dampingratio", cprim, 0.001);
    setprimattrib(outgeo, "type", cprim, "branchstitch");

    // Find the appropriate points for creating a bend/twist
    // constraint on the geometry.
    int pt0, pt1;
    if (!findBranchBendPoints(geo, ptnum, weld, pt0, pt1))
        return;
    // Compute Darbeaux vector for rotation.
    vector4 restvec = computeBendTwistRestVector(geo, pt0, pt1);
    // Skip creation if greater than max branching angle.
    float maxang = radians(maxbranchangle);
    // Compare against axis/angle represenation of quaternion.
    if (length(qconvert(restvec)) > maxang)
        return;
    cprim = addprim(outgeo, "polyline", array(pt0, pt1));
    setprimattrib(outgeo, "restvector", cprim, restvec);
    setprimgroup(outgeo, outbendgrp, cprim, 1);
}

// Called with the primitive representing the branchweld stitch/distance constraint.
void
removeBranchWeldConstraints(const int congeo, ptgeo; const int primnum; const int outgeo)
{
    int pts[] = primpoints(congeo, primnum);
    // If pt is not still branchwelded to weld, we need to delete constraints.
    int weld = pts[0];
    int pt = pts[1];
    int bw = point(ptgeo, "branchweld", pt);
    if (bw >= 0)
        bw = idtopoint(ptgeo, bw);
    // Still welded, nothing to do.
    if (weld == bw)
        return;
    // Remove this constraint stitching them together.
    removeprim(outgeo, primnum, 0);

    // Look for bend/twist constraints joining the bend points,
    // assume it was created for branchwelding and remove.
    int bendpt0, bendpt1;
    if (!findBranchBendPoints(ptgeo, pt, weld, bendpt0, bendpt1))
        return;
    int prims[] = pointprims(congeo, pt);
    foreach(int prim; prims)
    {
        string type = prim(congeo, "type", prim);
        if (type != "bendtwist")
            continue;
        int primpts[] = primpoints(congeo, prim);
        if (bendpt0 == primpts[0] && bendpt1 == primpts[1])
            removeprim(outgeo, prim, 0);
    }
}

// Returns array of neighboring points on triangles that can
// be iterated through to calculate volume and gradients
// using cross products.
int []
findVolumePoints(const int geo; const int ptnum)
{
    int prims[] = pointprims(geo, ptnum);
    int volpts[];
    foreach(int prim; prims)
    {
        int pts[] = primpoints(geo, prim);
        // Only triangles.
        if (len(pts) != 3)
            continue;
        int ppos = find(pts, ptnum);
        vector n = 0;
        if (ppos == 0)
            append(volpts, array(pts[2], pts[1]));
        if (ppos == 1)
            append(volpts, array(pts[0], pts[2]));
        if (ppos == 2)
            append(volpts, array(pts[1], pts[0]));
    }
    return volpts;
}

// Compute the volume for all triangles specified in the
// volumepts array for each point.  The volume is only
// counted if the current points is the lowest index of the tri.
float
computeVolume(const int geo; const int ptnum; const int volumepts[])
{
    int ntris = len(volumepts) / 2;
    float volume = 0;
    vector pos = point(geo, "P", ptnum);
    for(int i = 0; i < ntris; i++)
    {
        int pt1 = volumepts[i * 2];
        int pt2 = volumepts[i * 2 + 1];
        if (ptnum < pt1 && ptnum < pt2)
            volume += dot(pos,
                          cross(point(geo, "P", pt1), point(geo, "P", pt2)));
    }
    return volume / 6;
}


float
computePressureRestLength(const int geo; const int inpts[]; const string mode)
{

    float restvol = 0;
    // Remove any duplicates that can arise from welding.
    int pts[] = sortRemoveDuplicates(inpts);
    int volumepts[];
    foreach(int pt; pts)
    {
        if (mode == "volume")
        {
            restvol += point(geo, "volume", pt);
        }
        else
        {
            if (mode == "volumepts")
                volumepts = point(geo, "volumepts", pt);
            else // Use topology.
                volumepts = findVolumePoints(geo, pt);
            restvol += computeVolume(geo, pt, volumepts);
        }
    }
    return restvol;
}

float
computePressureRestLength(const int geo; const int inpts[])
{
    return computePressureRestLength(geo, inpts, "volume");
}

// Create a pressure constraint over the provided group of points.
void
createPressureConstraint(const int geo; const string srcgrp;
                       const int outgeo; const string outgrp)
{
    int pts[] = expandpointgroup(geo, srcgrp);
    // Don't create empty constraint.
    if (len(pts) == 0)
        return;

    int prim = addprim(outgeo, "polyline", pts);
    float restvol = computePressureRestLength(geo, pts, "volume");
    setprimattrib(outgeo, "restlength", prim, restvol);
    setprimgroup(outgeo, outgrp, prim, 1);
}

// Create a shapematch constraint over the provided group of points.
void
createShapeMatchConstraint(const int geo; const string srcgrp;
                           const int outgeo; const string outgrp)
{
    int pts[] = expandpointgroup(geo, srcgrp);
    // Don't create empty constraint.
    if (len(pts) == 0)
        return;
    // Set rest for each point from current geo position.
    foreach(int pt; pts)
    {
        vector pos = point(geo, "P", pt);
        setpointattrib(outgeo, "rest", pt, pos);
    }
    int prim = addprim(outgeo, "polyline", pts);
    setprimgroup(outgeo, outgrp, prim, 1);
}

float
computeTetVolumeRestLength(const int geo; const int pt0, pt1, pt2, pt3)
{
    vector p0 = point(geo, "P", pt0);
    vector p1 = point(geo, "P", pt1);
    vector p2 = point(geo, "P", pt2);
    vector p3 = point(geo, "P", pt3);
    return dot(cross(p0 - p1, p0 - p2), p0 - p3) / 6;
}

void
createTetVolumeConstraint(const int geo; const int ptnum; const string srcgrp;
                          const int outgeo; const string outgrp)
{
    int prims[] = pointprims(geo, ptnum);
    foreach(int prim; prims)
    {
        // Process tets where this point is minimum point number,
        // and all points are in src group.
        if(primintrinsic(geo, "typeid", prim) != 21)
            continue;
        int pts[] = primpoints(geo, prim);
        if (len(pts) != 4 ||
            ptnum != min(pts) ||
            !inpointgroup(geo, srcgrp, pts[1]) ||
            !inpointgroup(geo, srcgrp, pts[2]) ||
            !inpointgroup(geo, srcgrp, pts[3]))
            continue;

        int cprim = addprim(outgeo, "tet", pts);
        float volume = computeTetVolumeRestLength(geo, pts[0], pts[1], pts[2], pts[3]);
        setprimattrib(outgeo, "restlength", cprim, volume);
        setprimgroup(outgeo, outgrp, cprim, 1);
    }
}

float
computeAngleRestLength(const int geo;
                       const int pt0, pt1, pt2)
{
    vector p0 = point(geo, "P", pt0);
    vector p1 = point(geo, "P", pt1);
    vector p2 = point(geo, "P", pt2);

    vector n1 = normalize(p1 - p0);
    vector n2 = normalize(p2 - p1);
    return degrees(acos(clamp(dot(n1, n2), -1, 1)));
}


// Compute distance from vertex to centroid for triangle
float
computeVertexCentroidDistance(const vector p0, p1, p2)
{
    vector c = (p0 + p1 + p2) / 3;
    // Distance from vertex to centroid.
    return length(p1 - c);
}

// Compute the rest length for the provided triangle.
float
computeTriangleBendRestLength(const int geo; const int pt0, pt1, pt2)
{
    vector p0 = point(geo, "P", pt0);
    vector p1 = point(geo, "P", pt1);
    vector p2 = point(geo, "P", pt2);

    return computeVertexCentroidDistance(p0, p1, p2);
}

// Compute the rest length for the provided angle (degrees), assuming the provided
// triangle edge lengths.
float
computeTriangleBendRestLengthFromAngle(const int geo; const int pt0, pt1, pt2; const float angle)
{
    vector p0 = point(geo, "P", pt0);
    vector p1 = point(geo, "P", pt1);
    vector p2 = point(geo, "P", pt2);

    // Re-create a triangle with vertex at 0 and the given side lengths and angle and
    // compute vertex/centroid distance.
    float a = radians(angle);
    float r0 = length(p1 - p0);
    float r2 = length(p2 - p1);
    vector c = set(r2 * cos(a) - r0, r2 * sin(a), 0) / 3;
    return length(c); // |{0, 0, 0} - c|
}

void
createTriangleBendConstraints(const int geo; const int ptnum; const string type;
                              const string srcgrp; const float maxbranchangle;
                              const int outgeo; const string outgrp)
{
    int midpt = ptnum;
    int nbrs[] = neighbours(geo, midpt);
    foreach(int startpt; nbrs)
    {
        foreach(int endpt; nbrs)
        {
            if (startpt < endpt && inpointgroup(geo, srcgrp, startpt) && inpointgroup(geo, srcgrp, endpt))
            {
                float restlen = computeAngleRestLength(geo, startpt, midpt, endpt);
                if (restlen <= maxbranchangle)
                {
                    int cprim = addprim(outgeo, "polyline", array(startpt, midpt, endpt));
                    if (type == "trianglebend")
                        restlen = computeTriangleBendRestLength(geo, startpt, midpt, endpt);
                    setprimattrib(outgeo, "restlength", cprim, restlen);
                    setprimattrib(outgeo, "type", cprim, type);
                    setprimgroup(outgeo, outgrp, cprim, 1);
                }
            }
        }
    }
}

void
createGlueConstraints(const int geo, congeo; const int pt; const string srcgrp, dstgrp, primgrp, classname;
                      const int usecluster; const string clusterattrib;
                      const int numconstraint;
                      const float minrad, maxrad; const int maxpt; const int pref;
                      const float seed, threshold, ptthreshold;
                      const int outgeo; const string outgrp)
{
    vector pos = point(geo, "P", pt);
    int nconstraints = numconstraint;
    if (nconstraints <= 0)
        return;

    int myclass = point(geo, classname, pt);
    string myclass_s = point(geo, classname, pt);

    int mycluster = -2;
    string mycluster_s = "";
    string clustername = clusterattrib;
    if (usecluster)
    {
        mycluster = point(geo, clustername, pt);
        mycluster_s = point(geo, clustername, pt);
        // Cluster=-1 turns off gluing.
        if (mycluster == -1)
            return;
    }

    if ( float(rand(set(myclass, seed+M_PI))) < threshold)
    {
        return;
    }

    int indstgrp = inpointgroup(geo, dstgrp, pt);
    int hasprimgrp = strlen(primgrp) > 0;
    int nearpts[]  = nearpoints(geo, dstgrp, pos, maxrad, maxpt);
    if (pref == 1) // Farthest first
        nearpts = reverse(nearpts);
    foreach (int npt; nearpts)
    {
        // Skip this point.
        if (pt == npt)
            continue;

        if (minrad > 0 && distance(pos, point(geo, "P", npt)) < minrad)
            continue;
        // If src is in dest group, and dst is in source group, then
        // the nearpoints lookup will be symmetric, so only create for lower point number
        // to avoid duplicates.
        if (indstgrp && inpointgroup(congeo, srcgrp, npt) && npt < pt)
            continue;

        // Compare classes.
        int oclass = point(geo, classname, npt);
        string oclass_s = point(geo, classname, npt);
        if (oclass == myclass && oclass_s == myclass_s)
            continue;

        // Ignore disabled classes.
        if (float(rand(set(oclass, seed+M_PI))) < threshold)
            continue;

        // Ignore disabled point connections, ensuring any two unique pointsets
        // are always used for the seed.
        if (float(rand(set(min(pt, npt), max(pt, npt), seed))) < ptthreshold)
            continue;

        // Clustering.
        if (usecluster)
        {
            int ocluster = point(geo, clustername, npt);
            string ocluster_s = point(geo, clustername, npt);
            if ((ocluster != mycluster) || (ocluster_s != mycluster_s))
            {
                continue;
            }
        }

        // Check for already existing constraint primitive connecting the points.
        if (hasprimgrp)
        {
            int h = pointhedge(congeo, pt, npt);
            if (h < 0)
                h = pointhedge(congeo, npt, pt);
            if (h >= 0 && inprimgroup(congeo, primgrp, hedge_prim(congeo, h)))
                continue;
        }
        vector opos = point(geo, "P", npt);

        // Generate connecting line.
        int prim = addprim(outgeo, "polyline", array(pt, npt));
        setprimgroup(geoself(), outgrp, prim, 1);
        float rlen = distance(pos, opos);
        setprimattrib(geoself(), "restlength", prim, rlen);

        // Found a good constraint, decrement.
        nconstraints--;
        if (nconstraints <= 0)
            break;
    }
}

void
createStrutConstraints(const int geo; const int ptnum;
                       const string srcgrp, classname, dirattrib;
                       const int revnml, checknml;
                       const int numconstraint;
                       const float maxlen, jitter, rayoff;
                       const float seed, ptthreshold;
                       const int outgeo; const string outgrp)
{
    vector pos = point(geo, 'P', ptnum);

    vector nml = -point(geo, dirattrib, ptnum);

    nml = normalize(nml);
    if (revnml)
        nml *= -1;

    if ( float(rand(set(ptnum, ptnum/1024, seed))) < ptthreshold )
    {
        return;
    }

    int myclass = point(geo, classname, ptnum);
    string myclass_s = point(geo, classname, ptnum);

    for (int i = 0; i < numconstraint; i++)
    {
        vector hitpos;
        vector hituv;
        vector dir = nml;

        dir += (vector(rand( set(i, ptnum, ptnum/1024, seed) )) - 0.5) * jitter;

        int hitprim = intersect(geo, pos + dir * rayoff, dir * maxlen, hitpos, hituv);
        if (hitprim < 0)
            continue;

        // Valid hit!
        int nearpts[] = primpoints(geo, hitprim);
        if (len(nearpts) == 0)
            continue;
        int bestpt = -1;
        float bestdist = 1e23;
        foreach (int npt; nearpts)
        {
            if (!inpointgroup(geo, srcgrp, npt))
                continue;
            int oclass = point(geo, classname, npt);
            string oclass_s = point(geo, classname, npt);
            if (oclass != myclass || oclass_s != myclass_s)
                continue;
            float dist = distance( vector(point(geo, 'P', npt)), pos);
            if (dist < bestdist)
            {
                bestdist = dist;
                bestpt = npt;
            }
        }
        if (bestpt < 0)
            continue;
        if (checknml)
        {
            vector onml = point(geo, 'N', bestpt);
            if (revnml)
                onml *= -1;
            if (dot(onml, dir) < 0)
                continue;
        }

        // Connect ptnum and bestpt.
        vector opos = point(geo, 'P', bestpt);

        // Generate connecting line.
        int prim = addprim(outgeo, 'polyline', array(ptnum, bestpt));
        setprimgroup(outgeo, outgrp, prim, 1);
        float rlen = distance(opos, pos);
        setprimattrib(outgeo, 'restlength', prim, rlen);
    }
}

int
computeTetRestMatrix(const int geo; const int pt0, pt1, pt2, pt3; const float scale;
                     matrix3 restmatrix; float volume)
{
    // Compute inverse material matrix.
    vector p0 = point(geo, "P", pt0);
    vector p1 = point(geo, "P", pt1);
    vector p2 = point(geo, "P", pt2);
    vector p3 = point(geo, "P", pt3);

    matrix3 M = transpose(scale * set(p0 - p3, p1 - p3, p2 - p3));
    float detM = determinant(M);
    if (detM == 0)
        return 0;
    restmatrix = invert(M);
    volume = detM / 6;
    return 1;
}

int
computeTetFiberRestLength(const int geo; const int pt0, pt1, pt2, pt3; float volume; vector materialW)
{
    matrix3 restm;
    if (!computeTetRestMatrix(geo, pt0, pt1, pt2, pt3, 1, restm, volume))
        return 0;

    // Compute material uv
    materialW = {0, 0, 1};
    string wattrib = "materialW";
    if (haspointattrib(geo, wattrib))
    {
        materialW = point(geo, wattrib, pt0);
        materialW += point(geo, wattrib, pt1);
        materialW += point(geo, wattrib, pt2);
        materialW += point(geo, wattrib, pt3);
        materialW = normalize(materialW);
    }
    // Equivalent to mat3Tvecmul(Dm^-1, w) in OpenCL.
    materialW = materialW * transpose(restm);
    return 1;
}

void
createTetFiberConstraint(const int geo; const int ptnum; const string srcgrp;
                         const int outgeo; const string outgrp)
{
    int prims[] = pointprims(geo, ptnum);
    foreach(int prim; prims)
    {
        // Process tets where this point is minimum point number,
        // and all points are in src group.
        if(primintrinsic(geo, "typeid", prim) != 21)
            continue;
        int pts[] = primpoints(geo, prim);
        if (len(pts) != 4 ||
            ptnum != min(pts) ||
            !inpointgroup(geo, srcgrp, pts[1]) ||
            !inpointgroup(geo, srcgrp, pts[2]) ||
            !inpointgroup(geo, srcgrp, pts[3]))
            continue;

        // Compute inverse material matrix.
        vector materialW;
        float volume;
        if (!computeTetFiberRestLength(geo, pts[0], pts[1], pts[2], pts[3], volume, materialW))
            continue;

        int cprim = addprim(outgeo, "tet", pts);
        setprimattrib(outgeo, "restlength", cprim, volume);
        setprimattrib(outgeo, "restvector", cprim, vector4(materialW));
        setprimgroup(outgeo, outgrp, cprim, 1);
    }
}

int
computeTriRestMatrix(const int geo; const int pt0, pt1, pt2; const float scale;
                     matrix2 restmatrix; float area)
{
    vector p0 = point(geo, "P", pt0);
    vector p1 = point(geo, "P", pt1);
    vector p2 = point(geo, "P", pt2);

    // Xform from world to 2-d triangle space.
    matrix3 xform = fromTriangleSpaceXform(p0, p1, p2);
    xform = transpose(xform);

    vector2 P0 = (vector2)(p0 * xform);
    vector2 P1 = (vector2)(p1 * xform);
    vector2 P2 = (vector2)(p2 * xform);

    matrix2 M = transpose(scale * set(P0 - P2, P1 - P2));
    float detM = determinant(M);
    if (detM == 0)
        return 0;
    restmatrix = invert(M);
    area = abs(detM / 2);
    return 1;
}

// Closed form 2d polar decomposition.
// See http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
matrix2
polardecomp2d(matrix2 A)
{
    vector2 m = set(A.xx + A.yy, A.yx - A.xy);
    m = normalize(m);
    return set(m.x, -m.y, m.y, m.x);
}

// Scales a triangle and returns its area.
float
computeTriAreaRestLength(const int geo; const int pt0, pt1, pt2; const float scale)
{
    vector p0 = point(geo, "P", pt0);
    vector p1 = point(geo, "P", pt1);
    vector p2 = point(geo, "P", pt2);
    vector e0 = scale * (p1 - p0);
    vector e1 = scale * (p2 - p0);
    vector e2 = cross(e1, e0);
    float area = length(e2) / 2;
    return area;
}

void
createTriStretchConstraint(const int geo; const int ptnum; const string srcgrp, type;
                           const float restscale;
                           const int outgeo; const string outgrp)
{
    int prims[] = pointprims(geo, ptnum);
    foreach(int prim; prims)
    {
        // Process triangles where this point is minimum point number,
        // and all points are in src group.
        if(primintrinsic(geo, "typeid", prim) != 1)
            continue;
        int pts[] = primpoints(geo, prim);
        if (len(pts) != 3 ||
            ptnum != min(pts) ||
            !inpointgroup(geo, srcgrp, pts[1]) ||
            !inpointgroup(geo, srcgrp, pts[2]))
            continue;

        matrix2 restmatrix;
        float area;
        if (!computeTriRestMatrix(geo, pts[0], pts[1], pts[2], restscale, restmatrix, area))
            continue;

        int cprim = addprim(outgeo, "poly", pts);
        setprimattrib(outgeo, "restlength", cprim, area);
        // Store 2x2 matrix in restvector.
        vector4 restvec = set(restmatrix.xx, restmatrix.xy,
                              restmatrix.yx, restmatrix.yy);
        setprimattrib(outgeo, "restvector", cprim, restvec);
        setprimattrib(outgeo, "type", cprim, type);
        setprimgroup(outgeo, outgrp, cprim, 1);
    }
}

void
createTetStretchConstraint(const int geo; const int ptnum; const string srcgrp, type;
                           float restscale;
                           const int outgeo; const string outgrp)
{
    int prims[] = pointprims(geo, ptnum);
    foreach(int prim; prims)
    {
        // Process tets where this point is minimum point number,
        // and all points are in src group.
        if(primintrinsic(geo, "typeid", prim) != 21)
            continue;
        int pts[] = primpoints(geo, prim);
        if (len(pts) != 4 ||
            ptnum != min(pts) ||
            !inpointgroup(geo, srcgrp, pts[1]) ||
            !inpointgroup(geo, srcgrp, pts[2]) ||
            !inpointgroup(geo, srcgrp, pts[3]))
            continue;

        matrix3 restmatrix;
        float volume;
        if (!computeTetRestMatrix(geo, pts[0], pts[1], pts[2], pts[3], restscale, restmatrix, volume))
            continue;
        int cprim = addprim(outgeo, "tet", pts);
        setprimattrib(outgeo, "restlength", cprim, volume);
        setprimattrib(outgeo, "restmatrix", cprim, restmatrix);
        setprimattrib(outgeo, "restvector", cprim, vector4({0, 0, 0, 1}));
        setprimattrib(outgeo, "type", cprim, type);
        setprimgroup(outgeo, outgrp, cprim, 1);
    }
}

int
orientedRestDifference(const int geo, pts[]; const string type;
                       const vector4 restvector; vector aadiff)
{
    int isorient = 0;
    vector4 curorient;
    if (type == "pinorient")
    {
        curorient = point(geo, "orient", pts[0]);
        isorient = 1;
    }
    else if (type == "bendtwist")
    {
        curorient = computeBendTwistRestVector(geo, pts[0], pts[1]);
        isorient = 1;
    }

    if (isorient)
    {
        vector4 restconj = restvector * {-1, -1, -1, 1};
        // Angle/axis representation of orientation difference.
        aadiff = qconvert(qmultiply(restconj, curorient));
    }
    return isorient;
}

float
logscaleStiffness(const float k, stiffness)
{
    // Restrict to non-negative, finite values.
    return clamp(exp(k * log(stiffness + 1)) - 1, 0, 1e37);
}

// Returns the value that can be passed to logscaleStiffness to
// get x given the same stiffness.
float
invlogscaleStiffness(const float x, stiffness)
{
    // Restrict to non-negative, finite values.
    return clamp(log(x + 1) / log(stiffness + 1), 0, 1e37);
}

// Increment restvector by amount towards current deformation in
// axis/angle space.
// aadiff is angle/axis representation of orientation difference.
// Returns the degrees the orientation was updated by.
float
updateRestVectorOrient(const float inamount; const int isratio;
                       const vector aadiff; vector4 restvector)
{
    float amount = inamount;
    // Difference in degrees.
    float degdiff = degrees(length(aadiff));
    // Convert to ratio if amount already in degrees.
    if (!isratio)
        amount /= degdiff;
    // Increment towards current deformation in axis/angle space.
    // aadiff is angle/axis representation of orientation difference.
    vector rest = qconvert(restvector);
    rest += clamp(amount, 0, 1) * aadiff;
    restvector = quaternion(rest);
    return amount * degdiff;
}

float
squaredNorm2(const matrix2 a)
{
    // Below conversion doesn't exist for some reason.
    // vector2 rows[] = set(a);
    vector2 rows[];
    resize(rows, 2);
    rows[0] = set(a.xx, a.xy);
    rows[1] = set(a.yx, a.yy);
    return dot(rows[0], rows[0]) + dot(rows[1], rows[1]);
}

float
squaredNorm3(const matrix3 a)
{
    vector rows[] = set(a);
    return dot(rows[0], rows[0]) + dot(rows[1], rows[1]) + dot(rows[2], rows[2]);
}

// Update the 2x2 restmatrix stored in restvector by amount,
// given the difference vector4 supplied.
float
updateRestMatrix2(const float inamount; const int isratio; const vector4 vecdiff;
                  float restlength; vector4 restvector)
{
    matrix2 D_m_inv = set(restvector.x, restvector.y, restvector.z, restvector.w);
    matrix2 D_m = invert(D_m_inv);
    float amount = inamount;
    float sqrnorm = squaredNorm2(D_m);
    // Convert to ratio if in length units.
    // Use squared norm for "length"
    if (!isratio)
        amount /= sqrnorm;
    amount = clamp(amount, 0, 1);
    // restvector holds the inverse of the material space matrix, so we have to invert before
    // and after updating.
    matrix2 matdiff = set(vecdiff.x, vecdiff.y, vecdiff.z, vecdiff.w);
    D_m += amount * matdiff;
    D_m_inv = invert(D_m);
    // Store inverse of D_m in restvector and area in restlength.
    restvector = set(D_m_inv.xx, D_m_inv.xy, D_m_inv.yx, D_m_inv.yy);
    restlength = abs(determinant(D_m) / 2);
    // Return change in squared norm
    return abs(squaredNorm2(D_m) - sqrnorm);
}

// Update the restmatrix by amount given the difference matrix
// supplied.
float
updateRestMatrix(const float inamount; const int isratio; const matrix3 matdiff;
                 float restlength; matrix3 restmatrix)
{
    float amount = inamount;
    matrix3 D_m = invert(restmatrix);
    float sqrnorm = squaredNorm3(D_m);
    // Convert to ratio if in length units.
    // Use squared norm for "length"
    if (!isratio)
        amount /= sqrnorm;
    amount = clamp(amount, 0, 1);
    // restmatrix holds the inverse of the material space matrix, so we have to invert before
    // and after updating.
    D_m += amount * matdiff;
    // Store inverse of D_m in restmatrix and volume in restlength.
    restmatrix = invert(D_m);
    restlength = determinant(D_m) / 6;
    // Return change in squared norm
    return abs(squaredNorm3(D_m) - sqrnorm);
}

int
computeRestVectorDifference(const int geo, pts[];  const string type; const vector4 restvector;
                            vector4 vecdiff)
{
    if (!isTetFiber(type) && !isTriARAP(type))
        return 0;
    vecdiff = 0;
    if (isTetFiber(type))
    {
        float volume;
        vector curvector, rv = vector(restvector);
        if (computeTetFiberRestLength(geo, pts[0], pts[1], pts[2], pts[3], volume, curvector))
        {
            vecdiff = vector4(curvector - rv);
            vecdiff.w = 0;
        }
    }
    else if (isTriARAP(type))
    {
        // We need to compute the difference in material coordinates, so we get the polar decomposition
        // of the deformation gradient and transform the new coordinates into material space.
        matrix2 D_s_inv, D_m_inv = set(restvector.x, restvector.y, restvector.z, restvector.w);
        float area;
        // Compute the current (inverse) spatial matrix.  Bail out if degenerate.
        if (!computeTriRestMatrix(geo, pts[0], pts[1], pts[2], 1, D_s_inv, area))
            return 1;
        matrix2 D_s = invert(D_s_inv);
        // Deformation gradient.
        matrix2 F = D_s * D_m_inv;
        matrix2 R = polardecomp2d(F);
        D_s = transpose(R) * D_s;
        // Difference in material coordinates.
        matrix2 mdiff = D_s - invert(D_m_inv);
        vecdiff = set(mdiff.xx, mdiff.xy, mdiff.yx, mdiff.yy);
    }
    return 1;
}

int
computeRestMatrixDifference(const int geo, pts[];  const string type; const matrix3 restmatrix;
                            matrix3 matdiff)
{
    if (!isTetARAP(type))
        return 0;
    matdiff = 0;

    // We need to compute the difference in material coordinates, so we get the polar decomposition
    // of the deformation gradient and transform the new coordinates into material space.
    matrix3 D_s_inv;
    float volume;
    // Compute the current (inverse) spatial matrix.  Bail out if degenerate.
    if (!computeTetRestMatrix(geo, pts[0], pts[1], pts[2], pts[3], 1, D_s_inv, volume))
        return 1;
    matrix3 D_s = invert(D_s_inv);
    // Deformation gradient.
    matrix3 F = D_s * restmatrix;
    matrix3 R = polardecomp(F);
    D_s = transpose(R) * D_s;
    // Difference in material coordinates.
    matdiff = D_s - invert(restmatrix);

    return 1;
}

// Diff should be (signed) distance from curlength to restlength.
// Returns amount of plastic flow, for angular constraints in degrees.
float
plasticDeformation(const int geo, pts[];  const string intype; const float diff;
                   const float plasticrate, plasticthreshold, plastichardening, dt;
                   float restlength; vector4 restvector; matrix3 restmatrix;
                   float stiffness)
{
    float threshold = plasticthreshold;
    // Use aliased constraint type.
    string type = constraintAlias(intype);

    // Negative stretch value indicates ratio of current restlength;
    if (threshold < 0)
        threshold = -threshold * restlength;

    // Nothing to do if not past threshold.
    if (abs(diff) <= threshold)
        return 0;

    vector aadiff;
    vector4 vecdiff;
    matrix3 matdiff;
    // This plasticity model roughly matches the wire solver.
    float u = exp(-plasticrate * dt);
    float v = 1 - u;
    float flow = 0;
    // Need to recompute expensive difference for some constraint types,
    // but only once we already know we're past the plastic threshold.
    if (orientedRestDifference(geo, pts, type, restvector, aadiff))
    {
        flow = updateRestVectorOrient(v, 1, aadiff, restvector);
    }
    else if (computeRestVectorDifference(geo, pts, type, restvector, vecdiff))
    {
        flow = updateRestMatrix2(v, 1, vecdiff, restlength, restvector);
    }
    else if (computeRestMatrixDifference(geo, pts, type, restmatrix, matdiff))
    {
        // We don't want to update restlength (volume) under plastic deformation
        // for tetarap constraints, so they can still preserve their original volume
        // stored in restlength.
        float temprl = restlength;
        flow = updateRestMatrix(v, 1, matdiff, temprl, restmatrix);
    }
    else
    {
        flow = v * diff;
        restlength += flow;
    }
    // Update stiffness from hardening.
    float k = u + v * plastichardening;
    float s = logscaleStiffness(k, stiffness);
    // Ensure we stay finite for stiffness.
    stiffness = select(isfinite(s), s, stiffness);
    return flow;
}


int []
getSourcePoints(const string intype; const int pts[])
{
    string type = constraintAlias(intype);
    if (type == "distance" || type == "stretchshear" || type == "weld" || type == "ptprim")
    {
        return pts[:1];
    }
    return pts;
}

int []
getTargetPoints(const string intype; const int pts[])
{
    string type = constraintAlias(intype);
    if (type == "distance" || type == "stretchshear" || type == "weld" || type == "ptprim")
    {
        return pts[1:];
    }
    return {};
}

// Return a scaling from accumulating attribute values and/or
// scalar multiplier, depending on mode and promotion.
float
accumScaleValues(const string type;
                 const int ptgeo; const int inpts[];
                 const string primgeo; const int targetprim; const vector uv;
                 const string attr;
                 const float valscale; const string scalemode, promotion)
{
    float attrscale = 0;
    int div = 0;

    if (scalemode == "attrib" || scalemode == "attribvalue")
    {
        int pts[] = inpts;
        if (promotion == "source")
            pts = getSourcePoints(type, inpts);
        if (promotion == "target")
            pts = getTargetPoints(type, inpts);
        if (promotion == "min")
            attrscale = 1e23;
        if (promotion == "mult")
            attrscale = 1;
        // Accumulate point parts.
        if (haspointattrib(ptgeo, attr))
        {
            foreach(int pt; pts)
            {
                float ptval = point(ptgeo, attr, pt);
                if (promotion == "mean" || promotion == "source" || promotion == "target")
                {
                    attrscale += ptval;
                    div++;
                }
                else if (promotion == "max")
                {
                    attrscale = max(attrscale, ptval);
                    div = 1;
                }
                else if (promotion == "min")
                {
                    attrscale = min(attrscale, ptval);
                    div = 1;
                }
                else if (promotion == "mult")
                {
                    attrscale *= ptval;
                    div = 1;
                }
            }
        }
        // Accumulate primitive part.  The primuv function will use prim, vertex, or point attribs.
        // prim can never be a source.
        if (primgeo != "" && targetprim >= 0 && promotion != "source" &&
            (hasprimattrib(primgeo, attr) || hasvertexattrib(primgeo, attr) || haspointattrib(primgeo, attr)))
        {
            float primval = primuv(primgeo, attr, targetprim, uv);
            if (promotion == "mean" || promotion == "target")
            {
                attrscale += primval;
                div++;
            }
            else if (promotion == "max")
            {
                attrscale = max(attrscale, primval);
                div = 1;
            }
            else if (promotion == "min")
            {
                attrscale = min(attrscale, primval);
                div = 1;
            }
            else if (promotion == "mult")
            {
                attrscale *= primval;
                div = 1;
            }
        }
    }
    float scale = 1;
    if (div > 0)
        scale *= attrscale / div;
    if (scalemode == "value" || scalemode == "attribvalue")
        scale *= valscale;
    return scale;
}

vector
falseColor(const float val, minval, maxval)
{
    return hsvtorgb(0.7 * (1 - fit(val, minval, maxval, 0, 1)), 1, 1);
}

int
weldTarget(const int geo, ptnum)
{
    if (haspointattrib(geo, "weld"))
    {
        int weld = point(geo, "weld", ptnum);
        if (weld >= 0)
            return idtopoint(geo, weld);
    }
    return ptnum;
}

float computeStress(const string type; const vector Lin;
                    const float dt; const int normalize)
{
    vector L = Lin;
    // We want to ignore the volume component for tetARAP/volume constraints.
    L.z = select(isTetARAPVol(type), 0, Lin.z);
    if (!normalize)
        return length(L);
    // nonlinear ARAP divides by just dt. Although it doesn't work that well,
    // it's still better than dt^2
    float scale = 1.0 / select(isNonLinearARAP(type), dt, dt * dt);
    return length(L) * scale;
}

float
maxConstraintStress(const int ptgeo, congeo, ptnum; const string types[])
{
    // Use the weld target if any.
    int pt = weldTarget(ptgeo, ptnum);
    // Iterate over all constraint primitives connected
    // to each point and find the max "stress".
    int prims[] = pointprims(congeo, pt);
    float stress = 0;
    foreach(int prim; prims)
    {
        string type = prim(congeo, "type", prim);
        if (find(types, type) < 0)
            continue;
        stress = max(stress, prim(congeo, "stress", prim));
    }
    return stress;
}

float
computeRestLengthDifference(const int geo, pts[];  const string intype;
                            const float restlength; const vector4 restvector;
                            const vector restdir; const string targetpath; const int targetprim;
                            const matrix3 restmatrix)
{
    float curlength;
    vector4 curorient;
    // Use aliased constraint type.
    string type = constraintAlias(intype);

    if (type == "pin")
    {
        curlength = distance(point(geo, "P", pts[0]), vector(restvector));
    }
    else if (type == "distance" || type == "stretchshear")
    {
        curlength = computeDistanceRestLength(geo, pts[0], pts[1]);
    }
    else if (type == "distanceline")
    {
        curlength = computeDistanceLineRestLength(geo, pts[0], vector(restvector), restdir, targetpath, targetprim);
    }
    else if (type == "pressure")
    {
        curlength = computePressureRestLength(geo, pts, "volume");
    }
    else if (type == "tetvolume")
    {
        curlength = computeTetVolumeRestLength(geo, pts[0], pts[1], pts[2], pts[3]);
    }
    else if (type == "bend")
    {
        if (!computeDihedralRestLength(geo, pts[0], pts[1], pts[2], pts[3],
                                               curlength))
            return 0;
    }
    else if (type == "angle")
    {
        curlength = computeAngleRestLength(geo, pts[0], pts[1], pts[2]);
    }
    else if (type == "trianglebend")
    {
        curlength = computeTriangleBendRestLength(geo, pts[0], pts[1], pts[2]);
    }
    else if (type == "ptprim")
    {
        curlength = computePointPrimRestLength(geo, pts[0], pts[1:], restvector);
    }
    // Use axis/angle orientation difference if exists.
    vector aadiff;
    if (orientedRestDifference(geo, pts, type, restvector, aadiff))
        return degrees(length(aadiff));

    vector4 vecdiff;
    if (computeRestVectorDifference(geo, pts, type, restvector, vecdiff))
        return squaredNorm2(set(vecdiff.x, vecdiff.y, vecdiff.z, vecdiff.w));

    matrix3 matdiff;
    if (computeRestMatrixDifference(geo, pts, type, restmatrix, matdiff))
        return squaredNorm3(matdiff);

    float diff = curlength - restlength;
    return diff;
}

// Update restlength or restvector from the current rest state
// as depicted in the point positions in geo.
// Amount can be a 0-1 ratio or a unit amount.
void
updateFromRest(const int geo, pts[]; const string intype;
               const float inamount; const int isratio;
               float restlength; vector4 restvector; matrix3 restmatrix;
                const vector restdir; const string targetpath; const int targetprim)
{
    float amount = inamount;
    // Check if nothing to do.
    if (isratio && amount <= 0)
        return;

    // Use aliased constraint type.
    string type = constraintAlias(intype);
    float diff = 0;
    // Use axis/angle orientation difference if exists.
    vector aadiff = 0;
    vector4 vecdiff = 0;
    matrix3 matdiff = 0;
    if (orientedRestDifference(geo, pts, type, restvector, aadiff))
    {
        updateRestVectorOrient(inamount, isratio, aadiff, restvector);
    }
    else if (computeRestVectorDifference(geo, pts, type, restvector, vecdiff))
    {
        if (isTriARAP(type))
        {
            updateRestMatrix2(inamount, isratio, vecdiff, restlength, restvector);
        }
        else
        {
            // Convert to ratio if in length units.
            if (!isratio)
                amount /= length(vecdiff);
            amount = clamp(amount, 0, 1);
            restvector += amount * vecdiff;
            restlength += amount * diff;
        }
    }
    else if (computeRestMatrixDifference(geo, pts, type, restmatrix, matdiff))
    {
        updateRestMatrix(inamount, isratio, matdiff, restlength, restmatrix);
    }
    else
    {
        diff = computeRestLengthDifference(geo, pts, type, restlength, restvector,
                                                 restdir, targetpath, targetprim, restmatrix);
        // Convert to ratio if in length units.
        if (!isratio)
            amount /= abs(diff);
        restlength += clamp(amount, 0, 1) * diff;
    }
}

float
computeStiffnessDropoff(const int geo, pts[]; const string type;
                        const float restlength; const vector4 restvector;
                        const vector restdir; const string targetpath; const int targetprim;
                        const matrix3 restmatrix;
                        const float indropoff; const float stiffnessorig; const float stiffnessmin)
{
    // Nothing to do for no dropoff.
    float dropoff = abs(indropoff);
    if (dropoff == 0)
        return stiffnessorig;
    float diff = computeRestLengthDifference(geo, pts, type, restlength, restvector,
                                             restdir, targetpath, targetprim, restmatrix);
    diff = abs(diff);
    float k = 0;
    // If we have a minimum desired stiffness, use its inverse as the lower bound.
    if (stiffnessmin > 0)
        k = invlogscaleStiffness(stiffnessmin, stiffnessorig);

    float scale;
    // If dropoff is negative, decrease stiffness towards dropoff, else increase.
    if (indropoff < 0)
        scale = fit(diff, 0, dropoff, 1, k);
    else
        scale = fit(diff, 0, dropoff, k, 1);
    // log scale
    return logscaleStiffness(scale, stiffnessorig);
}

// Scalar rest metric used as the divisor for ratio-based visualizations and plastic deformation.
float
restMetric(const string type; const float restlength; const vector4 restvector;
           const matrix3 restmatrix)
{
    // 2d and 3d restmatrices store inverse of material space coordinates.
    if (isTriARAP(type))
        return squaredNorm2(invert(set(restvector.x, restvector.y, restvector.z, restvector.w)));
    if (isTetARAP(type))
        return squaredNorm3(invert(restmatrix));
    return restlength;
}

float
constraintMetric(const int ptgeo, congeo, prim; const string type, metric)
{
    if (metric == "stretchstress" || metric == "bendstress")
        return prim(congeo, "stress", prim);

    float restlength = prim(congeo, "restlength", prim);
    vector4 restvector = prim(congeo, "restvector", prim);
    vector restdir = prim(congeo, "restdir", prim);
    string targetpath = prim(congeo, "target_path", prim);
    int targetprim = prim(congeo, "target_prim", prim);
    matrix3 restmatrix = prim(congeo, "restmatrix", prim);
    float val = computeRestLengthDifference(ptgeo, primpoints(congeo, prim), type,
                                            restlength, restvector,
                                            restdir, targetpath, targetprim, restmatrix);
    val = abs(val);
    if (metric == "stretchratio")
        val /= restMetric(type, restlength, restvector, restmatrix);
    return val;
}

float
maxConstraintMetric(const int ptgeo, congeo, ptnum; const string types[], metric)
{
    // Use the weld target if any.
    int pt = weldTarget(ptgeo, ptnum);
    // Iterate over all constraint primitives of specified types connected
    // to input point and evaluate the max metric.
    int prims[] = pointprims(congeo, pt);
    float maxval = 0;
    foreach(int prim; prims)
    {
        string type = prim(congeo, "type", prim);
        if (find(types, type) < 0)
            continue;
        maxval = max(maxval, constraintMetric(ptgeo, congeo, prim, type, metric));
    }
    return maxval;
}

float
warpWeftScale(const int geo, pts[]; const float warp, weft, shear; const string materialuv)
{
    vector p0 = point(geo, materialuv, pts[0]);
    vector p1 = point(geo, materialuv, pts[1]);
    p1 -= p0;
    float angle = degrees(atan2(p1.x, p1.y)) % 180;

    if (angle > 90)
    {
        angle = 180 - angle;
    }

    float scale;

    angle /= 90;
    if (angle < 0.5)
    {
        scale = lerp(warp, shear, angle*2);
    }
    else
    {
        scale = lerp(shear, weft, (angle-0.5)*2);
    }
    return scale;
}

// Return an array of pts welded to or from the points in pts array.
// NOTE: The geometry should have weldattrib with default value -1
// for this function work correctly.
int[]
findwelds(const string geo; const int pts[]; const string weldattrib)
{
    string weldgrp;
    string weldstr[];
    int welds[];
    if (len(pts) == 0)
        return welds;
    foreach(int pt; pts)
    {
        int weld = point(geo, weldattrib, pt);
        if (weld >=0)
        {
            // Include any points this point is welded to.
            append(weldstr, itoa(weld));
            // And any points also welded to those points.
            append(weldstr, sprintf("\@weld=%d", weld));
        }
        // Include any points welded to this point.
        append(weldstr, sprintf("\@weld=%d", pt));
    }
    // Expand point group ensures we don't get duplicates.
    welds = expandpointgroup(geo, join(weldstr," "));
    return welds;
}

float
walkdist(const string geo; const string grp; const vector goalpos;
         const int type; const string weldattrib;
         int curprim; vector curuv)
{
    int prims[], pts[];
    string primstr[];
    int visited[];
    float dist = 0;
    int hasgrp = strlen(grp) > 0;
    int haswelds = haspointattrib(geo, weldattrib);

    while(find(visited, curprim) < 0)
    {
        append(visited, curprim);
        primstr = array(itoa(curprim));
        pts = primpoints(geo, curprim);
        int npts = len(pts);
        // Include welded points if needed.
        if (haswelds)
            append(pts, findwelds(geo, pts, weldattrib));

        foreach(int pt; pts)
        {
            prims = pointprims(geo, pt);
            foreach(int prim; prims)
            {
                if (prim != curprim &&                                              // skip current prim
                    (!hasgrp || inprimgroup(geo, grp, prim)) &&                     // skip not in group
                    (type < 0 || primintrinsic(geo, "typeid", prim) == type))       // only matching type
                    append(primstr, itoa(prim));
            }
        }
        dist = xyzdist(geo, join(primstr, " "), goalpos, curprim, curuv);
    }
    return dist;
}

// Version used for Attach constraints that doesn't limit by type or npts or support welds.
float
walkdist(const string geo; const string grp; const vector goalpos;
         int curprim; vector curuv)
{
    return walkdist(geo, grp, goalpos, -1, "", curprim, curuv);
}

// From "Real-Time Collision Detection" by Ericson.
// with modifications to return closest vertex or edge
// barycentric coords.
vector
closestpttriangle(const vector p, a, b, c;
                  int vert, edge; vector uv)
{
    vert = edge = -1;
    // Check if P in vertex region outside A
    vector ab = b - a;
    vector ac = c - a;
    vector ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        vert = 0;
        uv = set(1, 0, 0);
        return a; // barycentric coordinates (1,0,0)
    }

    // Check if P in vertex region outside B
    vector bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        vert = 1;
        uv = set(0, 1, 0);
        return b; // barycentric coordinates (0,1,0)
    }

    // Check if P in edge region of AB, if so return projection of P onto AB
    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        edge = 0;
        float v = d1 / (d1 - d3);
        uv = set(1-v, v, 0);
        return a + v * ab; // barycentric coordinates (1-v,v,0)
    }

    // Check if P in vertex region outside C
    vector cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        vert = 2;
        uv = set(0, 0, 1);
        return c; // barycentric coordinates (0,0,1)
    }

    // Check if P in edge region of AC, if so return projection of P onto AC
    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        edge = 2;
        float w = d2 / (d2 - d6);
        uv = set(1-w, 0, w);
        return a + w * ac; // barycentric coordinates (1-w,0,w)
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        edge = 1;
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        uv = set(0, 1-w, w);
        return b + w * (c - b); // barycentric coordinates (0,1-w,w)
    }

    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    uv = set(1-v-w, v, w);
    return a + ab * v + ac * w; // = u*a + v*b + w*c, u = va * denom = 1.0f-v-w
}

// Wrapper that translates vertex or edge ordinals to point numbers
// or half-edges, respectively, as well as translating barycentric coords.
vector
closestpttriangle(const string geo; const int prim, pts[]; const vector p;
                  int ptnum, hedge; vector uv)
{
    ptnum = hedge = -1;
    vector p0 = point(geo, "P", pts[0]);
    vector p1 = point(geo, "P", pts[1]);
    vector p2 = point(geo, "P", pts[2]);
    int vert, edge;
    vector closepos = closestpttriangle(p, p0, p1, p2, vert, edge, uv);
    if (vert >= 0)
        ptnum = pts[vert];
    else if (edge >= 0)
        hedge = vertexhedge(geo, primvertex(geo, prim, edge));

    // Translate from barycentric as returned by inner closestpttriangle.
    uv = set(uv.y, uv.z, 0);
    return closepos;
}

int
ispolytri(const string geo; const int prim, pts[])
{
    return len(pts) == 3 &&
           primintrinsic(geo, "typeid", prim) == 1 &&
           primintrinsic(geo, "closed", prim) == 1;
}

// Similar to walkdist but uses closest point to triangle tests for speed.
// At the moment only supports triangles, but *might* be expanded
// to quads at some point.
float
triwalkdist(const string geo; const string grp; const vector goalpos;
            const string weldattrib;
            int curprim; vector curuv)
{
    int visited[];
    int hasgrp = strlen(grp) > 0;
    int haswelds = haspointattrib(geo, weldattrib);

    int pts[] = primpoints(geo, curprim);
    // Nothing much we can do if not a triangle
    if (!ispolytri(geo, curprim, pts))
    {
        vector curpos = primuv(geo, "P", curprim, curuv);
        return xyzdist(geo, itoa(curprim), curpos, curprim, curuv);
    }

    int curpt = -1, curhedge = -1;
    vector curpos;
    curpos = closestpttriangle(geo, curprim, pts, goalpos,
                               curpt, curhedge, curuv);
    float dist = distance(curpos, goalpos);

    while(find(visited, curprim) < 0)
    {
        append(visited, curprim);
        pts = primpoints(geo, curprim);
        // If still in current triangle, we're done.
        if (curpt < 0 && curhedge < 0)
             break;
        int prims[], nextpts[];

        if (curhedge >= 0)
        {
            // On triangle edge, find edge of next primitive.
            int nexthedge = hedge_nextequiv(geo, curhedge);

            do
            {
                // Get prim from that edge.
                int nextprim = hedge_prim(geo, nexthedge);
                // Ensure it meets group and type criteria and add to prims.
                if (nextprim != curprim &&
                    (!hasgrp || inprimgroup(geo, grp, nextprim)) &&
                    ispolytri(geo, nextprim, primpoints(geo, nextprim)))
                    append(prims, nextprim);
                // Find any welds from the end points of the edge.
                if (haswelds)
                    nextpts = findwelds(geo,
                                        array(hedge_srcpoint(geo, nexthedge), hedge_dstpoint(geo, nexthedge)),
                                        weldattrib);
                nexthedge = hedge_nextequiv(geo, nexthedge);
            } while (nexthedge != curhedge);

        }
        else
        {
            // On vertex, add to list of next points to check.
            nextpts = array(curpt);
            // And welded points.
            if (haswelds)
            {
                int weldpts[] = findwelds(geo, nextpts, weldattrib);
                // Avoid duplicates.
                removevalue(weldpts, curpt);
                append(nextpts, weldpts);
            }
        }

        // Add any triangles from next set of points.
        foreach(int pt; nextpts)
        {
            foreach(int prim; pointprims(geo, pt))
                if (prim != curprim &&
                    (!hasgrp || inprimgroup(geo, grp, prim)) &&
                    ispolytri(geo, prim, primpoints(geo, prim)))
                append(prims, prim);

        }

        // Check each candidate triangle for closer.
        foreach(int prim; prims)
        {
            pts = primpoints(geo, prim);
            vector nextuv;
            int nextpt = -1;
            int nexthedge = -1;
            vector nextpos = closestpttriangle(geo, prim, pts, goalpos,
                                               nextpt, nexthedge, nextuv);
            float d = distance(nextpos, goalpos);
            if (d < dist)
            {
                dist = d;
                curprim = prim;
                curuv = nextuv;
                curpt = nextpt;
                curhedge = nexthedge;
            }
        }
    }
    return dist;
}

// Extract the rotation from the provided matrix using the algorithm from
// "A robust method to extract the rotational part of deformations."
// The input q is hopefully the rotation from a previous timestep or constraint
// solve for faster convergence.
vector4
extractRotation(const matrix3 A; const vector4 qin; const int maxiter)
{
    vector4 q = qin;
    vector Arow[] = set(A);
    for(int i = 0; i < maxiter; i++)
    {
        vector Rrow[] = set(qconvert(q));
        vector omega = cross(Rrow[0], Arow[0]) + cross(Rrow[1], Arow[1]) + cross(Rrow[2], Arow[2]);
        omega /= abs(dot(Rrow[0], Arow[0]) + dot(Rrow[1], Arow[1]) + dot(Rrow[2], Arow[2])) + 1e-5f;
        float w = length(omega);
        if (w < 1e-5f)
            break;
        q = qmultiply(quaternion(w, omega / w), q);
        q = normalize(q);
    }
    return q;
}

// Kahan sum vector and matrix accumulators.
void
accumVec(const vector v; vector sum, c)
{
    vector y = v - c; 
    vector t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

void
accumMat3(const matrix3 v; matrix3 sum, c)
{
    matrix3 y = v - c; 
    matrix3 t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// Compute the rest and current centers of mass, and
// best-fit rotation from rest space to current space.
// The R quateronion should the last best known rotation
// or the unit quaternion if unknown.
void computeCmAndRot(const int geo, congeo, pts[];
                     vector restcm, cm; vector4 R)
{
    // Rest and current centers-of-mass using Kahan sum for accuracy.
    int npts = len(pts);
    restcm = cm = 0;
    vector restcmerr = 0, cmerr = 0;
    foreach(int pt; pts)
    {
        accumVec(point(congeo, "rest", pt), restcm, restcmerr);
        accumVec(point(geo, "P", pt), cm, cmerr);
    }
    restcm /= npts;
    cm /= npts;

    // Covariance matrix using Kahan sum for accuracy.
    matrix3 A = 0, Aerr = 0;
    foreach(int pt; pts)
    {
        vector rest = point(congeo, "rest", pt);
        vector pos = point(geo, "P", pt);
        vector q = rest - restcm;
        vector p = pos - cm;
        accumMat3(outerproduct(q, p), A, Aerr);
    }

    // Rotation from rest space to current.
    R = extractRotation(A, R, 20);
}

int
updateFromRestShapeMatch(const int geo, congeo, conpts[], srcpts[];
                         const float inamount; const int isratio;
                         const int outgeo)
{
    // Check if nothing to do.
    if (isratio && inamount <= 0)
        return 0;

    vector restcm, cm;
    vector4 R = {0, 0, 0, 1};
    // Find centers of mass and rotation from rest to current.
    computeCmAndRot(geo, congeo, srcpts, restcm, cm, R);
    // Current rotation back to rest.
    vector4 Rinv = qinvert(R);
    int n = len(srcpts);
    for(int i=0; i < n; i++)
    {
        // Diff from current point to rest in rest space.
        vector pos = point(geo, "P", srcpts[i]);
        vector rest = point(congeo, "rest", conpts[i]);
        vector diff = (qrotate(Rinv, pos - cm) + restcm) - rest;
        // Convert to ratio if in length units.
        float amount = inamount;
        if (!isratio)
            amount = inamount / length(diff);
        amount = clamp(amount, 0, 1);
        rest += diff * amount;
        // Update rest point attribute.
        setpointattrib(outgeo, "rest", conpts[i], rest);
    }
    return 1;
}

float
plasticDeformationShapeMatch(const int geo, congeo, pts[];
                             const float plasticrate, plasticthreshold,
                             plastichardening, dt;
                             const vector4 Rin;
                             float stiffness;
                             const int outgeo)
{
    // This plasticity model roughly matches the wire solver.
    float u = exp(-plasticrate * dt);
    float v = 1 - u;
    float flow = 0;

    vector restcm, cm;
    vector4 R = Rin;
    // Find centers of mass and rotation from rest to current.
    computeCmAndRot(geo, congeo, pts, restcm, cm, R);
    // Current rotation back to rest.
    vector4 Rinv = qinvert(R);

    // Check every point in the constraint for plastic flow.
    foreach(int pt; pts)
    {
        // Current point transformed to rest space.
        vector rest = point(congeo, "rest", pt);
        vector pos = point(geo, "P", pt);
        vector currest = qrotate(Rinv, pos - cm) + restcm;
        float diff = distance(currest, rest);
        if (diff > plasticthreshold)
        {
            vector newrest = lerp(rest, currest, v);
            float localflow = v * diff;
            setpointattrib(outgeo, "rest", pt, newrest);
            setpointattrib(outgeo, "plasticflow", pt, localflow, "add");
            flow += localflow;
        }
    }

    // Update stiffness from hardening.
    float k = u + v * plastichardening;
    float s = logscaleStiffness(k, stiffness);
    // Ensure we stay finite for stiffness.
    stiffness = select(isfinite(s), s, stiffness);

    // Return total flow for constraint.
    return flow;
}


// Returns whether the type supports smoothing in the constraint solve.
// We use this during graph-coloring to clear out the worksets for the
// smoothing sizes and offsets if no smoothing is possible, rather
// than call a bunch of empty constraint passes.
int
hasSmoothing(const string type)
{
    string nosmoothtypes[] = { "bend", "trianglebend", "angle", "tetfiber", "tetfibernorm", "stretchshear",
                               "bendtwist", "pinorient", "pressure", "shapematch" };
    return (find(nosmoothtypes, type) < 0);
}

void
createSurfaceStrutConstraint(const int edgegeo; const int ptnum; const string srcgrp;
                             const int outgeo; const string outgrp; const int dist)
{
    int seen[], dst[], nbrs[] = neighbours(edgegeo, ptnum);
    append(seen, ptnum);
    for (int i=0; i < dist; i++)
    {
        int newnbrs[];
        foreach(int n; nbrs)
        {
            if (!inpointgroup(edgegeo, srcgrp, n))
                continue;
            if (i == dist-1)
            {
                // Final iteration outwards, add point to dst if valid.
                if (n > ptnum)
                    append(dst, n);
            }
            // Only consider unseen neighbours
            else if(find(seen, n) < 0)
            {
                // Mark neighbor as seen and append its neighbors.
                append(seen, n);
                append(newnbrs, neighbours(edgegeo, n));
            }
        }
        nbrs = newnbrs;
    }

    // It's a few times faster to accumlate everything in dst
    // and do a binary search against sorted seen than checking
    // seen before inserting into dst.
    seen = sort(seen);
    // Faster to sort dst to filter duplicates than
    // checking uniqueness on every pass via linear search
    dst = sort(dst);
    int last = -1;
    foreach(int d; dst)
    {
        if (last == d)
            continue;
        if (findsorted(seen, d) < 0)
        {
            int prim = addprim(outgeo, "polyline", ptnum, d);
            setprimgroup(outgeo, outgrp, prim, 1);
        }
        last = d;
    }
}

#endif
