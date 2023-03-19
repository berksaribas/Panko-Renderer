#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <glm/glm.hpp>

inline void find_min_max(float x0, float x1, float x2, float& min, float& max)
{
    min = max = x0;
    if (x1 < min)
        min = x1;
    if (x1 > max)
        max = x1;
    if (x2 < min)
        min = x2;
    if (x2 > max)
        max = x2;
}

inline bool plane_box_overlap(glm::vec3 normal, glm::vec3 vert, glm::vec3 maxbox)
{
    glm::vec3 vmin, vmax;
    float v;
    for (size_t q = 0; q < 3; q++)
    {
        v = vert[q];
        if (normal[q] > 0.0f)
        {
            vmin[q] = -maxbox[q] - v;
            vmax[q] = maxbox[q] - v;
        }
        else
        {
            vmin[q] = maxbox[q] - v;
            vmax[q] = -maxbox[q] - v;
        }
    }
    if (glm::dot(normal, vmin) > 0.0f)
        return false;
    if (glm::dot(normal, vmax) >= 0.0f)
        return true;

    return false;
}

/*======================== X-tests ========================*/

inline bool axis_test_x01(float a, float b, float fa, float fb, const glm::vec3& v0,
                          const glm::vec3& v2, const glm::vec3& boxhalfsize, float& rad,
                          float& min, float& max, float& p0, float& p2)
{
    p0 = a * v0.y - b * v0.z;
    p2 = a * v2.y - b * v2.z;
    if (p0 < p2)
    {
        min = p0;
        max = p2;
    }
    else
    {
        min = p2;
        max = p0;
    }
    rad = fa * boxhalfsize.y + fb * boxhalfsize.z;
    if (min > rad || max < -rad)
        return false;
    return true;
}
inline bool axis_test_x2(float a, float b, float fa, float fb, const glm::vec3& v0,
                         const glm::vec3& v1, const glm::vec3& boxhalfsize, float& rad,
                         float& min, float& max, float& p0, float& p1)
{
    p0 = a * v0.y - b * v0.z;
    p1 = a * v1.y - b * v1.z;
    if (p0 < p1)
    {
        min = p0;
        max = p1;
    }
    else
    {
        min = p1;
        max = p0;
    }
    rad = fa * boxhalfsize.y + fb * boxhalfsize.z;
    if (min > rad || max < -rad)
        return false;
    return true;
}

/*======================== Y-tests ========================*/

inline bool axis_test_y02(float a, float b, float fa, float fb, const glm::vec3& v0,
                          const glm::vec3& v2, const glm::vec3& boxhalfsize, float& rad,
                          float& min, float& max, float& p0, float& p2)
{
    p0 = -a * v0.x + b * v0.z;
    p2 = -a * v2.x + b * v2.z;
    if (p0 < p2)
    {
        min = p0;
        max = p2;
    }
    else
    {
        min = p2;
        max = p0;
    }
    rad = fa * boxhalfsize.x + fb * boxhalfsize.z;
    if (min > rad || max < -rad)
        return false;
    return true;
}

inline bool axis_test_y1(float a, float b, float fa, float fb, const glm::vec3& v0,
                         const glm::vec3& v1, const glm::vec3& boxhalfsize, float& rad,
                         float& min, float& max, float& p0, float& p1)
{
    p0 = -a * v0.x + b * v0.z;
    p1 = -a * v1.x + b * v1.z;
    if (p0 < p1)
    {
        min = p0;
        max = p1;
    }
    else
    {
        min = p1;
        max = p0;
    }
    rad = fa * boxhalfsize.x + fb * boxhalfsize.z;
    if (min > rad || max < -rad)
        return false;
    return true;
}

/*======================== Z-tests ========================*/
inline bool axis_test_z12(float a, float b, float fa, float fb, const glm::vec3& v1,
                          const glm::vec3& v2, const glm::vec3& boxhalfsize, float& rad,
                          float& min, float& max, float& p1, float& p2)
{
    p1 = a * v1.x - b * v1.y;
    p2 = a * v2.x - b * v2.y;
    if (p1 < p2)
    {
        min = p1;
        max = p2;
    }
    else
    {
        min = p2;
        max = p1;
    }
    rad = fa * boxhalfsize.x + fb * boxhalfsize.y;
    if (min > rad || max < -rad)
        return false;
    return true;
}

inline bool axis_test_z0(float a, float b, float fa, float fb, const glm::vec3& v0,
                         const glm::vec3& v1, const glm::vec3& boxhalfsize, float& rad,
                         float& min, float& max, float& p0, float& p1)
{
    p0 = a * v0.x - b * v0.y;
    p1 = a * v1.x - b * v1.y;
    if (p0 < p1)
    {
        min = p0;
        max = p1;
    }
    else
    {
        min = p1;
        max = p0;
    }
    rad = fa * boxhalfsize.x + fb * boxhalfsize.y;
    if (min > rad || max < -rad)
        return false;
    return true;
}

bool tri_box_overlap(glm::vec3 boxcenter, glm::vec3 boxhalfsize, glm::vec3 tv0, glm::vec3 tv1,
                     glm::vec3 tv2)
{
    /*    use separating axis theorem to test overlap between triangle and box */
    /*    need to test for overlap in these directions: */
    /*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
    /*       we do not even need to test these) */
    /*    2) normal of the triangle */
    /*    3) crossproduct(edge from tri, {x,y,z}-directin) */
    /*       this gives 3x3=9 more tests */
    glm::vec3 v0, v1, v2;
    float min, max, p0, p1, p2, rad, fex, fey, fez;
    glm::vec3 normal, e0, e1, e2;

    /* This is the fastest branch on Sun */
    /* move everything so that the boxcenter is in (0,0,0) */
    v0 = tv0 - boxcenter;
    v1 = tv1 - boxcenter;
    v2 = tv2 - boxcenter;

    /* compute triangle edges */
    e0 = v1 - v0;
    e1 = v2 - v1;
    e2 = v0 - v2;

    /* Bullet 3:  */
    /*  test the 9 tests first (this was faster) */
    fex = fabsf(e0.x);
    fey = fabsf(e0.y);
    fez = fabsf(e0.z);

    if (!axis_test_x01(e0.z, e0.y, fez, fey, v0, v2, boxhalfsize, rad, min, max, p0, p2))
        return false;
    if (!axis_test_y02(e0.z, e0.x, fez, fex, v0, v2, boxhalfsize, rad, min, max, p0, p2))
        return false;
    if (!axis_test_z12(e0.y, e0.x, fey, fex, v1, v2, boxhalfsize, rad, min, max, p1, p2))
        return false;

    fex = fabsf(e1.x);
    fey = fabsf(e1.y);
    fez = fabsf(e1.z);

    if (!axis_test_x01(e1.z, e1.y, fez, fey, v0, v2, boxhalfsize, rad, min, max, p0, p2))
        return false;
    if (!axis_test_y02(e1.z, e1.x, fez, fex, v0, v2, boxhalfsize, rad, min, max, p0, p2))
        return false;
    if (!axis_test_z0(e1.y, e1.x, fey, fex, v0, v1, boxhalfsize, rad, min, max, p0, p1))
        return false;

    fex = fabsf(e2.x);
    fey = fabsf(e2.y);
    fez = fabsf(e2.z);
    if (!axis_test_x2(e2.z, e2.y, fez, fey, v0, v1, boxhalfsize, rad, min, max, p0, p1))
        return false;
    if (!axis_test_y1(e2.z, e2.x, fez, fex, v0, v1, boxhalfsize, rad, min, max, p0, p1))
        return false;
    if (!axis_test_z12(e2.y, e2.x, fey, fex, v1, v2, boxhalfsize, rad, min, max, p1, p2))
        return false;

    /* Bullet 1: */
    /*  first test overlap in the {x,y,z}-directions */
    /*  find min, max of the triangle each direction, and test for overlap in */
    /*  that direction -- this is equivalent to testing a minimal AABB around */
    /*  the triangle against the AABB */

    /* test in X-direction */
    find_min_max(v0.x, v1.x, v2.x, min, max);
    if (min > boxhalfsize.x || max < -boxhalfsize.x)
        return false;

    /* test in Y-direction */
    find_min_max(v0.y, v1.y, v2.y, min, max);
    if (min > boxhalfsize.y || max < -boxhalfsize.y)
        return false;

    /* test in Z-direction */
    find_min_max(v0.z, v1.z, v2.z, min, max);
    if (min > boxhalfsize.z || max < -boxhalfsize.z)
        return false;

    /* Bullet 2: */
    /*  test if the box intersects the plane of the triangle */
    /*  compute plane equation of triangle: normal*x+d=0 */
    normal = glm::cross(e0, e1);
    if (!plane_box_overlap(normal, v0, boxhalfsize))
        return false;

    return true; /* box and triangle overlaps */
}

inline float edge_function(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c)
{
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
}

// Christer Ericson's Real-Time Collision Detection
glm::vec3 calculate_barycentric(glm::vec2 p, glm::vec2 a, glm::vec2 b, glm::vec2 c)
{
    // glm::vec2 v0 = b - a, v1 = c - a, v2 = p - a;
    // float d00 = glm::dot(v0, v0);
    // float d01 = glm::dot(v0, v1);
    // float d11 = glm::dot(v1, v1);
    // float d20 = glm::dot(v2, v0);
    // float d21 = glm::dot(v2, v1);
    // float denom = d00 * d11 - d01 * d01;
    //
    // float v = (d11 * d20 - d01 * d21) / denom;
    // float w = (d00 * d21 - d01 * d20) / denom;
    // float u = 1.0f - v - w;
    // return { u, v, w };

    glm::vec2 v0 = b - a, v1 = c - a, v2 = p - a;
    float den = v0.x * v1.y - v1.x * v0.y;
    float v = (v2.x * v1.y - v1.x * v2.y) / den;
    float w = (v0.x * v2.y - v2.x * v0.y) / den;
    float u = 1.0f - v - w;
    return {u, v, w};
}

glm::vec3 apply_barycentric(glm::vec3 barycentricCoordinates, glm::vec3 a, glm::vec3 b,
                            glm::vec3 c)
{
    return {barycentricCoordinates.x * a.x + barycentricCoordinates.y * b.x +
                barycentricCoordinates.z * c.x,
            barycentricCoordinates.x * a.y + barycentricCoordinates.y * b.y +
                barycentricCoordinates.z * c.y,
            barycentricCoordinates.x * a.z + barycentricCoordinates.y * b.z +
                barycentricCoordinates.z * c.z};
}

#endif