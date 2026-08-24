#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

namespace {
constexpr int kCubics = 16;
constexpr int kSamples = 32;

__device__ inline void point(const float* control, int cubic, int sample, int samples, float& x, float& y) {
    const float t = float(sample) / float(samples - 1), u = 1.f - t;
    const float* c = control + cubic * 8;
    x = u*u*u*c[0] + 3*u*u*t*c[2] + 3*u*t*t*c[4] + t*t*t*c[6];
    y = u*u*u*c[1] + 3*u*u*t*c[3] + 3*u*t*t*c[5] + t*t*t*c[7];
}

__device__ inline void sample_path(const float* path, float* points, int samples) {
    for (int index = threadIdx.x; index < kCubics * samples; index += blockDim.x) {
        point(path, index / samples, index % samples, samples, points[index * 2], points[index * 2 + 1]);
    }
    __syncthreads();
}

__device__ inline void path_bounds(const float* path, float* bounds) {
    if (threadIdx.x == 0) {
        float min_x = path[0], min_y = path[1], max_x = path[0], max_y = path[1];
        for (int index = 1; index < kCubics * 4; ++index) {
            const float x = path[index * 2], y = path[index * 2 + 1];
            min_x = fminf(min_x, x);
            min_y = fminf(min_y, y);
            max_x = fmaxf(max_x, x);
            max_y = fmaxf(max_y, y);
        }
        bounds[0] = min_x;
        bounds[1] = min_y;
        bounds[2] = max_x;
        bounds[3] = max_y;
    }
    __syncthreads();
}

// Solve a cubic Bezier's y(t) == y as a polynomial, rather than replacing the
// curve with a polyline.  Splitting at the (at most two) derivative roots
// leaves monotonic intervals, on which a short bisection finds every crossing.
// This is deliberately a small independent implementation: the renderer only
// needs ray crossings, not a general-purpose path library.
__device__ inline float cubic_component(const float* path, int cubic, float t, int axis) {
    const float u = 1.f - t;
    const float* c = path + cubic * 8;
    return u*u*u*c[axis] + 3.f*u*u*t*c[2 + axis] +
           3.f*u*t*t*c[4 + axis] + t*t*t*c[6 + axis];
}

__device__ inline float cubic_derivative(const float* path, int cubic, float t, int axis) {
    const float u = 1.f - t;
    const float* c = path + cubic * 8;
    return 3.f*u*u*(c[2 + axis] - c[axis]) +
           6.f*u*t*(c[4 + axis] - c[2 + axis]) +
           3.f*t*t*(c[6 + axis] - c[4 + axis]);
}

__device__ inline float cubic_hull_distance_sq(const float* path, int cubic, float px, float py) {
    const float* c = path + cubic * 8;
    float min_x = c[0], max_x = c[0], min_y = c[1], max_y = c[1];
    for (int point_index = 1; point_index < 4; ++point_index) {
        min_x = fminf(min_x, c[point_index * 2]); max_x = fmaxf(max_x, c[point_index * 2]);
        min_y = fminf(min_y, c[point_index * 2 + 1]); max_y = fmaxf(max_y, c[point_index * 2 + 1]);
    }
    const float dx = px < min_x ? min_x - px : (px > max_x ? px - max_x : 0.f);
    const float dy = py < min_y ? min_y - py : (py > max_y ? py - max_y : 0.f);
    return dx*dx + dy*dy;
}

__device__ inline int ray_winding(const float* path, float px, float py) {
    int winding = 0;
    for (int cubic = 0; cubic < kCubics; ++cubic) {
        const float* c = path + cubic * 8;
        // dy/dt = A t^2 + B t + C.  Its roots partition y(t) into monotonic
        // intervals, so this finds cubic intersections without tessellation.
        const float A = 3.f * (-c[1] + 3.f*c[3] - 3.f*c[5] + c[7]);
        const float B = 6.f * (c[1] - 2.f*c[3] + c[5]);
        const float C = 3.f * (c[3] - c[1]);
        float cuts[4] = {0.f, 1.f, 1.f, 1.f};
        int cut_count = 2;
        if (fabsf(A) > 1e-8f) {
            const float disc = B*B - 4.f*A*C;
            if (disc > 0.f) {
                const float root = sqrtf(disc);
                const float t0 = (-B - root) / (2.f*A);
                const float t1 = (-B + root) / (2.f*A);
                if (t0 > 1e-6f && t0 < 1.f-1e-6f) cuts[cut_count++] = t0;
                if (t1 > 1e-6f && t1 < 1.f-1e-6f) cuts[cut_count++] = t1;
            }
        } else if (fabsf(B) > 1e-8f) {
            const float t = -C / B;
            if (t > 1e-6f && t < 1.f-1e-6f) cuts[cut_count++] = t;
        }
        // There are at most four cuts; insertion sort keeps the endpoint
        // convention stable at extrema and shared cubic endpoints.
        for (int i = 1; i < cut_count; ++i) {
            float value = cuts[i]; int j = i - 1;
            while (j >= 0 && cuts[j] > value) { cuts[j+1] = cuts[j]; --j; }
            cuts[j+1] = value;
        }
        for (int interval = 0; interval + 1 < cut_count; ++interval) {
            float lo = cuts[interval], hi = cuts[interval + 1];
            float yl = cubic_component(path, cubic, lo, 1) - py;
            float yh = cubic_component(path, cubic, hi, 1) - py;
            // Half-open interval: count a root only when the curve crosses
            // the horizontal ray, never when it merely touches at an extremum.
            if (!((yl <= 0.f && yh > 0.f) || (yh <= 0.f && yl > 0.f))) continue;
            // 1/256 in parameter space is already subpixel-accurate on the
            // 64px optimisation canvas; filtered coverage absorbs the small
            // residual before export is validated by Cairo.
            for (int iteration = 0; iteration < 8; ++iteration) {
                const float mid = .5f * (lo + hi);
                const float ym = cubic_component(path, cubic, mid, 1) - py;
                if ((yl <= 0.f && ym <= 0.f) || (yl >= 0.f && ym >= 0.f)) {
                    lo = mid; yl = ym;
                } else {
                    hi = mid;
                }
            }
            const float t = .5f * (lo + hi);
            if (cubic_component(path, cubic, t, 0) > px) {
                winding += cubic_derivative(path, cubic, t, 1) > 0.f ? 1 : -1;
            }
        }
    }
    return winding;
}

__global__ void coverage_forward_kernel(const float* controls, float* output, int batches,
                                        int height, int width, int subpixels,
                                        float x_base, float y_base, bool evenodd) {
    const int pixels = height * width, batch = blockIdx.x;
    if (batch >= batches) return;
    const float* path = controls + batch * kCubics * 8;
    __shared__ float bounds[4];
    path_bounds(path, bounds);
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        float coverage = 0.f;
        for (int subpixel = 0; subpixel < subpixels * subpixels; ++subpixel) {
            const float px = x_base + float(pixel % width) + (float(subpixel % subpixels) + .5f) / subpixels;
            const float py = y_base + float(pixel / width) + (float(subpixel / subpixels) + .5f) / subpixels;
            if (px >= bounds[0] && px <= bounds[2] && py >= bounds[1] && py <= bounds[3]) {
                const int w = ray_winding(path, px, py);
                coverage += evenodd ? float(abs(w) & 1) : float(w != 0);
            }
        }
        output[batch * pixels + pixel] = coverage / float(subpixels * subpixels);
    }
}

__global__ void coverage_backward_kernel(const float* controls, const float* upstream,
                                         float* gradients, int batches, int height, int width,
                                         int subpixels, float x_base, float y_base) {
    const int pixels = height * width, batch = blockIdx.x;
    if (batch >= batches) return;
    const float* path = controls + batch * kCubics * 8;
    float* gradient = gradients + batch * kCubics * 8;
    __shared__ float bounds[4];
    __shared__ float reduction[8][256];
    path_bounds(path, bounds);
    float accumulated[kCubics * 8] = {0.f};
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        const float d_output = upstream[batch * pixels + pixel] / float(subpixels * subpixels);
        // The coverage derivative is local.  Interior pixels have constant
        // coverage, so only a one-pixel band around the control hull does
        // useful work in the gradient pass.
        const float base_x = x_base + float(pixel % width);
        const float base_y = y_base + float(pixel / width);
        if (base_x < bounds[0] - 2.f || base_x > bounds[2] + 2.f ||
            base_y < bounds[1] - 2.f || base_y > bounds[3] + 2.f) continue;
        for (int subpixel = 0; subpixel < subpixels * subpixels; ++subpixel) {
            const float px = base_x + (float(subpixel % subpixels) + .5f) / subpixels;
            const float py = base_y + (float(subpixel / subpixels) + .5f) / subpixels;
            const int winding = ray_winding(path, px, py);
            const float sign = winding == 0 ? 1.f : -1.f;
            float best_distance = 1e20f, best_t = 0.f; int best_cubic = 0;
            for (int cubic = 0; cubic < kCubics; ++cubic) {
                if (cubic_hull_distance_sq(path, cubic, px, py) >= best_distance) continue;
                // A small fixed set of seeds, followed by Newton projection,
                // is a boundary-local closest-point solve.  It is independent
                // from the forward intersection calculation and only runs in
                // the antialias band.
                for (int seed = 0; seed < 3; ++seed) {
                    float t = .5f * seed;
                    for (int iteration = 0; iteration < 2; ++iteration) {
                        const float qx = cubic_component(path, cubic, t, 0) - px;
                        const float qy = cubic_component(path, cubic, t, 1) - py;
                        const float dx = cubic_derivative(path, cubic, t, 0);
                        const float dy = cubic_derivative(path, cubic, t, 1);
                        // Gauss-Newton is stable for the short local update
                        // steps used by SAMVG and avoids a global curve solve.
                        t = fminf(1.f, fmaxf(0.f, t - (qx*dx + qy*dy) / (dx*dx + dy*dy + 1e-6f)));
                    }
                    const float qx = cubic_component(path, cubic, t, 0) - px;
                    const float qy = cubic_component(path, cubic, t, 1) - py;
                    const float distance = qx*qx + qy*qy;
                    if (distance < best_distance) { best_distance = distance; best_t = t; best_cubic = cubic; }
                }
            }
            const float distance = sqrtf(best_distance + 1e-12f);
            const float alpha = 1.f / (1.f + expf(sign * distance / .25f));
            const float factor = d_output * (-sign) * alpha * (1.f-alpha) / .25f / distance;
            const float qx = cubic_component(path, best_cubic, best_t, 0) - px;
            const float qy = cubic_component(path, best_cubic, best_t, 1) - py;
            const float u = 1.f - best_t;
            const float basis[4] = {u*u*u, 3.f*u*u*best_t, 3.f*u*best_t*best_t, best_t*best_t*best_t};
            for (int control = 0; control < 4; ++control) {
                const int offset = best_cubic * 8 + control * 2;
                accumulated[offset] += factor * qx * basis[control];
                accumulated[offset + 1] += factor * qy * basis[control];
            }
        }
    }
    for (int cubic = 0; cubic < kCubics; ++cubic) {
        for (int component = 0; component < 8; ++component)
            reduction[component][threadIdx.x] = accumulated[cubic * 8 + component];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride)
                for (int component = 0; component < 8; ++component)
                    reduction[component][threadIdx.x] += reduction[component][threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            for (int component = 0; component < 8; ++component)
                gradient[cubic * 8 + component] = reduction[component][0];
        __syncthreads();
    }
}

__device__ inline void contours_bounds(const float* controls, int first, int last, float* bounds) {
    if (threadIdx.x == 0) {
        const float* first_path = controls + first * kCubics * 8;
        float min_x = first_path[0], min_y = first_path[1], max_x = first_path[0], max_y = first_path[1];
        for (int contour = first; contour < last; ++contour) {
            const float* path = controls + contour * kCubics * 8;
            for (int point_index = 0; point_index < kCubics * 4; ++point_index) {
                const float x = path[point_index * 2], y = path[point_index * 2 + 1];
                min_x = fminf(min_x, x); min_y = fminf(min_y, y);
                max_x = fmaxf(max_x, x); max_y = fmaxf(max_y, y);
            }
        }
        bounds[0] = min_x; bounds[1] = min_y; bounds[2] = max_x; bounds[3] = max_y;
    }
    __syncthreads();
}

__global__ void multi_coverage_forward_kernel(const float* controls, const int64_t* offsets,
                                              float* output, int paths, int height, int width,
                                              int subpixels, float x_base, float y_base,
                                              bool evenodd) {
    const int path_index = blockIdx.x, pixels = height * width;
    if (path_index >= paths) return;
    const int first = int(offsets[path_index]), last = int(offsets[path_index + 1]);
    __shared__ float bounds[4];
    contours_bounds(controls, first, last, bounds);
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        float coverage = 0.f;
        for (int subpixel = 0; subpixel < subpixels * subpixels; ++subpixel) {
            const float px = x_base + float(pixel % width) + (float(subpixel % subpixels) + .5f) / subpixels;
            const float py = y_base + float(pixel / width) + (float(subpixel / subpixels) + .5f) / subpixels;
            if (px < bounds[0] || px > bounds[2] || py < bounds[1] || py > bounds[3]) continue;
            int winding = 0;
            for (int contour = first; contour < last; ++contour)
                winding += ray_winding(controls + contour * kCubics * 8, px, py);
            coverage += evenodd ? float(abs(winding) & 1) : float(winding != 0);
        }
        output[path_index * pixels + pixel] = coverage / float(subpixels * subpixels);
    }
}

__global__ void multi_coverage_topology_forward_kernel(const float* controls, const int64_t* offsets,
                                                       float* output, uint16_t* topology, int paths,
                                                       int height, int width, int subpixels, float x_base,
                                                       float y_base, bool evenodd) {
    const int path_index = blockIdx.x, pixels = height * width;
    if (path_index >= paths) return;
    const int first = int(offsets[path_index]), last = int(offsets[path_index + 1]);
    __shared__ float bounds[4];
    contours_bounds(controls, first, last, bounds);
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        float coverage = 0.f; uint16_t mask = 0;
        for (int subpixel = 0; subpixel < subpixels * subpixels; ++subpixel) {
            const float px = x_base + float(pixel % width) + (float(subpixel % subpixels) + .5f) / subpixels;
            const float py = y_base + float(pixel / width) + (float(subpixel / subpixels) + .5f) / subpixels;
            int inside = 0;
            if (px >= bounds[0] && px <= bounds[2] && py >= bounds[1] && py <= bounds[3]) {
                int winding = 0;
                for (int contour = first; contour < last; ++contour)
                    winding += ray_winding(controls + contour * kCubics * 8, px, py);
                inside = evenodd ? (abs(winding) & 1) : (winding != 0);
            }
            mask |= uint16_t(inside) << subpixel;
            coverage += float(inside);
        }
        topology[path_index * pixels + pixel] = mask;
        output[path_index * pixels + pixel] = coverage / float(subpixels * subpixels);
    }
}

__global__ void multi_coverage_backward_kernel(const float* controls, const int64_t* offsets,
                                               const float* upstream, float* gradients, int paths,
                                               int height, int width, int subpixels,
                                               float x_base, float y_base) {
    const int path_index = blockIdx.x, pixels = height * width;
    if (path_index >= paths) return;
    const int first = int(offsets[path_index]), last = int(offsets[path_index + 1]);
    __shared__ float bounds[4];
    contours_bounds(controls, first, last, bounds);
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        const float d_output = upstream[path_index * pixels + pixel] / float(subpixels * subpixels);
        const float base_x = x_base + float(pixel % width), base_y = y_base + float(pixel / width);
        if (base_x < bounds[0] - 2.f || base_x > bounds[2] + 2.f ||
            base_y < bounds[1] - 2.f || base_y > bounds[3] + 2.f) continue;
        for (int subpixel = 0; subpixel < subpixels * subpixels; ++subpixel) {
            const float px = base_x + (float(subpixel % subpixels) + .5f) / subpixels;
            const float py = base_y + (float(subpixel / subpixels) + .5f) / subpixels;
            int winding = 0;
            for (int contour = first; contour < last; ++contour)
                winding += ray_winding(controls + contour * kCubics * 8, px, py);
            const float sign = winding == 0 ? 1.f : -1.f;
            float best_distance = 1e20f, best_t = 0.f; int best_contour = first, best_cubic = 0;
            for (int contour = first; contour < last; ++contour) {
                const float* path = controls + contour * kCubics * 8;
                for (int cubic = 0; cubic < kCubics; ++cubic) {
                    if (cubic_hull_distance_sq(path, cubic, px, py) >= best_distance) continue;
                    for (int seed = 0; seed < 3; ++seed) {
                    float t = .5f * seed;
                    for (int iteration = 0; iteration < 2; ++iteration) {
                        const float qx = cubic_component(path, cubic, t, 0) - px;
                        const float qy = cubic_component(path, cubic, t, 1) - py;
                        const float dx = cubic_derivative(path, cubic, t, 0);
                        const float dy = cubic_derivative(path, cubic, t, 1);
                        t = fminf(1.f, fmaxf(0.f, t - (qx*dx + qy*dy) / (dx*dx + dy*dy + 1e-6f)));
                    }
                    const float qx = cubic_component(path, cubic, t, 0) - px;
                    const float qy = cubic_component(path, cubic, t, 1) - py;
                    const float distance = qx*qx + qy*qy;
                    if (distance < best_distance) { best_distance = distance; best_t = t; best_contour = contour; best_cubic = cubic; }
                }
                }
            }
            const float distance = sqrtf(best_distance + 1e-12f);
            const float alpha = 1.f / (1.f + expf(sign * distance / .25f));
            const float factor = d_output * (-sign) * alpha * (1.f-alpha) / .25f / distance;
            const float* path = controls + best_contour * kCubics * 8;
            const float qx = cubic_component(path, best_cubic, best_t, 0) - px;
            const float qy = cubic_component(path, best_cubic, best_t, 1) - py;
            const float u = 1.f - best_t;
            const float basis[4] = {u*u*u, 3.f*u*u*best_t, 3.f*u*best_t*best_t, best_t*best_t*best_t};
            float* gradient = gradients + best_contour * kCubics * 8 + best_cubic * 8;
            for (int control = 0; control < 4; ++control) {
                atomicAdd(gradient + control * 2, factor * qx * basis[control]);
                atomicAdd(gradient + control * 2 + 1, factor * qy * basis[control]);
            }
        }
    }
}

// The fill decision is discrete, and its geometry derivative is represented by
// the nearest-boundary surrogate below.  Reusing the subpixel decisions from
// forward therefore avoids resolving every cubic/ray intersection again in
// backward, while preserving the same sign convention at the current step.
__global__ void multi_coverage_backward_topology_kernel(
    const float* controls, const int64_t* offsets, const int64_t* boundary_offsets,
    const int64_t* boundary_indices, const uint16_t* topology, const float* upstream,
    float* gradients, int paths, int height, int width, int subpixels, float x_base,
    float y_base) {
    const int path_index = blockIdx.x, pixels = height * width;
    if (path_index >= paths) return;
    const int first = int(offsets[path_index]), last = int(offsets[path_index + 1]);
    const int boundary_first = int(boundary_offsets[path_index]);
    const int boundary_last = int(boundary_offsets[path_index + 1]);
    __shared__ float bounds[4];
    contours_bounds(controls, first, last, bounds);
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        const float d_output = upstream[path_index * pixels + pixel] / float(subpixels * subpixels);
        const float base_x = x_base + float(pixel % width), base_y = y_base + float(pixel / width);
        if (base_x < bounds[0] - 2.f || base_x > bounds[2] + 2.f ||
            base_y < bounds[1] - 2.f || base_y > bounds[3] + 2.f) continue;
        const uint16_t mask = topology[path_index * pixels + pixel];
        if (boundary_first == boundary_last) continue;
        for (int subpixel = 0; subpixel < subpixels * subpixels; ++subpixel) {
            const float px = base_x + (float(subpixel % subpixels) + .5f) / subpixels;
            const float py = base_y + (float(subpixel / subpixels) + .5f) / subpixels;
            const float sign = (mask & (uint16_t(1) << subpixel)) ? -1.f : 1.f;
            float best_distance = 1e20f, best_t = 0.f; int best_contour = first, best_cubic = 0;
            for (int candidate = boundary_first; candidate < boundary_last; ++candidate) {
                const int local_contour = int(boundary_indices[candidate]);
                // Boundary indices are local to this packed tile.  Keep the
                // operator safe for externally supplied candidate tensors;
                // internal tile caches always satisfy this condition.
                if (local_contour < 0 || first + local_contour >= last) continue;
                const int contour = first + local_contour;
                const float* path = controls + contour * kCubics * 8;
                for (int cubic = 0; cubic < kCubics; ++cubic) {
                    if (cubic_hull_distance_sq(path, cubic, px, py) >= best_distance) continue;
                    for (int seed = 0; seed < 3; ++seed) {
                        float t = .5f * seed;
                        for (int iteration = 0; iteration < 2; ++iteration) {
                            const float qx = cubic_component(path, cubic, t, 0) - px;
                            const float qy = cubic_component(path, cubic, t, 1) - py;
                            const float dx = cubic_derivative(path, cubic, t, 0);
                            const float dy = cubic_derivative(path, cubic, t, 1);
                            t = fminf(1.f, fmaxf(0.f, t - (qx*dx + qy*dy) / (dx*dx + dy*dy + 1e-6f)));
                        }
                        const float qx = cubic_component(path, cubic, t, 0) - px;
                        const float qy = cubic_component(path, cubic, t, 1) - py;
                        const float distance = qx*qx + qy*qy;
                        if (distance < best_distance) { best_distance = distance; best_t = t; best_contour = contour; best_cubic = cubic; }
                    }
                }
            }
            const float distance = sqrtf(best_distance + 1e-12f);
            const float alpha = 1.f / (1.f + expf(sign * distance / .25f));
            const float factor = d_output * (-sign) * alpha * (1.f-alpha) / .25f / distance;
            const float* path = controls + best_contour * kCubics * 8;
            const float qx = cubic_component(path, best_cubic, best_t, 0) - px;
            const float qy = cubic_component(path, best_cubic, best_t, 1) - py;
            const float u = 1.f - best_t;
            const float basis[4] = {u*u*u, 3.f*u*u*best_t, 3.f*u*best_t*best_t, best_t*best_t*best_t};
            float* gradient = gradients + best_contour * kCubics * 8 + best_cubic * 8;
            for (int control = 0; control < 4; ++control) {
                atomicAdd(gradient + control * 2, factor * qx * basis[control]);
                atomicAdd(gradient + control * 2 + 1, factor * qy * basis[control]);
            }
        }
    }
}

__global__ void forward_kernel(const float* controls, float* output, int batches,
                               int height, int width, int samples, int subpixels,
                               float x_base, float y_base) {
    const int pixels = height * width;
    const int batch = blockIdx.x;
    if (batch >= batches) return;
    const float* path = controls + batch * kCubics * 8;
    __shared__ float points[kCubics * kSamples * 2];
    __shared__ float bounds[4];
    sample_path(path, points, samples);
    path_bounds(path, bounds);
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        for (int subpixel = 0; subpixel < subpixels * subpixels; ++subpixel) {
            const float px = x_base + float(pixel % width) +
                             (float(subpixel % subpixels) + .5f) / subpixels;
            const float py = y_base + float(pixel / width) +
                             (float(subpixel / subpixels) + .5f) / subpixels;
            if (px < bounds[0] || px > bounds[2] || py < bounds[1] || py > bounds[3]) {
                output[(batch * subpixels * subpixels + subpixel) * pixels + pixel] = 0.f;
                continue;
            }
            float winding = 0.f;
            for (int edge = 0; edge < kCubics * samples; ++edge) {
                const int next = (edge + 1) % (kCubics * samples);
                const float ax = points[edge * 2] - px, ay = points[edge * 2 + 1] - py;
                const float bx = points[next * 2] - px, by = points[next * 2 + 1] - py;
                winding += atan2f(ax * by - ay * bx, ax * bx + ay * by);
            }
            output[(batch * subpixels * subpixels + subpixel) * pixels + pixel] = winding;
        }
    }
}

__global__ void backward_kernel(const float* controls, const float* upstream,
                                float* gradients, int batches, int height, int width,
                                int samples, int subpixels, float x_base, float y_base) {
    const int pixels = height * width;
    const int batch = blockIdx.x;
    if (batch >= batches) return;
    const float* path = controls + batch * kCubics * 8;
    float* gradient = gradients + batch * kCubics * 8;
    __shared__ float points[kCubics * kSamples * 2];
    __shared__ float reduction[8][256];
    __shared__ float bounds[4];
    sample_path(path, points, samples);
    path_bounds(path, bounds);
    float accumulated[kCubics * 8] = {0.f};
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        for (int subpixel = 0; subpixel < subpixels * subpixels; ++subpixel) {
            const float px = x_base + float(pixel % width) +
                             (float(subpixel % subpixels) + .5f) / subpixels;
            const float py = y_base + float(pixel / width) +
                             (float(subpixel / subpixels) + .5f) / subpixels;
            if (px < bounds[0] || px > bounds[2] || py < bounds[1] || py > bounds[3]) {
                continue;
            }
            const float d_winding = upstream[(batch * subpixels * subpixels + subpixel) * pixels + pixel];
            for (int edge = 0; edge < kCubics * samples; ++edge) {
                const int next = (edge + 1) % (kCubics * samples);
                const float ax = points[edge * 2] - px, ay = points[edge * 2 + 1] - py;
                const float bx = points[next * 2] - px, by = points[next * 2 + 1] - py;
                const float cross = ax * by - ay * bx, dot = ax * bx + ay * by;
                const float scale = d_winding / (cross * cross + dot * dot + 1e-20f);
                const float dcross = scale * dot, ddot = -scale * cross;
                const float gx_a = dcross * by + ddot * bx;
                const float gy_a = -dcross * bx + ddot * by;
                const float gx_b = -dcross * ay + ddot * ax;
                const float gy_b = dcross * ax + ddot * ay;
                const float ta = float(edge % samples) / float(samples - 1), ua = 1.f - ta;
                const float tb = float(next % samples) / float(samples - 1), ub = 1.f - tb;
                const float ba[4] = {ua*ua*ua, 3*ua*ua*ta, 3*ua*ta*ta, ta*ta*ta};
                const float bb[4] = {ub*ub*ub, 3*ub*ub*tb, 3*ub*tb*tb, tb*tb*tb};
                for (int control = 0; control < 4; ++control) {
                    const int a = (edge / samples) * 8 + control * 2;
                    const int b = (next / samples) * 8 + control * 2;
                    accumulated[a] += ba[control] * gx_a;
                    accumulated[a + 1] += ba[control] * gy_a;
                    accumulated[b] += bb[control] * gx_b;
                    accumulated[b + 1] += bb[control] * gy_b;
                }
            }
        }
    }
    for (int cubic = 0; cubic < kCubics; ++cubic) {
        for (int component = 0; component < 8; ++component) {
            reduction[component][threadIdx.x] = accumulated[cubic * 8 + component];
        }
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                for (int component = 0; component < 8; ++component) {
                    reduction[component][threadIdx.x] += reduction[component][threadIdx.x + stride];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            for (int component = 0; component < 8; ++component) {
                gradient[cubic * 8 + component] = reduction[component][0];
            }
        }
        __syncthreads();
    }
}
}

torch::Tensor forward(torch::Tensor controls, int64_t height, int64_t width, int64_t samples,
                      double x_origin, double y_origin) {
    TORCH_CHECK(controls.is_cuda() && controls.scalar_type() == torch::kFloat32);
    at::cuda::CUDAGuard guard(controls.device());
    auto output = torch::zeros({controls.size(0), height, width}, controls.options());
    constexpr int threads = 256;
    forward_kernel<<<controls.size(0), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), output.data_ptr<float>(), controls.size(0), height, width, samples, 1,
        float(x_origin) - .5f, float(y_origin) - .5f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor backward(torch::Tensor controls, torch::Tensor upstream, int64_t height,
                       int64_t width, int64_t samples, double x_origin, double y_origin) {
    at::cuda::CUDAGuard guard(controls.device());
    auto gradients = torch::zeros_like(controls);
    constexpr int threads = 256;
    backward_kernel<<<controls.size(0), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), upstream.data_ptr<float>(), gradients.data_ptr<float>(),
        controls.size(0), height, width, samples, 1, float(x_origin) - .5f, float(y_origin) - .5f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gradients;
}

torch::Tensor forwards(torch::Tensor controls, int64_t height, int64_t width,
                       int64_t samples, int64_t subpixels, double x_origin,
                       double y_origin) {
    TORCH_CHECK(subpixels >= 1 && subpixels <= 4);
    at::cuda::CUDAGuard guard(controls.device());
    auto output = torch::zeros(
        {controls.size(0), subpixels * subpixels, height, width}, controls.options());
    constexpr int threads = 256;
    forward_kernel<<<controls.size(0), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), output.data_ptr<float>(), controls.size(0), height, width,
        samples, subpixels, float(x_origin), float(y_origin));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor backwards(torch::Tensor controls, torch::Tensor upstream, int64_t height,
                        int64_t width, int64_t samples, int64_t subpixels,
                        double x_origin, double y_origin) {
    at::cuda::CUDAGuard guard(controls.device());
    auto gradients = torch::zeros_like(controls);
    constexpr int threads = 256;
    backward_kernel<<<controls.size(0), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), upstream.data_ptr<float>(), gradients.data_ptr<float>(),
        controls.size(0), height, width, samples, subpixels, float(x_origin), float(y_origin));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gradients;
}

torch::Tensor coverage_forward(torch::Tensor controls, int64_t height, int64_t width,
                               int64_t subpixels, double x_origin, double y_origin,
                               bool evenodd) {
    TORCH_CHECK(controls.is_cuda() && controls.scalar_type() == torch::kFloat32);
    TORCH_CHECK(subpixels >= 1 && subpixels <= 4);
    at::cuda::CUDAGuard guard(controls.device());
    auto output = torch::zeros({controls.size(0), height, width}, controls.options());
    coverage_forward_kernel<<<controls.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), output.data_ptr<float>(), controls.size(0), height, width,
        subpixels, float(x_origin), float(y_origin), evenodd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor coverage_backward(torch::Tensor controls, torch::Tensor upstream, int64_t height,
                                int64_t width, int64_t subpixels, double x_origin,
                                double y_origin) {
    at::cuda::CUDAGuard guard(controls.device());
    auto gradients = torch::zeros_like(controls);
    coverage_backward_kernel<<<controls.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), upstream.data_ptr<float>(), gradients.data_ptr<float>(),
        controls.size(0), height, width, subpixels, float(x_origin), float(y_origin));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gradients;
}

torch::Tensor multi_coverage_forward(torch::Tensor controls, torch::Tensor offsets,
                                     int64_t height, int64_t width, int64_t subpixels,
                                     double x_origin, double y_origin, bool evenodd) {
    TORCH_CHECK(controls.is_cuda() && controls.scalar_type() == torch::kFloat32);
    TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt64);
    TORCH_CHECK(offsets.dim() == 1 && offsets.size(0) >= 2);
    at::cuda::CUDAGuard guard(controls.device());
    const auto paths = offsets.size(0) - 1;
    auto output = torch::zeros({paths, height, width}, controls.options());
    multi_coverage_forward_kernel<<<paths, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), offsets.data_ptr<int64_t>(), output.data_ptr<float>(), paths,
        height, width, subpixels, float(x_origin), float(y_origin), evenodd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> multi_coverage_forward_topology(
    torch::Tensor controls, torch::Tensor offsets, int64_t height, int64_t width,
    int64_t subpixels, double x_origin, double y_origin, bool evenodd,
    torch::Tensor topology) {
    at::cuda::CUDAGuard guard(controls.device());
    const auto paths = offsets.size(0) - 1;
    TORCH_CHECK(topology.is_cuda() && topology.scalar_type() == torch::kUInt16);
    TORCH_CHECK(
        topology.dim() == 3 && topology.size(0) == paths &&
        topology.size(1) == height && topology.size(2) == width,
        "topology workspace has the wrong shape");
    auto output = torch::zeros({paths, height, width}, controls.options());
    multi_coverage_topology_forward_kernel<<<paths, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), offsets.data_ptr<int64_t>(), output.data_ptr<float>(),
        topology.data_ptr<uint16_t>(), paths, height, width, subpixels, float(x_origin),
        float(y_origin), evenodd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, topology};
}

torch::Tensor multi_coverage_backward(torch::Tensor controls, torch::Tensor offsets,
                                      torch::Tensor upstream, int64_t height, int64_t width,
                                      int64_t subpixels, double x_origin, double y_origin) {
    at::cuda::CUDAGuard guard(controls.device());
    auto gradients = torch::zeros_like(controls);
    const auto paths = offsets.size(0) - 1;
    multi_coverage_backward_kernel<<<paths, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), offsets.data_ptr<int64_t>(), upstream.data_ptr<float>(),
        gradients.data_ptr<float>(), paths, height, width, subpixels, float(x_origin), float(y_origin));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gradients;
}

torch::Tensor multi_coverage_backward_topology(
    torch::Tensor controls, torch::Tensor offsets, torch::Tensor boundary_offsets,
    torch::Tensor boundary_indices, torch::Tensor topology, torch::Tensor upstream,
    int64_t height, int64_t width, int64_t subpixels, double x_origin, double y_origin) {
    TORCH_CHECK(topology.is_cuda() && topology.scalar_type() == torch::kUInt16);
    TORCH_CHECK(boundary_offsets.is_cuda() && boundary_offsets.scalar_type() == torch::kInt64);
    TORCH_CHECK(boundary_indices.is_cuda() && boundary_indices.scalar_type() == torch::kInt64);
    TORCH_CHECK(boundary_offsets.dim() == 1 && boundary_offsets.size(0) == offsets.size(0));
    at::cuda::CUDAGuard guard(controls.device());
    auto gradients = torch::zeros_like(controls);
    const auto paths = offsets.size(0) - 1;
    multi_coverage_backward_topology_kernel<<<paths, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        controls.data_ptr<float>(), offsets.data_ptr<int64_t>(),
        boundary_offsets.data_ptr<int64_t>(), boundary_indices.data_ptr<int64_t>(),
        topology.data_ptr<uint16_t>(), upstream.data_ptr<float>(), gradients.data_ptr<float>(),
        paths, height, width, subpixels, float(x_origin), float(y_origin));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gradients;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("backward", &backward);
    m.def("forwards", &forwards);
    m.def("backwards", &backwards);
    m.def("coverage_forward", &coverage_forward);
    m.def("coverage_backward", &coverage_backward);
    m.def("multi_coverage_forward", &multi_coverage_forward);
    m.def("multi_coverage_forward_topology", &multi_coverage_forward_topology);
    m.def("multi_coverage_backward", &multi_coverage_backward);
    m.def("multi_coverage_backward_topology", &multi_coverage_backward_topology);
}
