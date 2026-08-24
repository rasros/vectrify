#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

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

__global__ void forward_kernel(const float* controls, float* output, int batches,
                               int height, int width, int samples, float xo, float yo) {
    const int pixels = height * width;
    const int batch = blockIdx.x;
    if (batch >= batches) return;
    const float* path = controls + batch * kCubics * 8;
    __shared__ float points[kCubics * kSamples * 2];
    sample_path(path, points, samples);
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        const float px = xo + float(pixel % width), py = yo + float(pixel / width);
        float winding = 0.f;
        for (int edge = 0; edge < kCubics * samples; ++edge) {
            const int next = (edge + 1) % (kCubics * samples);
            const float ax = points[edge * 2] - px, ay = points[edge * 2 + 1] - py;
            const float bx = points[next * 2] - px, by = points[next * 2 + 1] - py;
            winding += atan2f(ax * by - ay * bx, ax * bx + ay * by);
        }
        output[batch * pixels + pixel] = winding;
    }
}

__global__ void backward_kernel(const float* controls, const float* upstream,
                                float* gradients, int batches, int height, int width,
                                int samples, float xo, float yo) {
    const int pixels = height * width;
    const int batch = blockIdx.x;
    if (batch >= batches) return;
    const float* path = controls + batch * kCubics * 8;
    float* gradient = gradients + batch * kCubics * 8;
    __shared__ float points[kCubics * kSamples * 2];
    __shared__ float reduction[8][256];
    sample_path(path, points, samples);
    float accumulated[kCubics * 8] = {0.f};
    for (int pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        const float px = xo + float(pixel % width), py = yo + float(pixel / width);
        const float d_winding = upstream[batch * pixels + pixel];
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
        controls.data_ptr<float>(), output.data_ptr<float>(), controls.size(0), height, width, samples,
        float(x_origin), float(y_origin));
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
        controls.size(0), height, width, samples, float(x_origin), float(y_origin));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gradients;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("backward", &backward);
}
