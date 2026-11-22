#version 450

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 projection;
    vec2 focalPixels;
    vec2 viewportPixels;
} camera;

struct Gaussian {
    vec4 meanOpacity;
    vec4 scale;
    vec4 rotation;
    vec4 color;
};

layout(std430, set = 0, binding = 1) readonly buffer GaussianBuffer { Gaussian values[]; } gaussians;
layout(location = 0) out vec2 localPosition;
layout(location = 1) out vec4 splatColor;

mat3 quaternionRotation(vec4 q) {
    q = normalize(q);
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return mat3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
        2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
        2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
}

void main() {
    const vec2 corners[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(-1, 1),
                                     vec2(-1, 1), vec2(1, -1), vec2(1, 1));
    Gaussian gaussian = gaussians.values[gl_InstanceIndex];
    vec4 cameraMean = camera.view * vec4(gaussian.meanOpacity.xyz, 1.0);
    if (cameraMean.z >= -1e-4) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        localPosition = vec2(0.0);
        splatColor = vec4(0.0);
        return;
    }

    mat3 rotation = quaternionRotation(gaussian.rotation);
    mat3 scale = mat3(gaussian.scale.x, 0, 0, 0, gaussian.scale.y, 0, 0, 0, gaussian.scale.z);
    mat3 covarianceWorld = rotation * scale * scale * transpose(rotation);
    mat3 viewRotation = mat3(camera.view);
    mat3 covarianceCamera = viewRotation * covarianceWorld * transpose(viewRotation);
    float inverseZ = 1.0 / cameraMean.z;
    mat3x2 jacobian = mat3x2(camera.focalPixels.x * inverseZ, 0.0,
                              -camera.focalPixels.x * cameraMean.x * inverseZ * inverseZ,
                              0.0, camera.focalPixels.y * inverseZ,
                              -camera.focalPixels.y * cameraMean.y * inverseZ * inverseZ);
    mat2 covariance = jacobian * covarianceCamera * transpose(jacobian);
    covariance[0][0] += 0.3;
    covariance[1][1] += 0.3;

    float trace = covariance[0][0] + covariance[1][1];
    float determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
    float root = sqrt(max(0.0, 0.25 * trace * trace - determinant));
    float lambda0 = max(0.3, 0.5 * trace + root);
    float lambda1 = max(0.3, 0.5 * trace - root);
    vec2 axis0 = abs(covariance[0][1]) > 1e-6
                     ? normalize(vec2(lambda0 - covariance[1][1], covariance[0][1]))
                     : vec2(1.0, 0.0);
    vec2 axis1 = vec2(-axis0.y, axis0.x);
    vec2 corner = corners[gl_VertexIndex];
    vec2 pixelOffset = 3.0 * (axis0 * sqrt(lambda0) * corner.x + axis1 * sqrt(lambda1) * corner.y);
    vec4 clip = camera.projection * cameraMean;
    clip.xy += (2.0 * pixelOffset / camera.viewportPixels) * clip.w;
    gl_Position = clip;
    localPosition = corner * 3.0;
    splatColor = vec4(gaussian.color.rgb, gaussian.meanOpacity.w);
}
