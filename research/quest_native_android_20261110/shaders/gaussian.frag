#version 450

layout(location = 0) in vec2 localPosition;
layout(location = 1) in vec4 splatColor;
layout(location = 0) out vec4 outputColor;

void main() {
    float alpha = min(0.99, splatColor.a * exp(-0.5 * dot(localPosition, localPosition)));
    if (alpha < (1.0 / 255.0)) discard;
    outputColor = vec4(splatColor.rgb * alpha, alpha);
}
