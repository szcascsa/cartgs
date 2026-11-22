#version 450
layout(std430,set=0,binding=3) readonly buffer Image { vec4 values[]; } image;
layout(push_constant) uniform Params { uint width,height,tilesX,srgb; vec4 background; } pc;
layout(location=0) out vec4 result;
void main() {
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    vec3 value=clamp(image.values[pixel.y*pc.width+pixel.x].rgb,0.0,1.0);
    if(pc.srgb!=0) value=mix(value/12.92,pow((value+0.055)/1.055,vec3(2.4)),greaterThan(value,vec3(0.04045)));
    result=vec4(value,1);
}
