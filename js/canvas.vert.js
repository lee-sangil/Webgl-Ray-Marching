const vertex = /* glsl */ `#version 300 es

precision mediump float;

in vec2 a_position;
in vec3 a_normal;

// uniform vec2 u_resolution;

out vec2 v_position;
out vec3 v_normal;

void main() {
    // 위치를 픽셀에서 0.0과 1.0사이로 변환
    vec2 zeroToOne = a_position;// / u_resolution;

    // 0->1에서 -1->+1로 변환 (클립 공간)
    vec2 clipSpace = zeroToOne * 2.0 - 1.0;
    clipSpace.y *= -1.;

    v_position = a_position;
    v_normal = a_normal;

    gl_Position = vec4(clipSpace, 0., 1.);
}
`
export default vertex