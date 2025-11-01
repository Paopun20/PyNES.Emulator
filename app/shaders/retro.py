shader = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform float u_time;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
    vec2 uv = v_uv;

    // Horizontal wavy distortion (signal wobble)
    uv.x += sin(uv.y * 6.0 + u_time * 2.0) * 0.002;

    // Curvature
    vec2 curve = uv * 2.0 - 1.0;
    curve *= 1.0 + 0.08 * dot(curve, curve);
    curve = (curve + 1.0) * 0.5;

    // Out of bounds = black
    if (curve.x < 0.0 || curve.x > 1.0 || curve.y < 0.0 || curve.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Chromatic Aberration (RGB split)
    float ca_strength = 0.003;
    vec3 col;
    col.r = texture(u_tex, curve + vec2(ca_strength, 0.0)).r;
    col.g = texture(u_tex, curve).g;
    col.b = texture(u_tex, curve - vec2(ca_strength, 0.0)).b;

    // Glow / Bloom
    vec2 glowOffset = 1.0 / vec2(textureSize(u_tex, 0)) * 2.0;
    vec3 glow = vec3(0.0);
    glow += texture(u_tex, curve + vec2(-glowOffset.x, 0.0)).rgb;
    glow += texture(u_tex, curve + vec2(glowOffset.x, 0.0)).rgb;
    glow += texture(u_tex, curve + vec2(0.0, -glowOffset.y)).rgb;
    glow += texture(u_tex, curve + vec2(0.0, glowOffset.y)).rgb;
    glow *= 0.25;
    col = mix(col, glow, 0.35);

    // Scanlines
    float scanline = 0.9 + 0.1 * sin((uv.y + u_time * 0.5) * 480.0);
    col *= scanline;

    // Flicker
    float flicker = 0.95 + 0.05 * sin(u_time * 100.0);
    col *= flicker;

    // Noise / Grain
    float noise = (rand(uv * u_time * 100.0) - 0.5) * 0.08;
    col += noise;

    // Vignette
    vec2 center = uv - 0.5;
    float vignette = 1.0 - dot(center, center) * 0.7;
    col *= vignette;

    // Slight color saturation boost
    col = pow(col, vec3(0.9)); // punchy retro colors

    fragColor = vec4(col, 1.0);
}
"""