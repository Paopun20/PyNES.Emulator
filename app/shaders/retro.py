from shaders.shader_class import Shader

@Shader
class retro:
    """
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

        // Slight wobble for analog signal
        uv.x += sin(uv.y * 5.0 + u_time * 1.5) * 0.0015;

        // Curvature
        vec2 curve = uv * 2.0 - 1.0;
        curve *= 1.0 + 0.07 * dot(curve, curve);
        curve = (curve + 1.0) * 0.5;

        if (curve.x < 0.0 || curve.x > 1.0 || curve.y < 0.0 || curve.y > 1.0) {
            fragColor = vec4(0.0);
            return;
        }

        // Chromatic aberration with edge dependence
        float ca_strength = 0.002 + 0.002 * length(curve - 0.5);
        vec3 col;
        col.r = texture(u_tex, curve + vec2(ca_strength, 0.0)).r;
        col.g = texture(u_tex, curve).g;
        col.b = texture(u_tex, curve - vec2(ca_strength, 0.0)).b;

        // Simulate color bleed (retro composite effect)
        vec3 bleed = vec3(
            texture(u_tex, curve + vec2(0.002, 0.0)).r,
            texture(u_tex, curve).g,
            texture(u_tex, curve - vec2(0.002, 0.0)).b
        );
        col = mix(col, bleed, 0.25);

        // Scanlines — darker and tighter like CRT
        float scanline = 0.85 + 0.15 * sin((uv.y) * 720.0);
        col *= scanline;

        // Slight flicker
        col *= 0.97 + 0.03 * sin(u_time * 60.0);

        // Add analog noise
        float noise = (rand(uv * u_time * 50.0) - 0.5) * 0.06;
        col += noise;

        // Mild vignette
        vec2 center = uv - 0.5;
        float vignette = 1.0 - dot(center, center) * 0.5;
        col *= vignette;

        // Slight gamma/saturation tweak
        col = pow(col, vec3(0.92));

        fragColor = vec4(col, 1.0);
    }
    """