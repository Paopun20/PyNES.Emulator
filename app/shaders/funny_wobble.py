from objects.shader_class import Shader

@Shader("Funny wobbly shader using screen resolution to scale effects.")
class funny_wobble:
    """
    #version 330 core

    in vec2 v_uv;
    out vec4 fragColor;
    uniform sampler2D u_tex;
    uniform float u_time;
    uniform vec2 u_resolution;

    float rand(vec2 co) {
        return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
    }

    void main() {
        // Normalize coordinates to screen resolution
        vec2 uv = v_uv;
        vec2 aspect = u_resolution.xy / min(u_resolution.x, u_resolution.y);
        uv = (uv - 0.5) * aspect + 0.5;

        // Wobble effect scaled by resolution
        uv.y += sin(uv.x * 20.0 * aspect.x + u_time * 3.0) * 0.03;
        uv.x += cos(uv.y * 15.0 * aspect.y + u_time * 2.5) * 0.03;

        // Melting/stretching effect
        uv += vec2(
            sin(u_time + uv.y * 10.0 * aspect.y) * 0.02,
            cos(u_time + uv.x * 12.0 * aspect.x) * 0.02
        );

        // Sample texture
        vec3 col = texture(u_tex, uv).rgb;

        // Random color pops scaled with resolution
        float r = rand(uv * aspect + u_time);
        if (r > 0.95) col = vec3(1.0, 0.0, 0.0);
        float g = rand(uv * aspect + u_time * 1.5);
        if (g > 0.96) col = vec3(0.0, 1.0, 0.0);
        float b = rand(uv * aspect + u_time * 2.0);
        if (b > 0.97) col = vec3(0.0, 0.0, 1.0);

        // Slight hue shift
        col = vec3(
            col.r * 0.8 + 0.2 * sin(u_time),
            col.g * 0.8 + 0.2 * sin(u_time + 2.0),
            col.b * 0.8 + 0.2 * sin(u_time + 4.0)
        );

        fragColor = vec4(col, 1.0);
    }
    """
