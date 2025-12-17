from objects.shaderclass import Shader

@Shader("Jello Madness shader: everything wobbles, stretches, and spins in rainbow chaos.", "Paopaodev")
class jello_madness:
    """
    #version 330 core

    in vec2 v_uv;
    out vec4 fragColor;
    uniform sampler2D u_tex;
    uniform float u_time;

    float rand(vec2 co) {
        return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
    }

    void main() {
        vec2 uv = v_uv;

        // Jello wobble effect
        uv.x += sin(u_time * 3.0 + uv.y * 20.0) * 0.05;
        uv.y += cos(u_time * 4.0 + uv.x * 25.0) * 0.05;

        // Stretching / squish effect
        uv += vec2(
            sin(u_time + uv.y * 10.0) * 0.03,
            cos(u_time + uv.x * 12.0) * 0.03
        );

        // Sample texture
        vec3 col = texture(u_tex, uv).rgb;

        // Random jello color pops
        float r = rand(uv + u_time);
        if (r > 0.95) col = vec3(1.0, 0.0, 0.0);
        float g = rand(uv + u_time * 1.5);
        if (g > 0.95) col = vec3(0.0, 1.0, 0.0);
        float b = rand(uv + u_time * 2.0);
        if (b > 0.95) col = vec3(0.0, 0.0, 1.0);

        // Rainbow twist
        float hueShift = u_time * 0.5;
        vec3 rainbow = vec3(
            0.5 + 0.5 * sin(hueShift + uv.x * 10.0),
            0.5 + 0.5 * sin(hueShift + uv.y * 10.0 + 2.0),
            0.5 + 0.5 * sin(hueShift + uv.x * 5.0 + 4.0)
        );
        col = mix(col, rainbow, 0.5);

        // Tiny jitter for chaotic motion
        col += (rand(uv * u_time * 20.0) - 0.5) * 0.05;

        fragColor = vec4(col, 1.0);
    }
    """
