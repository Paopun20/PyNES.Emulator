from objects.shaderclass import Shader

@Shader("Phantom Pulse Effect", "Paopaodev")
class phantom_pulse:
    """
    #version 330 core
    in vec2 v_uv;
    out vec4 fragColor;
    uniform sampler2D u_tex;
    uniform float u_time;

    void main() {
        vec3 col = texture(u_tex, v_uv).rgb;

        float pulse = sin(u_time * 3.0 + v_uv.y * 10.0) * 0.3;
        col.r += pulse * 0.5;
        col.g += pulse * 0.7;
        col.b += pulse;

        float flicker = sin(u_time * 20.0 + (v_uv.x+v_uv.y) * 30.0) * 0.1;
        col += vec3(flicker);

        col = clamp(col, 0.0, 1.0);

        fragColor = vec4(col, 1.0);
    }
    """
