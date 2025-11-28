from objects.shaderclass import Shader

@Shader("RGB Rainbow Time", "Paopaodev")
class rgb_ripple:
    """
    #version 330 core
    in vec2 v_uv;
    out vec4 fragColor;
    uniform sampler2D u_tex;
    uniform float u_time;
    void main() {
        vec3 col = texture(u_tex, v_uv).rgb;
        col.r += 0.5 * sin(u_time + v_uv.y * 10.0);
        col.g += 0.5 * sin(u_time + v_uv.x * 10.0 + 2.0);
        col.b += 0.5 * sin(u_time + (v_uv.x+v_uv.y) * 10.0 + 4.0);
        fragColor = vec4(col, 1.0);
    }
    """