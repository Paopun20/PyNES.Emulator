from objects.shaderclass import Shader

@Shader("description", "your name, not a your real name")
class template:
    """
    #version 430
    in vec2 v_uv;
    out vec4 fragColor;

    uniform sampler2D u_tex;

    void main() {
        fragColor = texture(u_tex, v_uv);
    }
    """