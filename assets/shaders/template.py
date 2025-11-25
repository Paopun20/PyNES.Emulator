from objects.shadercass import Shader

@Shader("description", "your name, not a your real name")
class template:
    """
    #version 430
    in vec2 v_uv;
    out vec4 fragColor;

    void main() {
        // Example: simple color based on UV
        fragColor = vec4(v_uv, 0.0, 1.0);
    }
    """