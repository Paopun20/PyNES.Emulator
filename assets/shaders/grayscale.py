from objects.shaderclass import Shader

@Shader("Make a grayscale image out of a color image, yat /j uesless", "Paopaodev")
class grayscale:
    """
    #version 330 core
    in vec2 v_uv;
    out vec4 fragColor;
    uniform sampler2D u_tex;

    void main() {
        vec3 col = texture(u_tex, v_uv).rgb;
        float gray = dot(col, vec3(0.299, 0.587, 0.114));
        fragColor = vec4(vec3(gray), 1.0);
    }
    """