shader = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform float u_time;

void main() {
    vec2 uv = v_uv;
    
    // Scanlines
    float scanline = sin(uv.y * 240.0 * 3.14159) * 0.04;
    
    // Vignette
    vec2 center = uv - 0.5;
    float vignette = 1.0 - dot(center, center) * 0.5;
    
    // Slight curvature
    vec2 curve = uv * 2.0 - 1.0;
    curve *= 1.0 + 0.05 * dot(curve, curve);
    curve = (curve + 1.0) * 0.5;
    
    // Check if out of bounds
    if (curve.x < 0.0 || curve.x > 1.0 || curve.y < 0.0 || curve.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    // Sample texture normally
    vec3 col = texture(u_tex, curve).rgb;
    col *= (1.0 - scanline) * vignette;
    
    fragColor = vec4(col, 1.0);
}
"""