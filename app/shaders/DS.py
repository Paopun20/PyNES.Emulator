# Nintendo DS

shader = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform float u_time; // Time uniform for animation

void main() {
    vec2 uv = v_uv;

    // Apply a simple scanline effect
    float scanline = sin(uv.y * 240.0 * 3.14159 * 2.0 + u_time * 5.0) * 0.05 + 0.95;
    
    // Apply a slight barrel distortion for a CRT-like feel
    vec2 center = vec2(0.5, 0.5);
    vec2 tex_coords = uv - center;
    float r2 = dot(tex_coords, tex_coords);
    float distortion_factor = 1.0 + r2 * 0.2; // Adjust 0.2 for more/less distortion
    tex_coords *= distortion_factor;
    tex_coords += center;

    // Ensure texture coordinates are within bounds after distortion
    if (tex_coords.x < 0.0 || tex_coords.x > 1.0 || tex_coords.y < 0.0 || tex_coords.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0); // Render black outside distorted area
        return;
    }

    vec3 col = texture(u_tex, tex_coords).rgb;

    // Apply color tint (e.g., a slight sepia or warm tone)
    vec3 tint = vec3(1.0, 0.95, 0.9); // Warm tint
    col *= tint;

    // Apply scanline effect
    col *= scanline;

    // Simulate a subtle glow/bloom
    vec3 bloom_color = vec3(0.1, 0.1, 0.05); // Warm glow
    col += bloom_color * pow(col.r + col.g + col.b, 2.0) * 0.1; // Adjust 0.1 for intensity

    fragColor = vec4(col, 1.0);
}
"""