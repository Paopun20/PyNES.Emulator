from shaders.shader_class import Shader

@Shader
class vhs:
    """
    precision mediump float;

    uniform sampler2D u_texture;
    uniform float u_time;      // animated time
    uniform vec2 u_resolution; // screen resolution
    varying vec2 v_texcoord;

    // Simple pseudo-random function
    float rand(vec2 co){
        return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }

    void main() {
        vec2 uv = v_texcoord;

        // Resolution-independent RGB shift
        float offset = 1.0 / u_resolution.x * 5.0; // ~5px shift
        vec3 color;
        color.r = texture2D(u_texture, uv + vec2(offset, 0.0)).r;
        color.g = texture2D(u_texture, uv).g;
        color.b = texture2D(u_texture, uv - vec2(offset, 0.0)).b;

        // Scanlines overlay
        float scanline = sin(uv.y * u_resolution.y * 3.0) * 0.05;
        color -= scanline;

        // Noise overlay
        float noise = (rand(uv * u_resolution.xy + u_time) - 0.5) * 0.1;
        color += noise;

        // Flicker effect
        float flicker = 0.05 * sin(u_time * 50.0 + uv.y * 10.0);
        color += flicker;

        // Random glitch lines
        if (rand(vec2(floor(u_time * 10.0), uv.y * 10.0)) > 0.95) {
            color.rgb += vec3(0.2, 0.2, 0.2); // bright glitch line
        }

        gl_FragColor = vec4(color, 1.0);
    }
    """
