from objects.shader_class import Shader

@Shader("A VHS-like shader with various distortions and effects.")
class VHS:
    """
    #version 430
    uniform float u_time;
    uniform vec2 u_scale;
    uniform sampler2D u_tex;
    uniform float u_overlay_mode; // 0=none, 1=timestamp, 2=play, 3=rec, 4=pause
    in vec2 v_uv;
    out vec4 fragColor;
    
    float onOff(float a, float b, float c)
    {
        return step(c, sin(u_time + a*cos(u_time*b)));
    }
    
    float ramp(float y, float start, float end)
    {
        float inside = step(start,y) - step(end,y);
        float fact = (y-start)/(end-start)*inside;
        return (1.-fact) * inside;
    }
    
    vec4 getVideo(vec2 uv)
    {
        vec2 look = uv;
        float window = 1.0/(1.0+20.0*(look.y-mod(u_time/4.0,1.0))*(look.y-mod(u_time/4.0,1.0)));
        look.x += (sin(look.y*10.0 + u_time)/50.0*onOff(4.0,4.0,0.3)*(1.0+cos(u_time*80.0))*window)*(0.1*2.0);
        float vShift = 0.4*onOff(2.0,3.0,0.9)*(sin(u_time)*sin(u_time*20.0) +
                                             (0.5 + 0.1*sin(u_time*200.0)*cos(u_time)));
        look.y = mod(look.y + vShift*0.1, 1.0);
        return texture(u_tex, look);
    }
    
    vec2 screenDistort(vec2 uv)
    {
        uv = (uv - 0.5) * 2.0;
        uv *= 1.1;
        uv.x *= 1.0 + pow((abs(uv.y) / 5.0), 2.0);
        uv.y *= 1.0 + pow((abs(uv.x) / 4.0), 2.0);
        uv  = (uv / 2.0) + 0.5;
        uv = uv * 0.92 + 0.04;
        return uv;
    }
    
    float random(vec2 uv)
    {
        return fract(sin(dot(uv, vec2(15.5151, 42.2561))) * 12341.14122 * sin(u_time * 0.03));
    }
    
    float noise(vec2 uv)
    {
        vec2 i = floor(uv);
        vec2 f = fract(uv);
        float a = random(i);
        float b = random(i + vec2(1.0,0.0));
        float c = random(i + vec2(0.0, 1.0));
        float d = random(i + vec2(1.0));
        vec2 u = smoothstep(0.0, 1.0, f);
        return mix(a,b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    }
    
    // Improved scan line distortion
    vec2 scandistort(vec2 uv) {
        float scan1 = clamp(cos(uv.y * 2.0 + u_time), 0.0, 1.0);
        float scan2 = clamp(cos(uv.y * 2.0 + u_time + 4.0) * 10.0, 0.0, 1.0);
        float amount = scan1 * scan2 * uv.x;
        
        // Apply horizontal displacement for scan line effect
        uv.x -= 0.003 * mix(1.0, 0.0, amount);
        
        return uv;
    }
    
    // VHS tape color bleeding (reduces color saturation, shifts hues)
    vec3 vhsColorGrade(vec3 color)
    {
        // Reduce saturation for that washed-out VHS look
        float gray = dot(color, vec3(0.299, 0.587, 0.114));
        color = mix(vec3(gray), color, 0.7);
        
        // Slight color shift (VHS tapes tend toward magenta/cyan)
        color.r *= 1.05;
        color.b *= 0.95;
        
        // Reduce overall brightness slightly
        color *= 0.92;
        
        return color;
    }
    
    // Tape damage - occasional static snow
    float tapeNoise(vec2 uv, float intensity)
    {
        float noiseVal = random(uv * vec2(u_time * 5.0, u_time * 3.0));
        return noiseVal * intensity;
    }
    
    // Horizontal sync issues - occasional full-screen displacement
    float syncIssue()
    {
        float sync = sin(u_time * 0.5) * sin(u_time * 0.37);
        return step(0.95, sync) * 0.015; // Rare, strong horizontal shift
    }
    
    // Sharpen reduction (VHS was never sharp)
    vec4 soften(sampler2D tex, vec2 uv)
    {
        vec4 color = texture(tex, uv);
        vec4 blur = vec4(0.0);
        
        float kernel[9];
        kernel[0] = 0.05; kernel[1] = 0.09; kernel[2] = 0.05;
        kernel[3] = 0.09; kernel[4] = 0.26; kernel[5] = 0.09;
        kernel[6] = 0.05; kernel[7] = 0.09; kernel[8] = 0.05;
        
        float pixelSize = 0.0015;
        int index = 0;
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                vec2 offset = vec2(float(x), float(y)) * pixelSize;
                blur += getVideo(uv + offset) * kernel[index];
                index++;
            }
        }
        
        return blur;
    }
    
    // Edge artifacts - VHS had visible edge distortion
    float edgeArtifacts(vec2 uv)
    {
        float edge = 0.0;
        
        // Left/right edge noise
        if(uv.x < 0.05 || uv.x > 0.95) {
            edge = noise(vec2(uv.y * 20.0, u_time)) * 0.3;
        }
        
        return edge;
    }
    
    // Bottom tracking noise
    float bottomNoise(vec2 uv)
    {
        if(uv.y > 0.9) {
            float n = noise(vec2(uv.x * 50.0, u_time * 2.0));
            return n * (uv.y - 0.9) * 10.0;
        }
        return 0.0;
    }
    
    // Draw a digit (0-9) at position
    float drawDigit(vec2 uv, int digit, vec2 pos, vec2 size)
    {
        vec2 p = (uv - pos) / size;
        if(p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return 0.0;
        
        // 7-segment display style digits
        float seg = 0.0;
        float t = 0.15; // thickness
        
        // Segments: top, topR, botR, bot, botL, topL, middle
        bool segments[10][7];
        // 0
        segments[0][0] = true; segments[0][1] = true; segments[0][2] = true; segments[0][3] = true;
        segments[0][4] = true; segments[0][5] = true; segments[0][6] = false;
        // 1
        segments[1][0] = false; segments[1][1] = true; segments[1][2] = true; segments[1][3] = false;
        segments[1][4] = false; segments[1][5] = false; segments[1][6] = false;
        // 2
        segments[2][0] = true; segments[2][1] = true; segments[2][2] = false; segments[2][3] = true;
        segments[2][4] = true; segments[2][5] = false; segments[2][6] = true;
        // 3
        segments[3][0] = true; segments[3][1] = true; segments[3][2] = true; segments[3][3] = true;
        segments[3][4] = false; segments[3][5] = false; segments[3][6] = true;
        // 4
        segments[4][0] = false; segments[4][1] = true; segments[4][2] = true; segments[4][3] = false;
        segments[4][4] = false; segments[4][5] = true; segments[4][6] = true;
        // 5
        segments[5][0] = true; segments[5][1] = false; segments[5][2] = true; segments[5][3] = true;
        segments[5][4] = false; segments[5][5] = true; segments[5][6] = true;
        // 6
        segments[6][0] = true; segments[6][1] = false; segments[6][2] = true; segments[6][3] = true;
        segments[6][4] = true; segments[6][5] = true; segments[6][6] = true;
        // 7
        segments[7][0] = true; segments[7][1] = true; segments[7][2] = true; segments[7][3] = false;
        segments[7][4] = false; segments[7][5] = false; segments[7][6] = false;
        // 8
        segments[8][0] = true; segments[8][1] = true; segments[8][2] = true; segments[8][3] = true;
        segments[8][4] = true; segments[8][5] = true; segments[8][6] = true;
        // 9
        segments[9][0] = true; segments[9][1] = true; segments[9][2] = true; segments[9][3] = true;
        segments[9][4] = false; segments[9][5] = true; segments[9][6] = true;
        
        digit = clamp(digit, 0, 9);
        
        // Draw segments
        // Top
        if(segments[digit][0] && p.y < t && p.x > t && p.x < 1.0-t) seg = 1.0;
        // Top right
        if(segments[digit][1] && p.x > 1.0-t && p.y < 0.5-t/2.0 && p.y > t) seg = 1.0;
        // Bottom right
        if(segments[digit][2] && p.x > 1.0-t && p.y > 0.5+t/2.0 && p.y < 1.0-t) seg = 1.0;
        // Bottom
        if(segments[digit][3] && p.y > 1.0-t && p.x > t && p.x < 1.0-t) seg = 1.0;
        // Bottom left
        if(segments[digit][4] && p.x < t && p.y > 0.5+t/2.0 && p.y < 1.0-t) seg = 1.0;
        // Top left
        if(segments[digit][5] && p.x < t && p.y < 0.5-t/2.0 && p.y > t) seg = 1.0;
        // Middle
        if(segments[digit][6] && p.y > 0.5-t/2.0 && p.y < 0.5+t/2.0 && p.x > t && p.x < 1.0-t) seg = 1.0;
        
        return seg;
    }
    
    // Draw timestamp overlay
    vec3 drawTimestamp(vec2 uv, vec3 color)
    {
        vec2 digitSize = vec2(0.015, 0.025);
        vec2 startPos = vec2(0.05, 0.92);
        float spacing = 0.018;
        
        // Time based on u_time (simulated)
        int hours = int(mod(u_time * 0.1, 24.0));
        int minutes = int(mod(u_time * 0.5, 60.0));
        int seconds = int(mod(u_time, 60.0));
        
        float timestamp = 0.0;
        
        // Draw HH:MM:SS
        timestamp += drawDigit(uv, hours / 10, startPos, digitSize);
        timestamp += drawDigit(uv, int(mod(float(hours), 10.0)), startPos + vec2(spacing, 0.0), digitSize);
        
        // Colon
        if(uv.x > startPos.x + spacing*2.0 && uv.x < startPos.x + spacing*2.0 + 0.005) {
            if((uv.y > startPos.y + 0.008 && uv.y < startPos.y + 0.012) || 
               (uv.y > startPos.y + 0.018 && uv.y < startPos.y + 0.022)) {
                timestamp = 1.0;
            }
        }
        
        timestamp += drawDigit(uv, minutes / 10, startPos + vec2(spacing*2.5, 0.0), digitSize);
        timestamp += drawDigit(uv, int(mod(float(minutes), 10.0)), startPos + vec2(spacing*3.5, 0.0), digitSize);
        
        // Colon
        if(uv.x > startPos.x + spacing*4.5 && uv.x < startPos.x + spacing*4.5 + 0.005) {
            if((uv.y > startPos.y + 0.008 && uv.y < startPos.y + 0.012) || 
               (uv.y > startPos.y + 0.018 && uv.y < startPos.y + 0.022)) {
                timestamp = 1.0;
            }
        }
        
        timestamp += drawDigit(uv, seconds / 10, startPos + vec2(spacing*5.0, 0.0), digitSize);
        timestamp += drawDigit(uv, int(mod(float(seconds), 10.0)), startPos + vec2(spacing*6.0, 0.0), digitSize);
        
        return mix(color, vec3(1.0, 1.0, 0.9), timestamp * 0.8);
    }
    
    // Draw PLAY symbol
    vec3 drawPlay(vec2 uv, vec3 color)
    {
        vec2 center = vec2(0.05, 0.05);
        vec2 p = uv - center;
        
        float symbol = 0.0;
        
        // Triangle pointing right
        if(abs(p.y) < 0.015 && p.x > 0.0 && p.x < 0.025) {
            if(abs(p.y) < (0.015 - p.x * 0.6)) {
                symbol = 1.0;
            }
        }
        
        return mix(color, vec3(1.0, 0.2, 0.2), symbol * 0.7);
    }
    
    // Draw REC symbol
    vec3 drawRec(vec2 uv, vec3 color)
    {
        vec2 dotPos = vec2(0.05, 0.05);
        float dotRadius = 0.01;
        
        float symbol = 0.0;
        
        // Red dot
        if(distance(uv, dotPos) < dotRadius) {
            symbol = 1.0;
        }
        
        // Blinking effect
        float blink = step(0.5, mod(u_time * 2.0, 1.0));
        
        // REC text
        vec2 textPos = vec2(0.07, 0.045);
        if(uv.y > textPos.y && uv.y < textPos.y + 0.015 && 
           uv.x > textPos.x && uv.x < textPos.x + 0.04) {
            symbol = 1.0;
        }
        
        return mix(color, vec3(1.0, 0.0, 0.0), symbol * blink * 0.8);
    }
    
    // Draw PAUSE symbol
    vec3 drawPause(vec2 uv, vec3 color)
    {
        vec2 center = vec2(0.05, 0.05);
        vec2 p = uv - center;
        
        float symbol = 0.0;
        
        // Two vertical bars
        if(abs(p.y) < 0.015) {
            if((p.x > -0.008 && p.x < -0.003) || (p.x > 0.003 && p.x < 0.008)) {
                symbol = 1.0;
            }
        }
        
        return mix(color, vec3(1.0, 1.0, 0.0), symbol * 0.7);
    }
    
    void main()
    {
        vec2 uv = v_uv;
        vec2 curUV = screenDistort(uv);
        
        // Apply sync issues
        curUV.x += syncIssue();
        
        uv = scandistort(curUV);
        
        // Use softened video
        vec4 video = soften(u_tex, uv);
        
        float vigAmt = 1.0;
        float x = 0.0;
        
        // Enhanced chromatic aberration with more offset
        video.r = getVideo(vec2(x+uv.x+0.002,uv.y+0.001)).r+0.05;
        video.g = getVideo(vec2(x+uv.x+0.000,uv.y-0.002)).g+0.05;
        video.b = getVideo(vec2(x+uv.x-0.003,uv.y+0.000)).b+0.05;
        video.r += 0.08*getVideo(0.75*vec2(x+0.025, -0.027)+vec2(uv.x+0.001,uv.y+0.001)).r;
        video.g += 0.05*getVideo(0.75*vec2(x+-0.022, -0.02)+vec2(uv.x+0.000,uv.y-0.002)).g;
        video.b += 0.08*getVideo(0.75*vec2(x+-0.02, -0.018)+vec2(uv.x-0.002,uv.y+0.000)).b;
        
        // Apply VHS color grading
        video.rgb = vhsColorGrade(video.rgb);
        
        video = clamp(video*0.6+0.4*video*video*1.0,0.0,1.0);
        
        // Dynamic vignette
        vigAmt = 3.0+0.3*sin(u_time + 5.0*cos(u_time*5.0));
        float vignette = (1.0 - vigAmt*(uv.y-0.5)*(uv.y-0.5))*(1.0 - vigAmt*(uv.x-0.5)*(uv.x-0.5));
        video *= vignette;
        
        // Add scan lines (visible horizontal lines)
        float scanline = sin(uv.y * 800.0) * 0.04;
        video.rgb -= scanline;
        
        // Mix in film grain
        float grainAmount = 0.08;
        video = mix(video, vec4(noise(uv * 75.0)), grainAmount);
        
        // Add tape static in dark areas
        float darkMultiplier = 1.0 - clamp(dot(video.rgb, vec3(0.333)), 0.0, 1.0);
        float tapeStatic = tapeNoise(uv, 0.15) * darkMultiplier;
        video.rgb += tapeStatic;
        
        // Add edge artifacts
        video.rgb += edgeArtifacts(uv);
        
        // Add bottom tracking noise
        video.rgb += bottomNoise(uv) * 0.5;
        
        // Occasional white noise flash (tape damage)
        float flash = step(0.998, random(vec2(u_time * 0.1)));
        video.rgb = mix(video.rgb, vec3(random(uv)), flash * 0.3);
        
        // Apply overlays based on mode
        int mode = int(u_overlay_mode);
        if(mode == 1) {
            video.rgb = drawTimestamp(uv, video.rgb);
        } else if(mode == 2) {
            video.rgb = drawPlay(uv, video.rgb);
        } else if(mode == 3) {
            video.rgb = drawRec(uv, video.rgb);
        } else if(mode == 4) {
            video.rgb = drawPause(uv, video.rgb);
        }
        
        fragColor = video;
        
        // Black bars outside screen distortion
        if(curUV.x<0.0 || curUV.x>1.0 || curUV.y<0.0 || curUV.y>1.0){
            fragColor = vec4(0.0,0.0,0.0,1.0);
        }
    }
    """
