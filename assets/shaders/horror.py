from objects.shadercass import Shader

@Shader("undescription", "Paopaodev")
class horror:
    """
    #version 430
    uniform float u_time;
    uniform vec2 u_scale;
    uniform sampler2D u_tex;
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
        float vShift = 0.4*onOff(2.0,3.0,0.9)*(sin(u_time)*sin(u_time*20.0) + (0.5 + 0.1*sin(u_time*200.0)*cos(u_time)));
        look.y = mod(look.y + vShift*0.1, 1.0);
        return texture(u_tex, look);
    }

    vec2 screenDistort(vec2 uv)
    {
        uv = (uv - 0.5) * 2.0;
        uv *= 1.1;
        uv.x *= 1.0 + pow((abs(uv.y) / 5.0), 2.0);
        uv.y *= 1.0 + pow((abs(uv.x) / 4.0), 2.0);
        uv = (uv / 2.0) + 0.5;
        uv = uv * 0.92 + 0.04;
        return uv;
    }

    float random(vec2 uv)
    {
        return fract(sin(dot(uv, vec2(15.5151, 42.2561))) * 12341.14122 * sin(u_time * 0.01));
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

    vec2 scandistort(vec2 uv) {
        float scan1 = clamp(cos(uv.y * 2.0 + u_time), 0.0, 1.0);
        float scan2 = clamp(cos(uv.y * 2.0 + u_time + 4.0) * 10.0, 0.0, 1.0);
        float amount = scan1 * scan2 * uv.x;
        uv.x -= 0.003 * mix(1.0, 0.0, amount);
        return uv;
    }

    vec3 vhsColorGrade(vec3 color)
    {
        float gray = dot(color, vec3(0.299, 0.587, 0.114));
        color = mix(vec3(gray), color, 0.7);
        color.r *= 1.05;
        color.b *= 0.95;
        color *= 0.92;
        return color;
    }

    float tapeNoise(vec2 uv, float intensity)
    {
        float noiseVal = random(uv * vec2(u_time * 5.0, u_time * 3.0));
        return noiseVal * intensity;
    }

    float syncIssue()
    {
        float sync = sin(u_time * 0.5) * sin(u_time * 0.37);
        return step(0.95, sync) * 0.015;
    }

    vec4 soften(sampler2D tex, vec2 uv)
    {
        vec4 blur = vec4(0.0);
        float kernel[9] = float[9](0.05,0.09,0.05, 0.09,0.26,0.09, 0.05,0.09,0.05);
        float pixelSize = 0.0015;
        int index = 0;
        for(int y=-1; y<=1; y++){
            for(int x=-1; x<=1; x++){
                vec2 offset = vec2(float(x), float(y)) * pixelSize;
                blur += getVideo(uv + offset) * kernel[index];
                index++;
            }
        }
        return blur;
    }

    float edgeArtifacts(vec2 uv)
    {
        float edge = 0.0;
        if(uv.x < 0.05 || uv.x > 0.95){
            edge = noise(vec2(uv.y * 20.0, u_time)) * 0.3;
        }
        return edge;
    }

    float bottomNoise(vec2 uv)
    {
        if(uv.y > 0.9){
            float n = noise(vec2(uv.x * 50.0, u_time * 2.0));
            return n * (uv.y - 0.9) * 10.0;
        }
        return 0.0;
    }

    bool getSegment(int digit, int seg)
    {
        // 0-9 segments: top, topR, botR, bot, botL, topL, middle
        bool s[10][7];
        s[0] = bool[7](true,true,true,true,true,true,false);
        s[1] = bool[7](false,true,true,false,false,false,false);
        s[2] = bool[7](true,true,false,true,true,false,true);
        s[3] = bool[7](true,true,true,true,false,false,true);
        s[4] = bool[7](false,true,true,false,false,true,true);
        s[5] = bool[7](true,false,true,true,false,true,true);
        s[6] = bool[7](true,false,true,true,true,true,true);
        s[7] = bool[7](true,true,true,false,false,false,false);
        s[8] = bool[7](true,true,true,true,true,true,true);
        s[9] = bool[7](true,true,true,true,false,true,true);
        return s[clamp(digit,0,9)][seg];
    }

    float drawDigit(vec2 uv, int digit, vec2 pos, vec2 size)
    {
        vec2 p = (uv - pos) / size;
        if(p.x<0.0||p.x>1.0||p.y<0.0||p.y>1.0) return 0.0;
        float seg = 0.0;
        float t = 0.15;
        // Top
        if(getSegment(digit,0) && p.y < t && p.x > t && p.x < 1.0-t) seg = 1.0;
        // Top right
        if(getSegment(digit,1) && p.x > 1.0-t && p.y < 0.5-t/2.0 && p.y > t) seg = 1.0;
        // Bottom right
        if(getSegment(digit,2) && p.x > 1.0-t && p.y > 0.5+t/2.0 && p.y < 1.0-t) seg = 1.0;
        // Bottom
        if(getSegment(digit,3) && p.y > 1.0-t && p.x > t && p.x < 1.0-t) seg = 1.0;
        // Bottom left
        if(getSegment(digit,4) && p.x < t && p.y > 0.5+t/2.0 && p.y < 1.0-t) seg = 1.0;
        // Top left
        if(getSegment(digit,5) && p.x < t && p.y < 0.5-t/2.0 && p.y > t) seg = 1.0;
        // Middle
        if(getSegment(digit,6) && p.y > 0.5-t/2.0 && p.y < 0.5+t/2.0 && p.x > t && p.x < 1.0-t) seg = 1.0;
        return seg;
    }

    vec3 drawTimestamp(vec2 uv, vec3 color)
    {
        vec2 digitSize = vec2(0.015, 0.025);
        vec2 startPos = vec2(0.05, 0.92);
        float spacing = 0.018;

        float timeS = mod(u_time, 60.0);          // seconds 0–59
        float timeM = mod(floor(u_time / 60.0), 60.0);   // minutes 0–59
        float timeH = mod(floor(u_time / 3600.0), 24.0); // hours 0–23

        int hours = int(floor(timeH));
        int minutes = int(floor(timeM));
        int seconds = int(floor(timeS));

        int hT = hours / 10; 
        int hO = hours % 10;

        int mT = minutes / 10; 
        int mO = minutes % 10;

        int sT = seconds / 10; 
        int sO = seconds % 10;

        // Use max instead of sum to avoid overbright
        float overlayMask = 0.0;

        overlayMask = max(overlayMask, drawDigit(uv, hT, startPos, digitSize));
        overlayMask = max(overlayMask, drawDigit(uv, hO, startPos + vec2(spacing, 0.0), digitSize));

        // Colon 1
        if(uv.x > startPos.x + spacing*2.0 && uv.x < startPos.x + spacing*2.0 + 0.005){
            if((uv.y > startPos.y + 0.008 && uv.y < startPos.y + 0.012) ||
               (uv.y > startPos.y + 0.018 && uv.y < startPos.y + 0.022)) 
            {
                overlayMask = max(overlayMask, 1.0);
            }
        }

        overlayMask = max(overlayMask, drawDigit(uv, mT, startPos + vec2(spacing*2.5, 0.0), digitSize));
        overlayMask = max(overlayMask, drawDigit(uv, mO, startPos + vec2(spacing*3.5, 0.0), digitSize));

        // Colon 2
        if(uv.x > startPos.x + spacing*4.5 && uv.x < startPos.x + spacing*4.5 + 0.005){
            if((uv.y > startPos.y + 0.008 && uv.y < startPos.y + 0.012) ||
               (uv.y > startPos.y + 0.018 && uv.y < startPos.y + 0.022))
            {
                overlayMask = max(overlayMask, 1.0);
            }
        }

        overlayMask = max(overlayMask, drawDigit(uv, sT, startPos + vec2(spacing*5.0, 0.0), digitSize));
        overlayMask = max(overlayMask, drawDigit(uv, sO, startPos + vec2(spacing*6.0, 0.0), digitSize));

        overlayMask = clamp(overlayMask, 0.0, 1.0);
        return mix(color, vec3(1.0, 1.0, 0.9), overlayMask * 0.8);
    }
    
    vec3 drawPlay(vec2 uv, vec3 color)
    {
        vec2 center = vec2(0.05, 0.05);
        vec2 p = uv - center;
        float symbol = 0.0;

        // Right-pointing triangle
        if(p.x > 0.0 && p.x < 0.025 && abs(p.y) < (0.015 - p.x * 0.6))
        {
            symbol = 1.0;
        }

        return mix(color, vec3(1.0, 0.2, 0.2), symbol * 0.7);
    }

    vec3 drawRec(vec2 uv, vec3 color)
    {
        vec2 dotPos = vec2(0.05, 0.05);
        float dotRadius = 0.01;
        float symbol = 0.0;

        // Red dot
        if(distance(uv, dotPos) < dotRadius)
        {
            symbol = 1.0;
        }

        // Blinking effect
        float blink = step(0.5, mod(u_time * 2.0, 1.0));

        // REC text box (simplified rectangular overlay)
        vec2 textPos = vec2(0.07, 0.045);
        if(uv.y > textPos.y && uv.y < textPos.y + 0.015 && uv.x > textPos.x && uv.x < textPos.x + 0.04)
        {
            symbol = 1.0;
        }

        return mix(color, vec3(1.0, 0.0, 0.0), symbol * blink * 0.8);
    }

    vec3 drawPause(vec2 uv, vec3 color)
    {
        vec2 center = vec2(0.05, 0.05);
        vec2 p = uv - center;
        float symbol = 0.0;

        // Two vertical bars
        if(abs(p.y) < 0.015)
        {
            if((p.x > -0.008 && p.x < -0.003) || (p.x > 0.003 && p.x < 0.008))
            {
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
        vec4 video = soften(u_tex, uv);

        float vigAmt = 3.0+0.3*sin(u_time+5.0*cos(u_time*5.0));
        float vignette = (1.0 - vigAmt*(uv.y-0.5)*(uv.y-0.5))*(1.0 - vigAmt*(uv.x-0.5)*(uv.x-0.5));
        video *= vignette;

        float scanline = sin(uv.y * 800.0) * 0.04;
        video.rgb -= scanline;
        video = mix(video, vec4(noise(uv*75.0)), 0.08);

        float darkMultiplier = 1.0 - clamp(dot(video.rgb, vec3(0.333)),0.0,1.0);
        video.rgb += tapeNoise(uv,0.15) * darkMultiplier;
        video.rgb += edgeArtifacts(uv);
        video.rgb += bottomNoise(uv) * 0.5;

        float flash = step(0.998, random(vec2(u_time*0.1)));
        video.rgb = mix(video.rgb, vec3(random(uv)), flash*0.3);

        video.rgb = drawTimestamp(uv, video.rgb);
        video.rgb = drawPlay(uv, video.rgb);
        // video.rgb = drawRec(uv, video.rgb);
        // video.rgb = drawPause(uv, video.rgb);

        fragColor = video;

        if(curUV.x<0.0 || curUV.x>1.0 || curUV.y<0.0 || curUV.y>1.0){
            fragColor = vec4(0.0,0.0,0.0,1.0);
        }
    }
    """
