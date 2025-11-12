from objects.shader_class import Shader

@Shader
class VHS:
    """
    #version 330

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

    vec2 scandistort(vec2 uv) {
        float scan1 = clamp(cos(uv.y * 2.0 + u_time), 0.0, 1.0);
        float scan2 = clamp(cos(uv.y * 2.0 + u_time + 4.0) * 10.0, 0.0, 1.0);
        float amount = scan1 * scan2 * uv.x;
        return uv;
    }

    void main()
    {
        vec2 uv = v_uv;
        vec2 curUV = screenDistort(uv);
        uv = scandistort(curUV);
        vec4 video = getVideo(uv);
        float vigAmt = 1.0;
        float x = 0.0;

        // Chromatic aberration
        video.r = getVideo(vec2(x+uv.x+0.001,uv.y+0.001)).r+0.05;
        video.g = getVideo(vec2(x+uv.x+0.000,uv.y-0.002)).g+0.05;
        video.b = getVideo(vec2(x+uv.x-0.002,uv.y+0.000)).b+0.05;
        video.r += 0.08*getVideo(0.75*vec2(x+0.025, -0.027)+vec2(uv.x+0.001,uv.y+0.001)).r;
        video.g += 0.05*getVideo(0.75*vec2(x+-0.022, -0.02)+vec2(uv.x+0.000,uv.y-0.002)).g;
        video.b += 0.08*getVideo(0.75*vec2(x+-0.02, -0.018)+vec2(uv.x-0.002,uv.y+0.000)).b;

        video = clamp(video*0.6+0.4*video*video*1.0,0.0,1.0);
        vigAmt = 3.0+0.3*sin(u_time + 5.0*cos(u_time*5.0));

        float vignette = (1.0 - vigAmt*(uv.y-0.5)*(uv.y-0.5))*(1.0 - vigAmt*(uv.x-0.5)*(uv.x-0.5));
        video *= vignette;

        fragColor = mix(video, vec4(noise(uv * 75.0)), 0.05);

        if(curUV.x<0.0 || curUV.x>1.0 || curUV.y<0.0 || curUV.y>1.0){
            fragColor = vec4(0.0,0.0,0.0,0.0);
        }
    }
    """
