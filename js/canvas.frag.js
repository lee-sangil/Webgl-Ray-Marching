const fragment = /* glsl */ `
// Author: Sangil Lee
// Refer to Tim Coster, https://timcoster.com

precision highp float;
uniform vec2 u_resolution; // Width & height of the shader
uniform float u_time; // Time elapsed
uniform sampler2D u_texture;
uniform vec2 u_angle; // camera rotation

// Constants
#define PI 3.1415925359
#define TWO_PI 6.2831852
#define MAX_STEPS 100 // Mar Raymarching steps
#define MAX_DIST 100. // Max Raymarching distance
#define SURF_DIST .01 // Surface distance
#define SAMPLE_DIST 0.05 // Sample distance

float random12(in vec2 x) {
    return fract(sin(dot(x, vec2(12.9898,54.233))) * 43758.5453123);
}

float random13(in vec3 x) {
    return fract(sin(dot(x, vec3(12.9898, 78.233, 45.678))) * 43758.5453123);
}

vec3 random33(in vec3 x, in float seed) {
    return vec3(
        random13(x + vec3(seed+1.0, 0.0, 0.0)), 
        random13(x + vec3(0.0, seed+1.0, 0.0)), 
        random13(x + vec3(0.0, 0.0, seed+1.0))
        );
}

float noise(in vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);

    float tl = random12(i); // top-left corner
    float tr = random12(i + vec2(1.0, 0.0)); // top-right corner
    float bl = random12(i + vec2(0.0, 1.0)); // bottom-left corner
    float br = random12(i + vec2(1.0, 1.0)); // bottom-right corner

    vec2 u = smoothstep(0., 1., f);

    return
    mix(
        mix(tl, tr, u.x), 
        mix(bl, br, u.x), 
    u.y);
}

float get_plane_dist(in vec3 p) {
    return p.y + 1.2;// + 0.5*noise(0.5*p.xz);
}

float get_sphere_dist(in vec3 p) {
    vec4 s = vec4(0, 0, 0, 1);
    return length(p - s.xyz) - s.w;
}

float get_optical_density(in vec3 p) {
    if (get_sphere_dist(p) < 0.0) return 0.1; // sphere density
    if (get_plane_dist(p) < 0.0) return 10.0; // ground density
    return 0.01; // atmosphere density
}

float get_refractive_index(in vec3 p) {
    if (get_sphere_dist(p) < 0.0) return 1.5; // sphere refractive index
    if (get_plane_dist(p) < 0.0) return 2.0; // ground refractive index
    return 1.0; // atmosphere refractive index
}

float get_reflectance(in vec3 p) {
    if (get_sphere_dist(p) < 0.0) return 0.1; // sphere reflectance
    if (get_plane_dist(p) < 0.0) return 0.0; // ground reflectance
    return 0.0;
}

float get_specular(in vec3 p) {
    if (get_sphere_dist(p) < 0.0) return 1.0; // sphere specular
    if (get_plane_dist(p) < 0.0) return 0.0; // ground specular
    return 0.;
}

vec2 get_sphere_tex_coord(in vec3 p) {
    vec4 s = vec4(0, 0, 0, 1);
    vec3 r = p - s.xyz;
    float phi = atan(sqrt(dot(r.xz, r.xz)), r.y);
    float theta = atan(r.z, r.x);
    return fract(vec2(phi / PI, theta / TWO_PI));
}

vec3 get_color(in vec3 p) {
    if (get_sphere_dist(p) < 0.0) return vec3(0.8, 0.9, 1.0); // sphere color
    // if (get_plane_dist(p) < 0.0) return vec3(0.8, 0.55, 0.3); // ground color
    // if (get_sphere_dist(p) < 0.0) return texture2D(u_texture, get_sphere_tex_coord(p)).rgb; // sphere color
    if (get_plane_dist(p) < 0.0) return texture2D(u_texture, fract(0.1*p.xz)).rgb; // ground color
    return vec3(1.2, 1.3, 1.5); // atmosphere color
}

vec3 get_light() {
    return vec3(6.*cos(u_time),5.,6.*sin(u_time));
}

float SDF(in vec3 p) {
    float sphereDist = get_sphere_dist(p);
    float planeDist = get_plane_dist(p);
    float d = min(sphereDist, planeDist);
    return d;
}

vec3 get_normal(in vec3 p) { 
    vec2 e = vec2(.01,0); // Epsilon
    vec3 n = vec3(
        SDF(p+e.xyy) - SDF(p-e.xyy),
        SDF(p+e.yxy) - SDF(p-e.yxy),
        SDF(p+e.yyx) - SDF(p-e.yyx)
    );
    return normalize(n);
}

// the Henyey-Greenstein phase function
float compute_phase(float g, float cos_theta)
{
    float denom = 1. + g * g - 2. * g * cos_theta;
    return 1. / (4. * PI) * (1. - g * g) / (denom * sqrt(denom));
}

// Fresnel reflectance
// Schlick's approximation
float compute_fresnel_reflectance(float eta, float u, float f0, float f90) {
    if (f0 == 0.0) return 0.0;

    float r0 = (eta - 1.) / (eta + 1.);
    r0 = r0 * r0;

    if (eta > 1.0) {
        float sin_refract = eta * (1. - u*u);
        if (sin_refract >= 1.0) return 1.0; // total internal reflection
        u = sqrt(1.0 - sin_refract*sin_refract);
    }

    float r = mix(r0, 1., pow(1. - u, 5.0));
    return mix(f0, f90, r);
}

float compute_intensity(in vec3 p) {
    vec3 l = get_light(); // light position
    vec3 ld = normalize(l-p); // light direction

    vec3 q = p;
    float intensity = 10.;
    for (int i = 0; i < MAX_STEPS; ++i) {
        float ds = SDF(q);
        float density = get_optical_density(q);

        float step_size;
        if (ds > 0.0)
            step_size = ds;
        else
            step_size = SAMPLE_DIST;

        q += step_size * ld;
        intensity *= exp(-density * step_size);

        if (dot(l - p, l - q) < 0.0 || intensity < 0.01)
            break;
    }
    return intensity;
}

vec3 compute_color_along_ray(in vec3 p, in vec3 v, in vec3 l_color, in float l_intensity) {
    vec3 c = vec3(0.);
    vec3 l = get_light();
    float dist = 0.;
    float transmittance = l_intensity;
    for (int j = 0; j < MAX_STEPS; ++j) {
        vec3 p = p + v * dist;
        float ds = SDF(p);
        
        vec3 color = get_color(p);
        float density = get_optical_density(p);
        float intensity = compute_intensity(p);
        
        float step_size;
        if (ds > SAMPLE_DIST)
            step_size = ds;
        else
            step_size = SAMPLE_DIST * (random13(p) + 0.5);

        dist += step_size;
        float phase = compute_phase(0.1, dot(v, normalize(p - l)));
        c += l_color * color * (1. - exp(-density * step_size)) * intensity * transmittance * phase;
        transmittance *= exp(-density * step_size);
        
        if (transmittance < 0.01 || dist > MAX_DIST) break;
    }
    return c;
}

vec3 compute_color(in vec3 ro, in vec3 rd) {
    vec3 c = vec3(0.);
    float transmittance = 1.;
    float dist = 0.;
    float total_dist = 0.;
    vec3 l = get_light(); // light position
    vec3 p = ro;
    vec3 v = rd;
    float ri = get_refractive_index(p);
    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 q = p + v * dist;
        float ds = SDF(q);

        vec3 color = get_color(q);
        float density = get_optical_density(q);
        float intensity = compute_intensity(q);
        if (ri != get_refractive_index(q)) {
            float ri_new = get_refractive_index(q);
            vec3 n = get_normal(q);
            n *= -sign(dot(v, n)); // make sure dot(v, n) < 0

            float reflectance = compute_fresnel_reflectance(ri / ri_new, -dot(v, n), get_reflectance(q), 1.0);
            vec3 color_accum = vec3(0.);
            const int N = 1; // number of samples
            for (int t = 0; t < N; ++t) {
                // reflection = diffuse + specular + fresnel
                // specular reflection + fresnel
                // vec3 rand_vec = normalize(random33(q, float(t))); // random unit vector
                vec3 v_reflect = reflect(v, n);
                v_reflect = mix(n, v_reflect, get_specular(q)); // random unit vector on the hemisphere
                color_accum += compute_color_along_ray(q, v_reflect, get_color(q), reflectance * transmittance * get_specular(q));

                // // diffuse reflection
                // // random unit vector on the hemisphere following Lambertian distribution
                // rand_vec = normalize(random33(q + vec3(1.), float(t))); // random unit vector
                // vec3 v_diffuse = normalize(n + rand_vec); // random unit vector on the hemisphere
                // v_diffuse *= dot(v_diffuse, n); // cosine weighted
                // float diffuse_intensity = 0.5*max(0., dot(normalize(l - q), n));
                // color_accum += compute_color_along_ray(q, v_diffuse, vec3(1.0), (1. - get_specular(q)) * transmittance * diffuse_intensity);

                // // diffuse refraction - todo
                // rand_vec = normalize(random33(q + vec3(2.), float(t))); // random unit vector
                // vec3 v_refract = refract(v, normalize(n + (1. - get_specular(q)) * rand_vec), ri / ri_new);
                // color_accum += compute_color_along_ray(q, v_refract, vec3(1.0), (1. - get_specular(q)) * transmittance);
            }
            color_accum /= float(N); // average color
            c += color_accum;

            // refraction
            v = refract(v, n, ri / ri_new);
            p = q;

            total_dist += dist;
            dist = 0.;
            ri = ri_new;
            transmittance *= 1. - reflectance;
        }
        
        float step_size;
        if (ds > SAMPLE_DIST)
            step_size = ds;
        else
            step_size = SAMPLE_DIST * (random13(q) + 0.5);

        dist += step_size;
        float phase = compute_phase(0.1, dot(v, normalize(q - l)));
        c += color * (1. - exp(-density * step_size)) * intensity * transmittance * phase;
        transmittance *= exp(-density * step_size);
        
        if (transmittance < 0.01 || dist + total_dist > MAX_DIST) break;
    }
    return c;
}

vec2 get_normalized_coordinates(in vec2 uv) {
    // intrinsic parameter
    vec2 f = vec2(600.);
    vec2 c = 0.5*u_resolution.xy;
    
    // normalized image plane
    vec2 xy = (uv - c) / f;
    
    // distortion
    vec2 k_d = vec2(0.3,0.0);
    vec2 p_d = vec2(0.0,0.0);
    float r2 = dot(xy, xy);
    return (1. + k_d.x*r2 + k_d.y*r2*r2) * xy + vec2(
        2.*p_d.x*xy.x*xy.y + p_d.y*(r2 + 2.*xy.x*xy.x), 
        2.*p_d.y*xy.x*xy.y + p_d.x*(r2 + 2.*xy.y*xy.y)
    );
}

mat4 compute_transformation(in float theta, in float phi) {
    mat3 Ry = mat3( // column-major
        cos(theta), 0., -sin(theta), // Ry[0]
        0., 1., 0., // Ry[1]
        sin(theta), 0., cos(theta) // Ry[2]
    );

    mat3 Rx = mat3( // column-major
        1., 0., 0., // Rx[0]
        0., cos(phi), sin(phi), // Rx[1]
        0., -sin(phi), cos(phi) // Rx[2]
    );

    mat3 R = Ry * Rx; // camera rotation
    float r = 6.0; // camera distance
    vec3 t = vec3(-r*cos(phi)*sin(theta), r*sin(phi), -r*cos(phi)*cos(theta)); // camera translation

    return mat4( // column-major
        vec4(R[0], 0.0),
        vec4(R[1], 0.0),
        vec4(R[2], 0.0),
        vec4(t, 1.0)
    );
}

void main()
{
    // image plane: u ~ [0, u_resolution.x], v ~ [0, u_resolution.y]
    vec2 uv = gl_FragCoord.xy;

    // normalized image plane
    vec2 xy = get_normalized_coordinates(uv);
    
    // compute transformation matrix
    mat4 T = compute_transformation(u_angle.x, u_angle.y); // yaw, pitch

    // camera
    vec3 co = vec3(0,0,0); // camera origin
    vec3 cd = normalize(vec3(xy,1)); // camera ray vector
    
    // ray
    vec3 ro = (T * vec4(co, 1.)).xyz; // ray origin w.r.t world
    vec3 rd = normalize((T * vec4(cd, 0.)).xyz); // ray vector w.r.t world

    // compute color
    vec3 color = compute_color(ro, rd); // apply color and texture

    gl_FragColor = vec4(color,1.0);
}
`
export default fragment