const fragment = /* glsl */ `#version 300 es
// Author: Sangil Lee
// Refer to Tim Coster, https://timcoster.com

precision highp float;
uniform vec2 u_resolution; // Width & height of the shader
uniform float u_time; // Time elapsed
uniform sampler2D u_texture;
uniform vec2 u_angle; // camera rotation
out vec4 fragColor;

// Constants
#define PI 3.1415925359
#define TWO_PI 6.2831852
#define MAX_STEPS 200 // Mar Raymarching steps
#define MAX_DIST 100.0 // Max Raymarching distance
#define MIN_TRNASMITTANCE 0.01 // Minimum transmittance
#define SAMPLE_DIST 0.04 // Sample distance
#define SURF_DIST (2.0 * SAMPLE_DIST) // Surface distance
#define NUM_RAYS 20 // Number of rays to trace

struct Ray {
    vec3 origin; // ray origin
    vec3 direction; // ray direction
    vec3 color; // ray color
    float transmittance; // transmittance
    float dist; // distance traveled before
};
Ray rays[NUM_RAYS];
int n_rays = 0; // number of rays

float seed = 0.0; // random seed
float hash13(vec3 p)
{
    seed += 1.0; // increment seed for each call
    p = p + vec3(seed, seed, seed); // perturb the input
    p  = fract(p * .1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
vec3 hash33(vec3 p)
{
    seed += 1.0; // increment seed for each call
    p = p + vec3(seed, seed, seed); // perturb the input
    p = fract(p * vec3(.1031, .1030, .0973));
    p += dot(p, p.yxz+33.33);
    return fract((p.xxy + p.yxx)*p.zyx);
}

float get_plane_dist(in vec3 p) {
    return p.y + 1.2;
}

float get_sphere_dist(in vec3 p) {
    vec4 s = vec4(0, 0, 0, 1);
    return length(p - s.xyz) - s.w;
}

int get_object_id(in vec3 p) {
    if (get_sphere_dist(p) < 0.0) return 1; // sphere
    if (get_plane_dist(p) < 0.0) return 2; // ground
    return 0; // atmosphere
}

float optical_density[3] = float[](0.01, 0.1, 10.0); // atmosphere, sphere, ground
float refractive_index[3] = float[](1.0, 1.5, 2.0); // atmosphere, sphere, ground
float reflectance[3] = float[](0.0, 0.1, 0.0); // atmosphere, sphere, ground
float specular[3] = float[](0.0, 0.8, 0.0); // atmosphere, sphere, ground

float get_optical_density(in vec3 p) {
    return optical_density[get_object_id(p)];
}

float get_refractive_index(in vec3 p) {
    return refractive_index[get_object_id(p)];
}

float get_reflectance(in vec3 p) {
    return reflectance[get_object_id(p)];
}

float get_specular(in vec3 p) {
    return specular[get_object_id(p)];
}

float get_roughness(in vec3 p, in vec3 q) {
    int id_p = get_object_id(p);
    int id_q = get_object_id(q);

    if (id_p == 0 && id_q == 1 || id_p == 1 && id_q == 0) return 0.03; // between atmosphere and sphere

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
    if (get_plane_dist(p) < 0.0) return texture(u_texture, fract(0.1*p.xz)).rgb; // ground color
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
    vec3 ld = normalize(l - p); // light direction

    vec3 q = p;
    float intensity = 15.;
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

vec3 compute_color_along_ray(in vec3 p, in vec3 v, in vec3 l_color, in float l_intensity, in float dist_before) {
    vec3 c = vec3(0.);
    vec3 l = get_light();
    float dist = 0.;
    float transmittance = l_intensity;
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

            // // consider diffusion
            // const int N = 2; // number of samples
            // for (int t = 0; t < N; ++t) {
            //     float div = 1. / float(N);
            //     if (n_rays >= NUM_RAYS-3) break; // check if we have enough rays

            //     // specular reflection + fresnel
            //     vec3 rand_vec = normalize(hash33(q) - 0.5); // random unit vector
            //     vec3 v_reflect = reflect(v, n + get_roughness(p, q) * rand_vec); // random unit vector on the hemisphere
            //     rays[n_rays++] = Ray(q + SURF_DIST * v_reflect, v_reflect, l_color * get_color(q), reflectance * transmittance * get_specular(q) * div, dist_before + dist);

            //     // diffuse reflection
            //     // random unit vector on the hemisphere following Lambertian distribution
            //     rand_vec = normalize(hash33(q) - 0.5); // random unit vector
            //     vec3 v_diffuse = normalize(n + rand_vec); // random unit vector on the hemisphere
            //     v_diffuse *= dot(v_diffuse, n); // cosine weighted
            //     float diffuse_intensity = 0.5 * max(0., dot(normalize(l - q), n));
            //     rays[n_rays++] = Ray(q + SURF_DIST * n, v_diffuse, l_color * get_color(q), reflectance * diffuse_intensity * transmittance * (1. - get_specular(q)) * div, dist_before + dist);

            //     // diffuse refraction
            //     rand_vec = normalize(hash33(q) - 0.5); // random unit vector
            //     vec3 v_refract = refract(v, normalize(n + get_roughness(p, q) * rand_vec), ri / ri_new);
            //     rays[n_rays++] = Ray(q, v_refract, l_color, (1. - reflectance) * transmittance * div, dist_before + dist);
            // }

            // without diffusion
            rays[n_rays++] = Ray(q, refract(v, n, ri / ri_new), l_color, (1. - reflectance) * transmittance, dist_before + dist);
            
            vec3 v_reflect = reflect(v, n);
            rays[n_rays++] = Ray(q + SURF_DIST * v_reflect, v_reflect, l_color * get_color(q), reflectance * transmittance, dist_before + dist);

            return c;
        }
        
        float step_size;
        if (ds > SAMPLE_DIST)
            step_size = ds;
        else
            step_size = SAMPLE_DIST * (hash13(q) + 0.5); // (0.5 * SAMPLE_DIST, 1.5 * SAMPLE_DIST)

        dist += step_size;
        float phase = compute_phase(0.1, dot(v, normalize(q - l)));
        c += l_color * color * (1. - exp(-density * step_size)) * intensity * transmittance * phase;
        transmittance *= exp(-density * step_size);
        
        if (transmittance < MIN_TRNASMITTANCE || dist_before + dist > MAX_DIST) break;
    }
    return c;
}

vec3 compute_color(in vec3 ro, in vec3 rd) {
    vec3 color = vec3(0.);

    rays[n_rays++] = Ray(ro, rd, vec3(1.0), 1.0, 0.0); // initialize ray: origin, direction, light color, transmittance, distance

    for (int i = 0; i < n_rays; ++i) {
        Ray ray = rays[i];
        vec3 p = ray.origin;
        vec3 v = ray.direction;
        vec3 c = ray.color;
        float t = ray.transmittance;
        float d = ray.dist;

        if (d > MAX_DIST) continue; // skip if distance exceeds max distance
        if (t < MIN_TRNASMITTANCE) continue; // skip if transmittance is too low

        color += compute_color_along_ray(p, v, c, t, d);
    }
    return color;
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
    fragColor = vec4(compute_color(ro, rd), 1.0);
}
`
export default fragment