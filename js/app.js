import fragment from './canvas.frag.js'
import vertex from './canvas.vert.js'

// create canvas element
const canvas = document.createElement('canvas');
document.body.appendChild(canvas);

canvas.style.width = '100%';
canvas.style.height = '100%';

// get webgl context
const gl = canvas.getContext('webgl');
if (!gl) alert('WebGL is not supported by your browser');

// get dpr value
const dpr = window.devicePixelRatio;

// create shaders
const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertex);
const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragment);

// create shader program
const program = createProgram(gl, vertexShader, fragmentShader);

// use program
gl.useProgram(program);

// enable to cull back face
gl.enable(gl.CULL_FACE);
gl.cullFace(gl.BACK); // BACK (default), FRONT, FRONT_AND_BACK

// In default, if the winding order of vertices is CCW, their triangle create a front face
gl.frontFace(gl.CCW); // CCW (default), CW

// create attributes
createAttribute(gl, program, "a_position", 2, [
    0, 0,
    0, 1,
    1, 1,
    0, 0,
    1, 1,
    1, 0,
]);
createAttribute(gl, program, "a_normal", 3, [
    1, 0, 0,
    0, 1, 1,
    0, 0, 1,
    1, 0, 0,
    0, 0, 1,
    0, 1, 1,
]);

// look up uniform locations
const u_mousePosition = gl.getUniformLocation(program, "u_mousePosition");
const u_resolution = gl.getUniformLocation(program, "u_resolution");
const u_angle = gl.getUniformLocation(program, "u_angle");
const u_time = gl.getUniformLocation(program, "u_time");
const u_texture = gl.getUniformLocation(program, "u_texture");

const image = new Image();
image.src = 'assets/checkerboard.jpg';
image.onload = ()=>{
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Set the parameters so we can render any size image.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    // Upload the image into the texture.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

    gl.uniform1i(u_texture,  0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture);
};

let is_mouse_down = false;
let mouse_position = [0, 0];
let angle = [0, Math.PI/12];
gl.uniform2f(u_angle, angle[0], angle[1]);

canvas.addEventListener('pointerdown', function(evt){
    is_mouse_down = true;
    mouse_position[0] = dpr * evt.clientX;
    mouse_position[1] = dpr * evt.clientY;
});
canvas.addEventListener('pointerup', function(evt){
    is_mouse_down = false;
});
canvas.addEventListener('pointermove', function(evt){
    const invScaleW = 1 / canvas.width;
    const invScaleH = 1 / canvas.height;
    gl.uniform2f(u_mousePosition, dpr*evt.clientX * invScaleW, dpr*evt.clientY * invScaleH);

    if (is_mouse_down) {
        angle[0] += (dpr * evt.clientX - mouse_position[0]) * 0.01;
        angle[1] += (dpr * evt.clientY - mouse_position[1]) * 0.01;
        mouse_position[0] = dpr * evt.clientX;
        mouse_position[1] = dpr * evt.clientY;

        angle[1] = Math.min(Math.max(angle[1], 0), Math.PI/2);
        
        gl.uniform2f(u_angle, angle[0], angle[1]);
    }
});

// update the dimension of canvas
window.addEventListener('resize', resize);
resize();

// render
window.requestAnimationFrame(render);

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (success) {
        return shader;
    }

    console.log(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
}

function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    const success = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (success) {
        return program;
    }

    console.log(gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
}

function createAttribute(gl, program, name, size, value) {
    // create a buffer
    const buffer = gl.createBuffer();

    // bind it to ARRAY_BUFFER
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(value), gl.STATIC_DRAW);

    // look up where the vertex data needs to go.
    const attributeLocation = gl.getAttribLocation(program, name);

    // turn on the attribute
    gl.enableVertexAttribArray(attributeLocation);

    // tell the attribute how to get data out of buffer (ARRAY_BUFFER)
    const normalize = false; // don't normalize the data
    const stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
    const offset = 0;        // start at the beginning of the buffer
    gl.vertexAttribPointer(attributeLocation, size, gl.FLOAT, normalize, stride, offset);
}

function resize() {
    // get displayed canvas size in pixels
    const displayWidth  = Math.round(canvas.clientWidth * dpr);
    const displayHeight = Math.round(canvas.clientHeight * dpr);

    // update canvas size
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // update gl viewport size
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(u_resolution, canvas.width, canvas.height);
}

function render(t) {
    gl.uniform1f(u_time, t * 0.001);

    // set color to clear canvas
    gl.clearColor(0, 0, 0, 0);

    // clear the canvas with the above color
    gl.clear(gl.COLOR_BUFFER_BIT);

    // draw triangles
    const primitiveType = gl.TRIANGLES;
    const offset = 0;
    const count = 6;
    gl.drawArrays(primitiveType, offset, count);

    requestAnimationFrame(render.bind(this));
}
