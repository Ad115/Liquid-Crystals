let L = 600; // Screen box dimensions in pixels
let initialization_done = false;

var positions = [];
var screenPositions = [];
var box_length;


// --- Helper functions
            
function toScreenPos(pos, idx, posArray) {
    return [600/box_length * pos[0], 
            600/box_length * pos[1]];
}

function fetchColor2D(i) {
    return 255 * noise( 1000*i + 0.001*frameCount );
}

function fetchColor3D(i) {
    let z = positions[i][2];

    if (z/box_length < 1/2) {
        return 2*255* z/box_length;
    } else {
        return 2*255* (box_length-z)/box_length;
    }
}

function fillShadow2D(color_) {
    fill(color_, 50);
}

function fillShadow3D(color_) {
    fill(color_, (100 - color_)/2);
}

function fillBody2D(color_) {
    fill(color_);
}

function fillBody3D(color_) {
    fill(color_, 255 - color_);
}

function bodySize2D(i) {
    return 1.2*600/box_length;
}

function bodySize3D(i) {
    let z = positions[i][2];
    let size = bodySize2D(i);

    if (z/box_length < 1/2) {
        return size* (1 - 0.8*z/box_length);
    } else {
        return size* (1 + 0.8*(box_length-z)/box_length);
    }
    
}

function initialize(dimensions, _box_length) {
    let is3D = (dimensions == 3);
    if (is3D) {
        fetchColor = fetchColor3D;
        fillShadow = fillShadow3D;
        fillBody = fillBody3D;
        bodySize = bodySize3D;
    } else {
        fetchColor = fetchColor2D;
        fillShadow = fillShadow2D;
        fillBody = fillBody2D;
        bodySize = bodySize2D;
    }

    box_length = _box_length;
    initialization_done = true;
}


// --- Receive the simulation data

var ws = new WebSocket("ws://127.0.0.1:5678/");

ws.onopen = function (event) {
                ws.send(1);
            };

ws.onmessage = function (event) {
                let data = JSON.parse(event.data);

                if (!initialization_done) {
                    initialize(data.dimensions, data.box_length);
                } else
                    positions = data;
            };


// --- Drawing functions

function setup() {
    createCanvas(L, L);
    background(100);
    fill(100);
    noStroke();
}

function draw() {      //  < --- MAIN DRAWING LOOP
    background(230);
    if (ws.readyState) {
        screenPositions = positions.map(toScreenPos);
        n = screenPositions.length;
        
        // Particle shadow
        for (i=0; i<n; i++) {
            var c = fetchColor(i);
            var screenPos = screenPositions[i];

            fillShadow(c);
            ellipse(screenPos[0], screenPos[1], 1.5*bodySize(i), 1.5*bodySize(i));
        }

        // Particle
        for (i=0; i<n; i++) {
            var c = fetchColor(i);

            var screenPos = screenPositions[i];

            fillBody(c);
            ellipse(screenPos[0], screenPos[1], bodySize(i), bodySize(i));
        }
        
        ws.send(1);
    }
}