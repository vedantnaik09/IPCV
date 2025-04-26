// src/static/js/main.js

const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const context = canvasElement.getContext('2d');

let streaming = false;

function startStreaming() {
    const socket = new WebSocket('ws://localhost:5000/video');

    socket.onopen = function() {
        console.log('WebSocket connection established');
    };

    socket.onmessage = function(event) {
        const imageData = event.data;
        const img = new Image();
        img.src = URL.createObjectURL(new Blob([imageData]));
        img.onload = function() {
            context.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        };
    };

    socket.onclose = function() {
        console.log('WebSocket connection closed');
    };
}

function init() {
    canvasElement.width = videoElement.clientWidth;
    canvasElement.height = videoElement.clientHeight;
    startStreaming();
}

window.onload = init;