import React, { useRef, useEffect, useState } from 'react';
import classes from './App.module.css';
import axios from "axios";

function App() {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [start, setStart] = useState({ x: 0, y: 0 });
    const [end, setEnd] = useState({ x: 0, y: 0 });
    const [clearCanvas, setClearCanvas] = useState(false);
    const [predict, setPredict] = useState([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    useEffect(() => {
        const canvas = canvasRef.current;
        canvas.width = 28;
        canvas.height = 28;
        canvas.style.width = '400px';
        canvas.style.height = '400px';
    }, []);

    useEffect(() => {
        if (clearCanvas) {
            const ctx = canvasRef.current.getContext('2d');
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            setClearCanvas(false);
        }
    }, [clearCanvas]);

    const handleMouseDown = (e) => {
        setIsDrawing(true);
        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setStart({ x: x / (canvasRef.current.offsetWidth / canvasRef.current.width), y: y / (canvasRef.current.offsetHeight / canvasRef.current.height) });
        setEnd({ x: start.x, y: start.y });
    };

    const handleMouseMove = (e) => {
        if (isDrawing) {
            const rect = canvasRef.current.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const scaledX = x / (canvasRef.current.offsetWidth / canvasRef.current.width);
            const scaledY = y / (canvasRef.current.offsetHeight / canvasRef.current.height);
            const ctx = canvasRef.current.getContext('2d');
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(scaledX, scaledY);
            ctx.lineWidth = 2;
            ctx.stroke();
            setStart({ x: scaledX, y: scaledY });
            setEnd({ x: scaledX, y: scaledY });

            // Get canvas data and log it to the console
            const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
            const data = imageData.data;
            const convertedData = [];
            for (let i = 0; i < data.length; i += 4) {
                convertedData.push(data[i + 3]);
            }

            console.log('Reshaped Data:', convertedData); // [1, 28, 28]

            axios.post('http://127.0.0.1:8000/model/recognize', convertedData)
              .then(response => {
                console.log(response.data);
                setPredict(response.data)
              })
              .catch(error => {
                console.error(error);
            });
        }
    };

    const handleMouseUp = () => {
        setIsDrawing(false);
        setStart({ x: 0, y: 0 }); // reset start point
        setEnd({ x: 0, y: 0 }); // reset end point
    };

    const handleMouseOut = () => {
        setIsDrawing(false);
    };

    const handleClearCanvas = () => {
        setClearCanvas(true);
    };

    useEffect(() => {

    }, [predict]);

    return (
        <div className={classes.app__wrapper}>
            <div>
                <canvas
                    ref={canvasRef}
                    className={classes.canvas}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseOut={handleMouseOut}
                />
                {predict.map((pred, i) => (
                    <div key={i} style={{width: pred * 100 + "px", backgroundColor: "red", transition: "0.1s"}}>{i}</div>
                ))}
                <button onClick={handleClearCanvas}>Clear Canvas</button>
            </div>
        </div>
    );
}

export default App;