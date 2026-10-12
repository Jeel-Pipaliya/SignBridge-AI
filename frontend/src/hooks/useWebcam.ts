import { useState, useEffect, useRef, useCallback } from 'react';

export interface UseWebcamOptions {
  width?: number;
  height?: number;
  fps?: number;
}

export const useWebcam = (options: UseWebcamOptions = {}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isActive, setIsActive] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);

  const fpsFramesRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(performance.now());

  const startCamera = useCallback(async () => {
    setError(null);
    setPermissionDenied(false);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam media devices API is not supported in this browser environment.');
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: options.width || 640 },
          height: { ideal: options.height || 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      setStream(mediaStream);
      setIsActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }
    } catch (err: any) {
      console.warn('Webcam initialization error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setPermissionDenied(true);
        setError('Camera permission was denied. Please allow camera access in your browser to enable live sign recognition.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No video capture hardware / webcam device was found on this system.');
      } else {
        setError(`Camera error: ${err.message || 'Unable to access video device.'}`);
      }
      setIsActive(false);
    }
  }, [options.width, options.height]);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    setFps(0);
  }, [stream]);

  // Frame tick counter for FPS measurement
  const markFrame = useCallback(() => {
    fpsFramesRef.current += 1;
    const now = performance.now();
    const elapsed = now - lastFpsTimeRef.current;
    if (elapsed >= 1000) {
      setFps(Math.round((fpsFramesRef.current * 1000) / elapsed));
      fpsFramesRef.current = 0;
      lastFpsTimeRef.current = now;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  return {
    videoRef,
    stream,
    isActive,
    error,
    permissionDenied,
    fps,
    startCamera,
    stopCamera,
    markFrame,
  };
};
