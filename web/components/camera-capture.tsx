"use client";

import { Camera, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export function CameraCapture({ onCapture }: { onCapture: (file: File) => void }) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (!open) return;
    let active = true;
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false })
      .then((stream) => {
        if (!active) return stream.getTracks().forEach((track) => track.stop());
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(() => setError("Camera access was blocked. Allow access or choose an image file."));
    return () => {
      active = false;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [open]);

  function close() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    setOpen(false);
    setError(null);
  }

  function capture() {
    const video = videoRef.current;
    if (!video?.videoWidth) return setError("The camera is still starting. Try again in a moment.");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return setError("The browser could not capture this frame.");
      onCapture(new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" }));
      close();
    }, "image/jpeg", 0.92);
  }

  if (!open) {
    return <button className="text-action" type="button" onClick={() => setOpen(true)}><Camera size={16} /> Use camera</button>;
  }
  return (
    <div className="camera-overlay" role="dialog" aria-modal="true" aria-labelledby="camera-title">
      <div className="camera-panel">
        <div className="camera-heading">
          <strong id="camera-title">Camera specimen</strong>
          <button type="button" onClick={close} aria-label="Close camera"><X size={18} /></button>
        </div>
        {error ? (
          <p className="inline-error" role="alert">{error}</p>
        ) : (
          <div className="camera-frame">
            <video ref={videoRef} autoPlay muted playsInline />
          </div>
        )}
        {!error && <button className="primary-action" type="button" onClick={capture}><Camera size={17} /> Capture frame</button>}
      </div>
    </div>
  );
}
