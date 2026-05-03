// src/components/UploadZone.jsx
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, ImageIcon, X } from "lucide-react";

export default function UploadZone({ onFile, loading }) {
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState(null);

  const onDrop = useCallback(
    (accepted) => {
      const file = accepted[0];
      if (!file) return;
      setPreview(URL.createObjectURL(file));
      setFileName(file.name);
      onFile(file);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    disabled: loading,
  });

  const clear = (e) => {
    e.stopPropagation();
    setPreview(null);
    setFileName(null);
    onFile(null);
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`
          relative flex flex-col items-center justify-center gap-3
          rounded-2xl border-2 border-dashed p-6 cursor-pointer
          transition-all duration-300 min-h-[160px]
          ${isDragActive
            ? "border-blue-400 bg-blue-500/10"
            : "border-navy-600 hover:border-blue-500/50 hover:bg-navy-700/30"}
          ${loading ? "opacity-50 pointer-events-none" : ""}
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-2 text-center">
          <div className="p-3 rounded-full bg-blue-500/10 border border-blue-500/20">
            <Upload className="w-6 h-6 text-blue-400" />
          </div>
          <p className="text-sm font-medium text-slate-300">
            {isDragActive ? "Drop it here…" : "Drag & drop an image"}
          </p>
          <p className="text-xs text-slate-500">or click to browse · JPG, PNG, WEBP</p>
        </div>
      </div>

      {/* Preview */}
      {preview && (
        <div className="relative rounded-2xl overflow-hidden border border-white/10 animate-fade-in">
          <img
            src={preview}
            alt="Preview"
            className="w-full h-52 object-cover"
          />
          {/* Overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-navy-900/80 to-transparent" />
          {/* File name */}
          <div className="absolute bottom-0 left-0 right-0 p-3 flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-slate-400 shrink-0" />
            <span className="text-xs text-slate-300 truncate">{fileName}</span>
          </div>
          {/* Clear button */}
          {!loading && (
            <button
              onClick={clear}
              className="absolute top-2 right-2 p-1.5 rounded-full bg-navy-900/80 border border-white/10
                         hover:bg-red-500/20 hover:border-red-400/30 transition-colors"
            >
              <X className="w-3.5 h-3.5 text-slate-400" />
            </button>
          )}
          {/* Loading overlay */}
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-navy-900/60 backdrop-blur-sm">
              <div className="flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-xs text-slate-400">Analysing…</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
