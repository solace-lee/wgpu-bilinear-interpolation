
import React, { useEffect, useState } from 'react';
import run from './index'

const App: React.FC = () => {
  const [ww, setWw] = useState<number>(400);
  const [wl, setWl] = useState<number>(200);
  const [invert, setInvert] = useState<boolean>(false);

  useEffect(() => {
    run()
  }, [])

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col items-center p-4 font-sans">
      <header className="w-full max-w-4xl text-center mb-4">
        <h1 className="text-4xl font-bold text-cyan-400">Rust + WASM + WGPU Image Renderer</h1>
        <p className="text-gray-400 mt-2">
          High-performance image processing using a Rust/WASM backend with WebGPU.
        </p>
      </header>

      <main className="w-full max-w-4xl flex flex-col md:flex-row gap-8 bg-gray-800 rounded-xl shadow-2xl p-6">
        <div className="flex-grow flex items-center justify-center bg-black rounded-lg overflow-hidden">
        </div>

        <div className="w-full md:w-64 flex flex-col space-y-6">
          <h2 className="text-2xl font-semibold border-b-2 border-cyan-500 pb-2">Controls</h2>

          <div>
            <label htmlFor="ww" className="block text-sm font-medium text-gray-300">
              Window Width: <span className="font-mono text-cyan-400">{ww.toFixed(0)}</span>
            </label>
            <input
              id="ww"
              type="range"
              min="1"
              max="2048"
              value={ww}
              onChange={(e) => setWw(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500 mt-2"
            />
          </div>

          <div>
            <label htmlFor="wl" className="block text-sm font-medium text-gray-300">
              Window Level: <span className="font-mono text-cyan-400">{wl.toFixed(0)}</span>
            </label>
            <input
              id="wl"
              type="range"
              min="-1024"
              max="1024"
              value={wl}
              onChange={(e) => setWl(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500 mt-2"
            />
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-gray-700">
            <span className="text-sm font-medium text-gray-300">Invert Colors</span>
            <label htmlFor="invert-toggle" className="inline-flex relative items-center cursor-pointer">
              <input type="checkbox" id="invert-toggle" className="sr-only peer" checked={invert} onChange={() => setInvert(!invert)} />
              <div className="w-11 h-6 bg-gray-600 rounded-full peer peer-focus:ring-2 peer-focus:ring-cyan-500 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
            </label>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
