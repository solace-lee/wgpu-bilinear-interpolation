
// This file defines the shape of objects passed from JS to WASM.
// It's used by wasm-bindgen for type checking.

export interface JsImageData {
  width: number;
  height: number;
  data: Float32Array;
  slope: number;
  intercept: number;
}

export interface JsViewport {
  transform: number[]; // [a, b, c, d, tx, ty]
  invert: boolean;
  canvas_size_x: number;
  canvas_size_y: number;
}

export interface JsOptions {
  ww: number;
  wl: number;
  color_map?: Float32Array;
}
