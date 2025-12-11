import init, { OffscreenImageRenderer } from './pkg/wgpu_bilinear_interpolation.js';
// import temp from './temp.json' assert { type: "json" };
import temp from './temp.json';

async function run() {
  await init(); // 初始化 WASM 模块

  // 1. 创建 Renderer (Rust 侧)
  const renderer = await OffscreenImageRenderer.new();

  const { imageData, viewport, options } = temp
  const { width, height, data, slope, intercept } = imageData
  const { transform, invert, canvasSize } = viewport
  const { ww, wl, colorMap } = options

  // 2. 准备数据
  const pixelData = new Float32Array(width * height);
  data.forEach((v, i) => {
    pixelData[i] = v
  })

  // 矩阵处理：JS 端需要先把 Transform 转换为 float array
  // 注意：Rust 代码期望的是 16 个元素的数组 (mat4)
  // 你需要把 JS 中的 transform.m (通常是6个元素) 转为 16 个元素的列主序矩阵
  const m = transform.m;
  const transformMatrix = new Float32Array([
    m[0], m[1], 0, 0,
    m[2], m[3], 0, 0,
    0, 0, 1, 0,
    m[4], m[5], 0, 1
  ]);

  const canvasWidth = canvasSize.x;
  const canvasHeight = canvasSize.y;

  // 3. 调用 Render
  try {
    const resultBytes = await renderer.render(
      width,
      height,
      pixelData,
      slope, // slope
      intercept, // intercept
      canvasWidth,
      canvasHeight,
      transformMatrix,
      invert, // invert
      ww, // ww
      wl, // wl
      null   // colormap (Option<Vec<f32>>)
    );

    // resultBytes 是 Uint8Array (Rust 的 Vec<u8>)
    // 创建 ImageData 显示
    const imageData = new ImageData(
      new Uint8ClampedArray(resultBytes.buffer),
      canvasWidth,
      canvasHeight
    );

    // ... 画到 Canvas 上 ...
    console.log('哈哈哈', imageData);
    renderToCanvas(imageData, canvasWidth, canvasHeight)
  } catch (e) {
    console.error(e);
  }
}

function renderToCanvas(imageData, width, height) {
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  canvas.width = width;
  canvas.height = height;
  canvas.style.position = 'fixed';
  canvas.style.top = '0px';
  canvas.style.left = '0px';
  canvas.style.zIndex = '10';
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
}
export default run;