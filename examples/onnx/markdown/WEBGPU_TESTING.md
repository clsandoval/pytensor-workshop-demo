# WebGPU Performance Testing Guide

## What's Changed

I've updated your demo to actually use WebGPU and created a benchmark tool to measure the performance difference.

### Files Modified/Created:

1. **`cnn_model_webgpu_demo.html`** - Updated to try WebGPU first, with WASM fallback
2. **`webgpu_vs_wasm_benchmark.html`** - NEW: Side-by-side performance comparison tool

## Why WebGPU Wasn't Working

Your original code only used `['wasm']` as the execution provider (line 324). It wasn't even attempting to use WebGPU.

```javascript
// OLD CODE (WASM only)
const providers = ['wasm'];
session = await ort.InferenceSession.create('cnn_model.onnx', {
    executionProviders: providers
});

// NEW CODE (WebGPU with fallback)
session = await ort.InferenceSession.create('cnn_model.onnx', {
    executionProviders: ['webgpu']  // Now actually trying WebGPU!
});
```

## How to Test

### Option 1: Quick Test (Updated Demo)

1. Open `cnn_model_webgpu_demo.html` in a modern browser (Chrome/Edge 113+)
2. Check the console and UI to see which execution provider loaded
3. Run the benchmark to see performance

**Browser Requirements:**
- Chrome/Edge 113+ (WebGPU stable)
- Firefox 121+ (WebGPU behind flag: `dom.webgpu.enabled`)
- Safari 18+ (WebGPU experimental)

### Option 2: Side-by-Side Benchmark (Recommended!)

1. Open `webgpu_vs_wasm_benchmark.html` in your browser
2. Wait for both backends to initialize
3. Click "Run Benchmark (50 iterations)"
4. See the speedup multiplier!

**This will show you:**
- Average, min, max inference times for both backends
- Throughput (FPS) for each
- Speedup multiplier (e.g., "5.2x faster")
- Which backend won

## Expected Performance

Based on research and typical results:

### Small CNN Model (like yours):
- **WASM (CPU)**: 5-20ms per inference
- **WebGPU (GPU)**: 1-5ms per inference
- **Expected Speedup**: 2-10x faster with WebGPU

### Larger Models (YOLO, ResNet):
- **WASM**: 500-4000ms per inference
- **WebGPU**: 50-200ms per inference
- **Expected Speedup**: 10-30x faster with WebGPU

## Common Issues

### WebGPU Not Available?

**Check browser support:**
```javascript
if (navigator.gpu) {
    console.log('WebGPU is available!');
} else {
    console.log('WebGPU not supported');
}
```

**Common reasons:**
1. Browser version too old (need Chrome 113+)
2. GPU drivers outdated
3. Running in incognito/private mode (some browsers disable WebGPU)
4. Hardware doesn't support WebGPU (very old GPUs)

### WebGPU Fails to Initialize?

**Possible causes:**
1. ONNX model has unsupported operators for WebGPU
2. GPU memory insufficient
3. Security/permission issues

The demo automatically falls back to WASM if WebGPU fails.

## Performance Tips

### For Best WebGPU Performance:

1. **Keep data on GPU**: Avoid CPU-GPU transfers
2. **Batch processing**: Process multiple inputs together
3. **Avoid tiny models**: WebGPU has overhead; benefits larger models more
4. **Use FP16**: Half-precision can be 2x faster (if your model supports it)

### For Your Demo:

Your CNN is relatively small, so you might see "only" 2-5x speedup. But for object detection models (YOLO), you'd see 10-30x speedup, which is where WebGPU really shines!

## Next Steps

### Want to test object detection?

If WebGPU is working well, you could:

1. Export a YOLOv5n or YOLOv8n model to ONNX
2. Run it with WebGPU backend
3. Process video frames in real-time

With WebGPU, you could realistically achieve:
- **YOLOv5n**: 20-30 FPS on video
- **YOLOv8n**: 15-25 FPS on video
- **Larger models**: 5-15 FPS

This would make a **very impressive demo** since you're doing real-time object detection purely in the browser!

## Debugging

### Enable verbose logging:

Open browser DevTools console and check for:
```
✓ WebGPU detected! Attempting to use WebGPU...
✓ WebGPU successfully initialized!
  Execution Provider: WebGPU
```

If you see errors, they'll tell you exactly what went wrong.

### Check actual provider being used:

The benchmark tool queries the GPU and shows:
- GPU name
- Memory limits
- Whether WebGPU actually loaded

## Questions?

Test the benchmark and let me know:
1. What speedup you're seeing (X.XXx)
2. Whether WebGPU initialized successfully
3. Your GPU model (shown in console)

This will help determine if object detection is feasible for your demo!
