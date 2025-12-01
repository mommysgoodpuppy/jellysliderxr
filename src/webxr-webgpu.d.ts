export {};

declare global {
  interface XRGPUViewSubImage {
    colorTexture: GPUTexture;
    depthStencilTexture: GPUTexture;
    getViewDescriptor(): GPUTextureViewDescriptor;
    viewport: XRViewport;
  }

  interface XRGPUBinding {
    getPreferredColorFormat(): GPUTextureFormat | null;
    getViewSubImage(
      layer: XRProjectionLayer,
      view: XRView,
    ): XRGPUViewSubImage;
    createProjectionLayer(init?: XRProjectionLayerInit): XRProjectionLayer;
  }

  var XRGPUBinding: {
    prototype: XRGPUBinding;
    new(session: XRSession, device: GPUDevice): XRGPUBinding;
  };
}
