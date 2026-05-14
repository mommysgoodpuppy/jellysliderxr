export {};

declare global {
  interface XRSessionInit {
    optionalFeatures?: (XRSessionFeature | 'hand-tracking')[];
    requiredFeatures?: (XRSessionFeature | 'hand-tracking')[];
  }

  interface XRGPUViewSubImage {
    colorTexture: GPUTexture;
    depthStencilTexture: GPUTexture;
    getViewDescriptor(): GPUTextureViewDescriptor;
    viewport: XRViewport;
  }

  interface XRGPUBinding {
    nativeProjectionScaleFactor?: number;
    getPreferredColorFormat(): GPUTextureFormat | null;
    getViewSubImage(
      layer: XRProjectionLayer,
      view: XRView,
    ): XRGPUViewSubImage;
    createProjectionLayer(
      init?: XRProjectionLayerInit & { scaleFactor?: number },
    ): XRProjectionLayer;
  }

  var XRGPUBinding: {
    prototype: XRGPUBinding;
    new(session: XRSession, device: GPUDevice): XRGPUBinding;
  };
}
