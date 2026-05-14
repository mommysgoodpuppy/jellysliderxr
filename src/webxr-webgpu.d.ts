export {};

declare global {
  type XRWebGPUProjectionLayerInit =
    & Omit<XRProjectionLayerInit, 'colorFormat' | 'depthFormat'>
    & {
      colorFormat?: GPUTextureFormat;
      alphaMode?: GPUCanvasAlphaMode;
    };

  interface XRGPUViewSubImage {
    colorTexture: GPUTexture;
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
    createProjectionLayer(init?: XRWebGPUProjectionLayerInit): XRProjectionLayer;
  }

  var XRGPUBinding: {
    prototype: XRGPUBinding;
    new(session: XRSession, device: GPUDevice): XRGPUBinding;
  };
}
