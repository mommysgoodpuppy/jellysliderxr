import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import * as sdf from '@typegpu/sdf';
import { fullScreenTriangle } from 'typegpu/common';

import { randf } from '@typegpu/noise';
import { Slider } from './slider.ts';
import { CameraController } from './camera.ts';
import * as m from 'wgpu-matrix';
import { EventHandler } from './events.ts';
import {
  DirectionalLight,
  HitInfo,
  LineInfo,
  ObjectType,
  Ray,
  rayMarchLayout,
  sampleLayout,
  SdfBbox,
} from './dataTypes.ts';
import {
  beerLambert,
  createBackgroundTexture,
  createTextures,
  fresnelSchlick,
  intersectBox,
} from './utils.ts';
import { TAAResolver } from './taa.ts';
import {
  AMBIENT_COLOR,
  AMBIENT_INTENSITY,
  AO_BIAS,
  AO_INTENSITY,
  AO_RADIUS,
  AO_STEPS,
  GROUND_ALBEDO,
  JELLY_IOR,
  JELLY_SCATTER_STRENGTH,
  LINE_HALF_THICK,
  LINE_RADIUS,
  MAX_DIST,
  MAX_STEPS,
  SPECULAR_INTENSITY,
  SPECULAR_POWER,
  SURF_DIST,
} from './constants.ts';
import { NumberProvider } from './numbers.ts';

const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const context = canvas.getContext('webgpu') as GPUCanvasContext;
const xrVrButton = document.getElementById('xr-button') as HTMLButtonElement | null;
const xrArButton = document.getElementById('xr-ar-button') as HTMLButtonElement | null;

const root = await tgpu.init({
  adapter: {
    xrCompatible: true,
  },
  device: {
    optionalFeatures: ['timestamp-query'],
  },
});
const hasTimestampQuery = root.enabledFeatures.has('timestamp-query');
context.configure({
  device: root.device,
  format: presentationFormat,
  alphaMode: 'premultiplied',
});

const NUM_POINTS = 17;

const SLIDER_START = d.vec2f(-1, 0);
const SLIDER_END = d.vec2f(0.9, 0);
const SLIDER_Y_OFFSET = -0.03;

const slider = new Slider(
  root,
  SLIDER_START,
  SLIDER_END,
  NUM_POINTS,
  SLIDER_Y_OFFSET,
);
const bezierTexture = slider.bezierTexture.createView();
const bezierBbox = slider.bbox;
const interactionPlaneHeight = (SLIDER_START[1] ?? 0) + SLIDER_Y_OFFSET;

const digitsProvider = new NumberProvider(root);
await digitsProvider.fillAtlas();
const digitsTextureView = digitsProvider.digitTextureAtlas.createView(
  d.texture2dArray(d.f32),
);

let qualityScale = 0.5;
let [width, height] = [
  canvas.width * qualityScale,
  canvas.height * qualityScale,
];

let textures = createTextures(root, width, height);
let backgroundTexture = createBackgroundTexture(root, width, height);

const filteringSampler = root['~unstable'].createSampler({
  magFilter: 'linear',
  minFilter: 'linear',
});

const baseCameraPosition = d.vec3f(0.024, 2.7, 1.9);
const baseCameraTarget = d.vec3f(0, 0, 0);
const baseCameraUp = d.vec3f(0, 1, 0);
const camera = new CameraController(
  root,
  baseCameraPosition,
  baseCameraTarget,
  baseCameraUp,
  Math.PI / 4,
  width,
  height,
);
const cameraUniform = camera.cameraUniform;

const XR_SCENE_SCALE = 0.2;
const XR_SCENE_TRANSLATION = [0, -0.5, -1.1] as const;
const xrScaleMatrix = m.mat4.scaling(
  [XR_SCENE_SCALE, XR_SCENE_SCALE, XR_SCENE_SCALE],
  m.mat4.identity(),
);
const xrTranslationMatrix = m.mat4.translation(
  XR_SCENE_TRANSLATION,
  m.mat4.identity(),
);
const xrSceneTransform = m.mat4.mul(
  xrTranslationMatrix,
  xrScaleMatrix,
  m.mat4.identity(),
);
const xrSceneTransformInv = m.mat4.invert(
  xrSceneTransform,
  m.mat4.identity(),
) ?? m.mat4.identity();

const MAX_HAND_JOINTS = 50;
const XR_HAND_JOINTS: XRHandJoint[] = [
  'wrist',
  'thumb-metacarpal',
  'thumb-phalanx-proximal',
  'thumb-phalanx-distal',
  'thumb-tip',
  'index-finger-metacarpal',
  'index-finger-phalanx-proximal',
  'index-finger-phalanx-intermediate',
  'index-finger-phalanx-distal',
  'index-finger-tip',
  'middle-finger-metacarpal',
  'middle-finger-phalanx-proximal',
  'middle-finger-phalanx-intermediate',
  'middle-finger-phalanx-distal',
  'middle-finger-tip',
  'ring-finger-metacarpal',
  'ring-finger-phalanx-proximal',
  'ring-finger-phalanx-intermediate',
  'ring-finger-phalanx-distal',
  'ring-finger-tip',
  'pinky-finger-metacarpal',
  'pinky-finger-phalanx-proximal',
  'pinky-finger-phalanx-intermediate',
  'pinky-finger-phalanx-distal',
  'pinky-finger-tip',
];
const XR_HAND_CAPTURE_RADIUS = 0.12;
const XR_HAND_RELEASE_RADIUS = 0.18;
const XR_HAND_APPROACH_MIN = 0.12;
const XR_HAND_APPROACH_MAX = 0.45;
const XR_HAND_VISUAL_SCALE = 6.0;
const XR_HAND_RENDER_MARGIN = 0.6;
const XR_PINCH_MAX_DISTANCE = 0.04;
const HAND_LEFT_TINT = d.vec3f(1.0, 0.63, 0.5);
const HAND_RIGHT_TINT = d.vec3f(0.5, 0.82, 1.0);

const XR_INTERACTION_JOINT: XRHandJoint = 'index-finger-tip';
const XR_HAND_Z_THRESHOLD = 0.08;
const XR_HAND_X_MARGIN = 0.12;
const XR_HAND_Y_MARGIN = 0.12;
const XR_HAND_RELEASE_MS = 150;
const XR_HAND_SMOOTHING = 0.35;
const XR_CONTROLLER_Z_THRESHOLD = 0.18;
const XR_CONTROLLER_ACTION_THRESHOLD = 0.35;
const XR_CONTROLLER_BUTTON_INDICES = [0, 1, 3, 4];
const XR_DEBUG_LOGS = false;

function logXrDebug(...args: unknown[]) {
  if (!XR_DEBUG_LOGS) return;
  console.log('[XR]', ...args);
}

interface XrHandCandidate {
  dragX: number;
  score: number;
  handedness: XRHandedness;
  tipDistance: number;
  withinBounds: boolean;
  isPinching: boolean;
}

const xrHandInteraction = {
  dragX: null as number | null,
  activeHand: null as XRHandedness | null,
  lastSeenTime: 0,
  isPinching: false,
};

const HandJointBuffer = d.struct({
  joints: d.arrayOf(d.vec4f, MAX_HAND_JOINTS),
  count: d.u32,
  padding: d.vec3f,
});
const handJointData = Array.from({ length: MAX_HAND_JOINTS }, () =>
  d.vec4f(0, -10, 0, 0)
);
const handJointPadding = d.vec3f(0, 0, 0);
const handJointsUniform = root.createUniform(HandJointBuffer, {
  joints: handJointData,
  count: d.u32(0),
  padding: handJointPadding,
});

const lightUniform = root.createUniform(DirectionalLight, {
  direction: std.normalize(d.vec3f(0.19, -0.24, 0.75)),
  color: d.vec3f(1, 1, 1),
});

const jellyColorUniform = root.createUniform(
  d.vec4f,
  d.vec4f(1.0, 0.45, 0.075, 1.0),
);

const randomUniform = root.createUniform(d.vec2f);
const blurEnabledUniform = root.createUniform(d.u32);
const interactionOnlyUniform = root.createUniform(d.u32, d.u32(0));
const interactionPlaneUniform = root.createUniform(
  d.f32,
  d.f32(interactionPlaneHeight),
);

const getRay = (ndc: d.v2f) => {
  'use gpu';
  const clipPos = d.vec4f(ndc.x, ndc.y, -1.0, 1.0);

  const invView = cameraUniform.$.viewInv;
  const invProj = cameraUniform.$.projInv;

  const viewPos = invProj.mul(clipPos);
  const viewPosNormalized = d.vec4f(viewPos.xyz.div(viewPos.w), 1.0);

  const worldPos = invView.mul(viewPosNormalized);

  const rayOrigin = invView.columns[3].xyz;
  const rayDir = std.normalize(worldPos.xyz.sub(rayOrigin));

  return Ray({
    origin: rayOrigin,
    direction: rayDir,
  });
};

const getSliderBbox = () => {
  'use gpu';
  return SdfBbox({
    left: d.f32(bezierBbox[3]),
    right: d.f32(bezierBbox[1]),
    bottom: d.f32(bezierBbox[2]),
    top: d.f32(bezierBbox[0]),
  });
};

const sdInflatedPolyline2D = (p: d.v2f) => {
  'use gpu';
  const bbox = getSliderBbox();

  const uv = d.vec2f(
    (p.x - bbox.left) / (bbox.right - bbox.left),
    (bbox.top - p.y) / (bbox.top - bbox.bottom),
  );
  const clampedUV = std.saturate(uv);

  const sampledColor = std.textureSampleLevel(
    bezierTexture.$,
    filteringSampler.$,
    clampedUV,
    0,
  );
  const segUnsigned = sampledColor.x;
  const progress = sampledColor.y;
  const normal = sampledColor.zw;

  return LineInfo({
    t: progress,
    distance: segUnsigned,
    normal: normal,
  });
};

const cap3D = (position: d.v3f) => {
  'use gpu';
  const endCap = slider.endCapUniform.$;
  const secondLastPoint = d.vec2f(endCap.x, endCap.y);
  const lastPoint = d.vec2f(endCap.z, endCap.w);

  const angle = std.atan2(
    lastPoint.y - secondLastPoint.y,
    lastPoint.x - secondLastPoint.x,
  );
  const rot = d.mat2x2f(
    std.cos(angle),
    -std.sin(angle),
    std.sin(angle),
    std.cos(angle),
  );

  let pieP = position.sub(d.vec3f(secondLastPoint, 0));
  pieP = d.vec3f(rot.mul(pieP.xy), pieP.z);
  const hmm = sdf.sdPie(pieP.zx, d.vec2f(1, 0), LINE_HALF_THICK);
  const extrudeEnd = sdf.opExtrudeY(
    pieP,
    hmm,
    0.001,
  ) - LINE_RADIUS;
  return extrudeEnd;
};

const sliderSdf3D = (position: d.v3f) => {
  'use gpu';
  const poly2D = sdInflatedPolyline2D(position.xy);

  let finalDist = d.f32(0.0);
  if (poly2D.t > 0.94) {
    finalDist = cap3D(position);
  } else {
    const body = sdf.opExtrudeZ(position, poly2D.distance, LINE_HALF_THICK) -
      LINE_RADIUS;
    finalDist = body;
  }

  return LineInfo({
    t: poly2D.t,
    distance: finalDist,
    normal: poly2D.normal,
  });
};

const GroundParams = {
  groundThickness: 0.03,
  groundRoundness: 0.02,
};

const rectangleCutoutDist = (position: d.v2f) => {
  'use gpu';
  const groundRoundness = GroundParams.groundRoundness;

  return sdf.sdRoundedBox2d(
    position,
    d.vec2f(1 + groundRoundness, 0.2 + groundRoundness),
    0.2 + groundRoundness,
  );
};

const getMainSceneDist = (position: d.v3f) => {
  'use gpu';
  const groundThickness = GroundParams.groundThickness;
  const groundRoundness = GroundParams.groundRoundness;

  return sdf.opUnion(
    sdf.sdPlane(position, d.vec3f(0, 1, 0), 0.06),
    sdf.opExtrudeY(
      position,
      -rectangleCutoutDist(position.xz),
      groundThickness - groundRoundness,
    ) - groundRoundness,
  );
};

const sliderApproxDist = (position: d.v3f) => {
  'use gpu';
  const bbox = getSliderBbox();

  const p = position.xy;
  if (
    p.x < bbox.left || p.x > bbox.right || p.y < bbox.bottom || p.y > bbox.top
  ) {
    return 1e9;
  }

  const poly2D = sdInflatedPolyline2D(p);
  const dist3D = sdf.opExtrudeZ(position, poly2D.distance, LINE_HALF_THICK) -
    LINE_RADIUS;

  return dist3D;
};

const HandJointHit = d.struct({
  distance: d.f32,
  jointIndex: d.i32,
});

const getHandJointDistance = (position: d.v3f) => {
  'use gpu';
  const result = HandJointHit({
    distance: d.f32(1e6),
    jointIndex: d.i32(-1),
  });
  const jointCount = d.i32(handJointsUniform.$.count);

  for (let i = 0; i < MAX_HAND_JOINTS; i++) {
    if (i >= jointCount) {
      break;
    }
    const joint = handJointsUniform.$.joints[i];
    const radius = std.abs(joint.w);
    if (radius <= 0) {
      continue;
    }
    const dist = std.length(position.sub(joint.xyz)) - radius;
    if (dist < result.distance) {
      result.distance = dist;
      result.jointIndex = i;
    }
  }

  return result;
};

const getSceneDist = (position: d.v3f) => {
  'use gpu';
  const mainScene = getMainSceneDist(position);
  const poly3D = sliderSdf3D(position);
  const handJoint = getHandJointDistance(position);

  const hitInfo = HitInfo();

  hitInfo.distance = mainScene;
  hitInfo.objectType = ObjectType.BACKGROUND;
  hitInfo.t = 0;

  if (handJoint.jointIndex >= 0 && handJoint.distance < hitInfo.distance) {
    hitInfo.distance = handJoint.distance;
    hitInfo.objectType = ObjectType.HAND;
    hitInfo.t = d.f32(handJoint.jointIndex);
  }

  if (poly3D.distance < hitInfo.distance) {
    hitInfo.distance = poly3D.distance;
    hitInfo.objectType = ObjectType.SLIDER;
    hitInfo.t = poly3D.t;
  }
  return hitInfo;
};

const getSceneDistForAO = (position: d.v3f) => {
  'use gpu';
  const mainScene = getMainSceneDist(position);
  const sliderApprox = sliderApproxDist(position);
  return std.min(mainScene, sliderApprox);
};

const sdfSlot = tgpu.slot<(pos: d.v3f) => number>();

const getNormalFromSdf = tgpu.fn([d.vec3f, d.f32], d.vec3f)(
  (position, epsilon) => {
    'use gpu';
    const k = d.vec3f(1, -1, 0);

    const offset1 = k.xyy.mul(epsilon);
    const offset2 = k.yyx.mul(epsilon);
    const offset3 = k.yxy.mul(epsilon);
    const offset4 = k.xxx.mul(epsilon);

    const sample1 = offset1.mul(sdfSlot.$(position.add(offset1)));
    const sample2 = offset2.mul(sdfSlot.$(position.add(offset2)));
    const sample3 = offset3.mul(sdfSlot.$(position.add(offset3)));
    const sample4 = offset4.mul(sdfSlot.$(position.add(offset4)));

    const gradient = sample1.add(sample2).add(sample3).add(sample4);

    return std.normalize(gradient);
  },
);

const getNormalCapSdf = getNormalFromSdf.with(sdfSlot, cap3D);
const getNormalMainSdf = getNormalFromSdf.with(sdfSlot, getMainSceneDist);

const getNormalCap = (pos: d.v3f) => {
  'use gpu';
  return getNormalCapSdf(pos, 0.01);
};

const getNormalMain = (position: d.v3f) => {
  'use gpu';
  if (std.abs(position.z) > 0.22 || std.abs(position.x) > 1.02) {
    return d.vec3f(0, 1, 0);
  }
  return getNormalMainSdf(position, 0.0001);
};

const getSliderNormal = (
  position: d.v3f,
  hitInfo: d.Infer<typeof HitInfo>,
) => {
  'use gpu';
  const poly2D = sdInflatedPolyline2D(position.xy);
  const gradient2D = poly2D.normal;

  const threshold = LINE_HALF_THICK * 0.85;
  const absZ = std.abs(position.z);
  const zDistance = std.max(
    0,
    (absZ - threshold) * LINE_HALF_THICK / (LINE_HALF_THICK - threshold),
  );
  const edgeDistance = LINE_RADIUS - poly2D.distance;

  const edgeContrib = 0.9;
  const zContrib = 1.0 - edgeContrib;

  const zDirection = std.sign(position.z);
  const zAxisVector = d.vec3f(0, 0, zDirection);

  const edgeBlendDistance = edgeContrib * LINE_RADIUS +
    zContrib * LINE_HALF_THICK;

  const blendFactor = std.smoothstep(
    edgeBlendDistance,
    0.0,
    zDistance * zContrib + edgeDistance * edgeContrib,
  );

  const normal2D = d.vec3f(gradient2D.xy, 0);
  const blendedNormal = std.mix(
    zAxisVector,
    normal2D,
    blendFactor * 0.5 + 0.5,
  );

  let normal = std.normalize(blendedNormal);

  if (hitInfo.t > 0.94) {
    const ratio = (hitInfo.t - 0.94) / 0.02;
    const fullNormal = getNormalCap(position);
    normal = std.normalize(std.mix(normal, fullNormal, ratio));
  }

  return normal;
};

const getNormal = (
  position: d.v3f,
  hitInfo: d.Infer<typeof HitInfo>,
) => {
  'use gpu';
  if (hitInfo.objectType === ObjectType.SLIDER && hitInfo.t < 0.96) {
    return getSliderNormal(position, hitInfo);
  }

  return std.select(
    getNormalCap(position),
    getNormalMain(position),
    hitInfo.objectType === ObjectType.BACKGROUND,
  );
};

const sqLength = (a: d.v3f) => {
  'use gpu';
  return std.dot(a, a);
};

const getFakeShadow = (
  position: d.v3f,
  lightDir: d.v3f,
): d.v3f => {
  'use gpu';
  const jellyColor = jellyColorUniform.$;
  const endCapX = slider.endCapUniform.$.x;

  if (position.y < -GroundParams.groundThickness) {
    // Applying darkening under the ground (the shadow cast by the upper ground layer)
    const fadeSharpness = 30;
    const inset = 0.02;
    const cutout = rectangleCutoutDist(position.xz) + inset;
    const edgeDarkening = std.saturate(1 - cutout * fadeSharpness);

    // Applying a slight gradient based on the light direction
    const lightGradient = std.saturate(-position.z * 4 * lightDir.z + 1);

    return d.vec3f(1).mul(edgeDarkening).mul(lightGradient * 0.5);
  } else {
    const finalUV = d.vec2f(
      (position.x - position.z * lightDir.x * std.sign(lightDir.z)) *
          0.5 + 0.5,
      1 - (-position.z / lightDir.z * 0.5) - 0.2,
    );
    const data = std.textureSampleLevel(
      bezierTexture.$,
      filteringSampler.$,
      finalUV,
      0,
    );

    // Normally it would be just data.y, but there transition is too sudden when the jelly is bunched up.
    // To mitigate this, we transition into a position-based transition.
    const jellySaturation = std.mix(
      0,
      data.y,
      std.saturate(position.x * 1.5 + 1.1),
    );
    const shadowColor = std.mix(
      d.vec3f(0, 0, 0),
      jellyColor.xyz,
      jellySaturation,
    );

    const contrast = 20 * std.saturate(finalUV.y) * (0.8 + endCapX * 0.2);
    const shadowOffset = -0.3;
    const featherSharpness = 10;
    const uvEdgeFeather = std.saturate(finalUV.x * featherSharpness) *
      std.saturate((1 - finalUV.x) * featherSharpness) *
      std.saturate((1 - finalUV.y) * featherSharpness) *
      std.saturate(finalUV.y);
    const influence = std.saturate((1 - lightDir.y) * 2) * uvEdgeFeather;
    return std.mix(
      d.vec3f(1),
      std.mix(
        shadowColor,
        d.vec3f(1),
        std.saturate(data.x * contrast + shadowOffset),
      ),
      influence,
    );
  }
};

const calculateAO = (position: d.v3f, normal: d.v3f) => {
  'use gpu';
  let totalOcclusion = d.f32(0.0);
  let sampleWeight = d.f32(1.0);
  const stepDistance = AO_RADIUS / AO_STEPS;

  for (let i = 1; i <= AO_STEPS; i++) {
    const sampleHeight = stepDistance * d.f32(i);
    const samplePosition = position.add(normal.mul(sampleHeight));
    const distanceToSurface = getSceneDistForAO(samplePosition) - AO_BIAS;
    const occlusionContribution = std.max(
      0.0,
      sampleHeight - distanceToSurface,
    );
    totalOcclusion += occlusionContribution * sampleWeight;
    sampleWeight *= 0.5;
    if (totalOcclusion > AO_RADIUS / AO_INTENSITY) {
      break;
    }
  }

  const rawAO = 1.0 - (AO_INTENSITY * totalOcclusion) / AO_RADIUS;
  return std.saturate(rawAO);
};

const calculateLighting = (
  hitPosition: d.v3f,
  normal: d.v3f,
  rayOrigin: d.v3f,
) => {
  'use gpu';
  const lightDir = std.neg(lightUniform.$.direction);

  const fakeShadow = getFakeShadow(hitPosition, lightDir);
  const diffuse = std.max(std.dot(normal, lightDir), 0.0);

  const viewDir = std.normalize(rayOrigin.sub(hitPosition));
  const reflectDir = std.reflect(std.neg(lightDir), normal);
  const specularFactor = std.max(std.dot(viewDir, reflectDir), 0) **
    SPECULAR_POWER;
  const specular = lightUniform.$.color.mul(
    specularFactor * SPECULAR_INTENSITY,
  );

  const baseColor = d.vec3f(0.9);

  const directionalLight = baseColor
    .mul(lightUniform.$.color)
    .mul(diffuse)
    .mul(fakeShadow);
  const ambientLight = baseColor.mul(AMBIENT_COLOR).mul(AMBIENT_INTENSITY);

  const finalSpecular = specular.mul(fakeShadow);

  return std.saturate(directionalLight.add(ambientLight).add(finalSpecular));
};

const applyAO = (
  litColor: d.v3f,
  hitPosition: d.v3f,
  normal: d.v3f,
) => {
  'use gpu';
  const ao = calculateAO(hitPosition, normal);
  const finalColor = litColor.mul(ao);
  return d.vec4f(finalColor, 1.0);
};

const rayMarchNoJelly = (rayOrigin: d.v3f, rayDirection: d.v3f) => {
  'use gpu';
  let distanceFromOrigin = d.f32();
  let hit = d.f32();

  for (let i = 0; i < 6; i++) {
    const p = rayOrigin.add(rayDirection.mul(distanceFromOrigin));
    hit = getMainSceneDist(p);
    distanceFromOrigin += hit;
    if (distanceFromOrigin > MAX_DIST || hit < SURF_DIST * 10) {
      break;
    }
  }

  if (distanceFromOrigin < MAX_DIST) {
    return renderBackground(
      rayOrigin,
      rayDirection,
      distanceFromOrigin,
      std.select(d.f32(), 0.87, blurEnabledUniform.$ === 1),
    ).xyz;
  }
  return d.vec3f();
};

const renderPercentageOnGround = (
  hitPosition: d.v3f,
  center: d.v3f,
  percentage: number,
) => {
  'use gpu';

  const textWidth = 0.38;
  const textHeight = 0.33;

  if (
    std.abs(hitPosition.x - center.x) > textWidth * 0.5 ||
    std.abs(hitPosition.z - center.z) > textHeight * 0.5
  ) {
    return d.vec4f();
  }

  const localX = hitPosition.x - center.x;
  const localZ = hitPosition.z - center.z;

  const uvX = (localX + textWidth * 0.5) / textWidth;
  const uvZ = (localZ + textHeight * 0.5) / textHeight;

  if (uvX < 0.0 || uvX > 1.0 || uvZ < 0.0 || uvZ > 1.0) {
    return d.vec4f();
  }

  return std.textureSampleLevel(
    digitsTextureView.$,
    filteringSampler.$,
    d.vec2f(uvX, uvZ),
    percentage,
    0,
  );
};

const renderBackground = (
  rayOrigin: d.v3f,
  rayDirection: d.v3f,
  backgroundHitDist: number,
  offset: number,
) => {
  'use gpu';
  if (interactionOnlyUniform.$ === 1) {
    return d.vec4f();
  }
  const hitPosition = rayOrigin.add(rayDirection.mul(backgroundHitDist));

  const percentageSample = renderPercentageOnGround(
    hitPosition,
    d.vec3f(0.72, 0, 0),
    d.u32((slider.endCapUniform.$.x + 0.43) * 84),
  );

  let highlights = d.f32();

  const highlightWidth = d.f32(1);
  const highlightHeight = 0.2;
  let offsetX = d.f32();
  let offsetZ = d.f32(0.05);

  const lightDir = lightUniform.$.direction;
  const causticScale = 0.2;
  offsetX -= lightDir.x * causticScale;
  offsetZ += lightDir.z * causticScale;

  const endCapX = slider.endCapUniform.$.x;
  const sliderStretch = (endCapX + 1) * 0.5;

  if (
    std.abs(hitPosition.x + offsetX) < highlightWidth &&
    std.abs(hitPosition.z + offsetZ) < highlightHeight
  ) {
    const uvX_orig = (hitPosition.x + offsetX + highlightWidth * 2) /
      highlightWidth * 0.5;
    const uvZ_orig = (hitPosition.z + offsetZ + highlightHeight * 2) /
      highlightHeight * 0.5;

    const centeredUV = d.vec2f(uvX_orig - 0.5, uvZ_orig - 0.5);
    const finalUV = d.vec2f(
      centeredUV.x,
      1 - ((std.abs(centeredUV.y - 0.5) * 2) ** 2) * 0.3,
    );

    const density = std.max(
      0,
      (std.textureSampleLevel(bezierTexture.$, filteringSampler.$, finalUV, 0)
        .x - 0.25) * 8,
    );

    const fadeX = std.smoothstep(0, -0.2, hitPosition.x - endCapX);
    const fadeZ = 1 - (std.abs(centeredUV.y - 0.5) * 2) ** 3;
    const fadeStretch = std.saturate(1 - sliderStretch);
    const edgeFade = std.saturate(fadeX) * std.saturate(fadeZ) * fadeStretch;

    highlights = density ** 3 * edgeFade * 3 * (1 + lightDir.z) / 1.5;
  }

  const originYBound = std.saturate(rayOrigin.y + 0.01);
  const posOffset = hitPosition.add(
    d.vec3f(0, 1, 0).mul(
      offset *
        (originYBound / (1.0 + originYBound)) *
        (1 + randf.sample() / 2),
    ),
  );
  const newNormal = getNormalMain(posOffset);

  // Calculate fake bounce lighting
  const jellyColor = jellyColorUniform.$;
  const sqDist = sqLength(hitPosition.sub(d.vec3f(endCapX, 0, 0)));
  const bounceLight = jellyColor.xyz.mul(1 / (sqDist * 15 + 1) * 0.4);
  const sideBounceLight = jellyColor.xyz
    .mul(1 / (sqDist * 40 + 1) * 0.3)
    .mul(std.abs(newNormal.z));

  const litColor = calculateLighting(posOffset, newNormal, rayOrigin);
  const backgroundColor = applyAO(
    GROUND_ALBEDO.mul(litColor),
    posOffset,
    newNormal,
  )
    .add(d.vec4f(bounceLight, 0))
    .add(d.vec4f(sideBounceLight, 0));

  const textColor = std.saturate(backgroundColor.xyz.mul(d.vec3f(0.5)));

  return d.vec4f(
    std.mix(backgroundColor.xyz, textColor, percentageSample.x).mul(
      1.0 + highlights,
    ),
    1.0,
  );
};

const renderInteractionOverlay = (
  rayOrigin: d.v3f,
  rayDirection: d.v3f,
) => {
  'use gpu';
  const planeY = interactionPlaneUniform.$;
  const denom = rayDirection.y;
  if (std.abs(denom) < 1e-4) {
    return d.vec4f();
  }
  const t = (planeY - rayOrigin.y) / denom;
  if (t <= 0.0) {
    return d.vec4f();
  }
  const hitPosition = rayOrigin.add(rayDirection.mul(t));

  const bbox = getSliderBbox();
  const left = bbox.left - XR_HAND_X_MARGIN;
  const right = bbox.right + XR_HAND_X_MARGIN;
  const bottom = bbox.bottom - XR_HAND_Y_MARGIN;
  const top = bbox.top + XR_HAND_Y_MARGIN;
  const depth = XR_HAND_RENDER_MARGIN;

  if (
    hitPosition.x < left ||
    hitPosition.x > right ||
    hitPosition.y < bottom ||
    hitPosition.y > top ||
    std.abs(hitPosition.z) > depth
  ) {
    return d.vec4f();
  }

  const u = std.saturate((hitPosition.x - left) / (right - left));
  const v = std.saturate((hitPosition.z + depth) / (depth * 2));
  const edge = std.min(
    std.min(u, 1 - u),
    std.min(v, 1 - v),
  );
  const highlight = std.smoothstep(0.0, 0.3, edge);
  const rim = std.smoothstep(0.0, 0.05, edge);
  const tint = d.vec3f(0.12, 0.6, 1.0);
  const color = tint.mul(0.25 + highlight * 0.65);
  const rimColor = d.vec3f(0.9, 0.95, 1.0).mul(rim * 0.4);

  const grid = std.sin((u + v) * 40.0) * 0.02;
  const finalColor = std.saturate(color.add(rimColor).add(d.vec3f(grid)));
  const alpha = 0.15 + highlight * 0.25;

  return d.vec4f(finalColor, alpha);
};

const renderHandJoint = (
  hitPosition: d.v3f,
  hitInfo: d.Infer<typeof HitInfo>,
  rayOrigin: d.v3f,
) => {
  'use gpu';
  let jointIndex = d.i32(std.floor(hitInfo.t + 0.5));
  if (jointIndex < 0) {
    jointIndex = 0;
  }
  if (jointIndex >= MAX_HAND_JOINTS) {
    jointIndex = MAX_HAND_JOINTS - 1;
  }

  const joint = handJointsUniform.$.joints[jointIndex];
  const encodedRadius = joint.w;
  const normal = std.normalize(hitPosition.sub(joint.xyz));

  const tint = std.mix(
    HAND_LEFT_TINT,
    HAND_RIGHT_TINT,
    std.step(0.0, encodedRadius),
  );
  const litColor = calculateLighting(hitPosition, normal, rayOrigin).mul(tint);
  const rim = std.pow(
    1 -
      std.max(
        0.0,
        std.dot(normal, std.normalize(rayOrigin.sub(hitPosition))),
      ),
    2.0,
  );
  const glow = tint.mul(rim * 0.25);

  return d.vec4f(std.saturate(litColor.add(glow)), 1.0);
};

const rayMarch = (rayOrigin: d.v3f, rayDirection: d.v3f, uv: d.v2f) => {
  'use gpu';
  let totalSteps = d.u32();

  const isInteractionOnly = interactionOnlyUniform.$ === 1;

  let backgroundDist = d.f32();
  if (isInteractionOnly) {
    backgroundDist = d.f32(MAX_DIST);
  }
  let background = d.vec4f();

  if (isInteractionOnly) {
    background = renderInteractionOverlay(rayOrigin, rayDirection);
  } else {
    for (let i = 0; i < MAX_STEPS; i++) {
      const p = rayOrigin.add(rayDirection.mul(backgroundDist));
      const hit = getMainSceneDist(p);
      backgroundDist += hit;
      if (hit < SURF_DIST) {
        break;
      }
    }
    background = renderBackground(
      rayOrigin,
      rayDirection,
      backgroundDist,
      d.f32(),
    );
  }

  const bbox = getSliderBbox();
  const zDepth = d.f32(0.25);

  const sliderMin = d.vec3f(
    bbox.left - XR_HAND_RENDER_MARGIN,
    bbox.bottom - XR_HAND_RENDER_MARGIN,
    -zDepth - XR_HAND_RENDER_MARGIN,
  );
  const sliderMax = d.vec3f(
    bbox.right + XR_HAND_RENDER_MARGIN,
    bbox.top + XR_HAND_RENDER_MARGIN,
    zDepth + XR_HAND_RENDER_MARGIN,
  );

  const intersection = intersectBox(
    rayOrigin,
    rayDirection,
    sliderMin,
    sliderMax,
  );

  if (!intersection.hit) {
    return background;
  }

  let distanceFromOrigin = std.max(d.f32(0.0), intersection.tMin);

  for (let i = 0; i < MAX_STEPS; i++) {
    if (totalSteps >= MAX_STEPS) {
      break;
    }

    const currentPosition = rayOrigin.add(rayDirection.mul(distanceFromOrigin));

    const hitInfo = getSceneDist(currentPosition);
    distanceFromOrigin += hitInfo.distance;
    totalSteps++;

    if (hitInfo.distance < SURF_DIST) {
      const hitPosition = rayOrigin.add(rayDirection.mul(distanceFromOrigin));

      if (hitInfo.objectType === ObjectType.SLIDER) {
        const N = getNormal(hitPosition, hitInfo);
        const I = rayDirection;
        const cosi = std.min(
          1.0,
          std.max(0.0, std.dot(std.neg(I), N)),
        );
        const F = fresnelSchlick(cosi, d.f32(1.0), d.f32(JELLY_IOR));

        const reflection = std.saturate(d.vec3f(hitPosition.y + 0.2));

        const eta = 1.0 / JELLY_IOR;
        const k = 1.0 - eta * eta * (1.0 - cosi * cosi);
        let refractedColor = d.vec3f();
        if (k > 0.0) {
          const refrDir = std.normalize(
            std.add(
              I.mul(eta),
              N.mul(eta * cosi - std.sqrt(k)),
            ),
          );
          const p = hitPosition.add(refrDir.mul(SURF_DIST * 2.0));
          const exitPos = p.add(refrDir.mul(SURF_DIST * 2.0));

          const env = rayMarchNoJelly(exitPos, refrDir);
          const progress = hitInfo.t;
          const jellyColor = jellyColorUniform.$;

          const scatterTint = jellyColor.xyz.mul(1.5);
          const density = d.f32(20.0);
          const absorb = d.vec3f(1.0).sub(jellyColor.xyz).mul(density);

          const T = beerLambert(absorb.mul(progress ** 2), 0.08);

          const lightDir = std.neg(lightUniform.$.direction);

          const forward = std.max(0.0, std.dot(lightDir, refrDir));
          const scatter = scatterTint.mul(
            JELLY_SCATTER_STRENGTH * forward * progress ** 3,
          );
          refractedColor = env.mul(T).add(scatter);
        }

        const jelly = std.add(
          reflection.mul(F),
          refractedColor.mul(1 - F),
        );

        return d.vec4f(jelly, 1.0);
      }

      if (hitInfo.objectType === ObjectType.HAND) {
        return renderHandJoint(hitPosition, hitInfo, rayOrigin);
      }

      break;
    }

    if (!isInteractionOnly && distanceFromOrigin > backgroundDist) {
      break;
    }
  }

  return background;
};

const raymarchFn = tgpu['~unstable'].fragmentFn({
  in: {
    uv: d.vec2f,
  },
  out: d.vec4f,
})(({ uv }) => {
  randf.seed2(randomUniform.$.mul(uv));

  const ndc = d.vec2f(uv.x * 2 - 1, -(uv.y * 2 - 1));
  const ray = getRay(ndc);

  const color = rayMarch(
    ray.origin,
    ray.direction,
    uv,
  );
  return d.vec4f(std.tanh(color.xyz.mul(1.3)), 1);
});

const fragmentMain = tgpu['~unstable'].fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  return std.textureSample(
    sampleLayout.$.currentTexture,
    filteringSampler.$,
    input.uv,
  );
});

const RAY_MARCH_FORMAT: GPUTextureFormat = 'rgba8unorm';
const rayMarchPipelines = new Map<
  GPUTextureFormat,
  ReturnType<typeof createRayMarchPipeline>
>();

function createRayMarchPipeline(format: GPUTextureFormat) {
  return root['~unstable']
    .withVertex(fullScreenTriangle, {})
    .withFragment(raymarchFn, { format })
    .createPipeline();
}

function getRayMarchPipeline(format: GPUTextureFormat) {
  let pipeline = rayMarchPipelines.get(format);
  if (!pipeline) {
    pipeline = createRayMarchPipeline(format);
    rayMarchPipelines.set(format, pipeline);
  }
  return pipeline;
}

const renderPipeline = root['~unstable']
  .withVertex(fullScreenTriangle, {})
  .withFragment(fragmentMain, { format: presentationFormat })
  .createPipeline();

const eventHandler = new EventHandler(canvas);
let sliderInputX = eventHandler.currentMouseX;
let lastTimeStamp = performance.now();
let frameCount = 0;
const taaResolver = new TAAResolver(root, width, height);
let xrSession: XRSession | null = null;
let xrRefSpace: XRReferenceSpace | null = null;
let xrGpuBinding: XRGPUBinding | null = null;
let xrProjectionLayer: XRProjectionLayer | null = null;
let xrColorFormat: GPUTextureFormat | null = null;
let isPresentingXr = false;
let activeXrMode: XRSessionMode | null = null;
let xrVrSupported = false;
let xrArSupported = false;

let attributionDismissed = false;
const attributionElement = document.getElementById(
  'attribution',
) as HTMLDivElement;

function dismissAttribution() {
  if (!attributionDismissed && attributionElement) {
    attributionElement.style.opacity = '0';
    attributionElement.style.pointerEvents = 'none';
    attributionDismissed = true;
  }
}

canvas.addEventListener('mousedown', dismissAttribution, { once: true });
canvas.addEventListener('touchstart', dismissAttribution, { once: true });
canvas.addEventListener('wheel', dismissAttribution, { once: true });

function createBindGroups() {
  return {
    rayMarch: root.createBindGroup(rayMarchLayout, {
      backgroundTexture: backgroundTexture.sampled,
    }),
    render: [0, 1].map((frame) =>
      root.createBindGroup(sampleLayout, {
        currentTexture: taaResolver.getResolvedTexture(frame),
      })
    ),
  };
}

let bindGroups = createBindGroups();

function getDeltaSeconds(timestamp: number) {
  if (!Number.isFinite(timestamp)) {
    return 0;
  }
  const delta = Math.min((timestamp - lastTimeStamp) * 0.001, 0.1);
  lastTimeStamp = timestamp;
  return delta;
}

function updateSimulation(deltaTime: number, overrideDragX?: number | null) {
  randomUniform.write(
    d.vec2f((Math.random() - 0.5) * 2, (Math.random() - 0.5) * 2),
  );

  eventHandler.update();

  if (typeof overrideDragX === 'number') {
    sliderInputX = overrideDragX;
  } else if (eventHandler.isPointerDown) {
    sliderInputX = eventHandler.currentMouseX;
  }

  slider.setDragX(sliderInputX);
  slider.update(deltaTime);
}

function render(timestamp: number) {
  if (isPresentingXr) {
    return;
  }

  frameCount++;
  camera.jitter();
  const deltaTime = getDeltaSeconds(timestamp);
  updateSimulation(deltaTime);

  const currentFrame = frameCount % 2;

  getRayMarchPipeline(RAY_MARCH_FORMAT)
    .withColorAttachment({
      view: root.unwrap(textures[currentFrame].sampled),
      loadOp: 'clear',
      storeOp: 'store',
    })
    .with(bindGroups.rayMarch)
    .draw(3);

  taaResolver.resolve(
    textures[currentFrame].sampled,
    frameCount,
    currentFrame,
  );

  renderPipeline
    .withColorAttachment({
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      storeOp: 'store',
    })
    .with(bindGroups.render[currentFrame])
    .draw(3);

  if (!isPresentingXr) {
    requestAnimationFrame(render);
  }
}

function handleResize() {
  [width, height] = [
    canvas.width * qualityScale,
    canvas.height * qualityScale,
  ];
  camera.updateProjection(Math.PI / 4, width, height);
  textures = createTextures(root, width, height);
  backgroundTexture = createBackgroundTexture(root, width, height);
  taaResolver.resize(width, height);
  frameCount = 0;

  bindGroups = createBindGroups();
}

async function setupXrButtons() {
  if (!navigator.xr) {
    setXrButtonState(xrVrButton, 'WebXR unavailable', true);
    setXrButtonState(xrArButton, 'WebXR unavailable', true);
    return;
  }

  if (!navigator.gpu) {
    setXrButtonState(xrVrButton, 'WebGPU unavailable', true);
    setXrButtonState(xrArButton, 'WebGPU unavailable', true);
    return;
  }

  if (!('XRGPUBinding' in window)) {
    setXrButtonState(xrVrButton, 'XR/WebGPU unsupported', true);
    setXrButtonState(xrArButton, 'XR/WebGPU unsupported', true);
    return;
  }

  const [vrSupported, arSupported] = await Promise.all([
    navigator.xr.isSessionSupported('immersive-vr'),
    navigator.xr.isSessionSupported('immersive-ar'),
  ]);
  xrVrSupported = vrSupported;
  xrArSupported = arSupported;

  configureXrButton(xrVrButton, vrSupported, 'VR', 'immersive-vr');
  configureXrButton(xrArButton, arSupported, 'AR', 'immersive-ar');
}

function setXrButtonState(
  button: HTMLButtonElement | null,
  text: string,
  disabled: boolean,
) {
  if (!button) return;
  button.textContent = text;
  button.disabled = disabled;
}

function configureXrButton(
  button: HTMLButtonElement | null,
  supported: boolean,
  label: string,
  mode: XRSessionMode,
) {
  if (!button) return;
  if (!supported) {
    setXrButtonState(button, `${label} not supported`, true);
    return;
  }
  setXrButtonState(button, `Enter ${label}`, false);
  button.onclick = () => {
    if (xrSession && activeXrMode === mode) {
      xrSession.end();
    } else if (!xrSession) {
      void startXrSession(mode);
    }
  };
}

async function startXrSession(mode: XRSessionMode) {
  if (!navigator.xr) {
    return;
  }

  const targetButton = mode === 'immersive-vr' ? xrVrButton : xrArButton;
  const otherButton = mode === 'immersive-vr' ? xrArButton : xrVrButton;

  if (targetButton) {
    setXrButtonState(
      targetButton,
      mode === 'immersive-vr' ? 'Starting VR…' : 'Starting AR…',
      true,
    );
  }
  if (otherButton) {
    otherButton.disabled = true;
  }

  try {
    const session = await navigator.xr.requestSession(mode, {
      requiredFeatures: ['webgpu'],
      optionalFeatures: ['hand-tracking'],
    });
    xrSession = session;
    isPresentingXr = true;
    activeXrMode = mode;
    interactionOnlyUniform.write(d.u32(mode === 'immersive-ar' ? 1 : 0));

    session.addEventListener('end', onXrSessionEnded);

    xrGpuBinding = new XRGPUBinding(session, root.device);
    const preferredColorFormat =
      xrGpuBinding.getPreferredColorFormat() ?? presentationFormat;
    xrColorFormat = preferredColorFormat;
    const projectionLayer = xrGpuBinding.createProjectionLayer({
      colorFormat: preferredColorFormat,
    });
    xrProjectionLayer = projectionLayer;
    session.updateRenderState({ layers: [projectionLayer] });

    xrRefSpace = await session.requestReferenceSpace('local');

    if (targetButton) {
      setXrButtonState(
        targetButton,
        mode === 'immersive-vr' ? 'Exit VR' : 'Exit AR',
        false,
      );
    }
    if (otherButton) {
      otherButton.disabled = true;
    }

    session.requestAnimationFrame(onXrFrame);
  } catch (error) {
    console.error('Failed to start XR session', error);
    xrSession = null;
    xrRefSpace = null;
    xrGpuBinding = null;
    xrProjectionLayer = null;
    isPresentingXr = false;
    activeXrMode = null;
    interactionOnlyUniform.write(d.u32(0));
    xrHandInteraction.dragX = null;
    xrHandInteraction.activeHand = null;
    xrHandInteraction.lastSeenTime = 0;
    xrHandInteraction.isPinching = false;
    commitHandJointUniform(0);
    sliderInputX = eventHandler.currentMouseX;
    if (targetButton) {
      const supported = mode === 'immersive-vr' ? xrVrSupported : xrArSupported;
      const label = mode === 'immersive-vr' ? 'VR' : 'AR';
      setXrButtonState(
        targetButton,
        supported ? `Enter ${label}` : `${label} not supported`,
        !supported,
      );
    }
    if (otherButton) {
      const supported = mode === 'immersive-vr' ? xrArSupported : xrVrSupported;
      const label = mode === 'immersive-vr' ? 'AR' : 'VR';
      setXrButtonState(
        otherButton,
        supported ? `Enter ${label}` : `${label} not supported`,
        !supported,
      );
    }
    requestAnimationFrame(render);
  }
}

function onXrSessionEnded() {
  xrSession = null;
  xrRefSpace = null;
  xrGpuBinding = null;
  xrProjectionLayer = null;
  isPresentingXr = false;
   activeXrMode = null;
   interactionOnlyUniform.write(d.u32(0));
  xrHandInteraction.dragX = null;
  xrHandInteraction.activeHand = null;
  xrHandInteraction.lastSeenTime = 0;
  xrHandInteraction.isPinching = false;
  commitHandJointUniform(0);
  sliderInputX = eventHandler.currentMouseX;
  setXrButtonState(
    xrVrButton,
    xrVrSupported ? 'Enter VR' : 'VR not supported',
    !xrVrSupported,
  );
  setXrButtonState(
    xrArButton,
    xrArSupported ? 'Enter AR' : 'AR not supported',
    !xrArSupported,
  );

  camera.updateView(baseCameraPosition, baseCameraTarget, baseCameraUp);
  camera.updateProjection(Math.PI / 4, width, height);
  lastTimeStamp = performance.now();
  requestAnimationFrame(render);
}

function onXrFrame(time: DOMHighResTimeStamp, frame: XRFrame) {
  if (!xrSession || !xrRefSpace || !xrGpuBinding || !xrProjectionLayer) {
    return;
  }

  xrSession.requestAnimationFrame(onXrFrame);

  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose) {
    return;
  }

  const deltaTime = getDeltaSeconds(time);
  const handOverride = updateXrHandInteraction(frame, time);
  updateSimulation(deltaTime, handOverride);

  const xrFormat = xrColorFormat ?? RAY_MARCH_FORMAT;
  const xrPipeline = getRayMarchPipeline(xrFormat);

  for (const view of pose.views) {
    updateCameraFromXrView(view);
    const subImage = xrGpuBinding.getViewSubImage(xrProjectionLayer, view);
    const colorView = subImage.colorTexture.createView(
      subImage.getViewDescriptor(),
    );

    xrPipeline
      .withColorAttachment({
        view: colorView,
        loadOp: 'clear',
        storeOp: 'store',
      })
      .with(bindGroups.rayMarch)
      .draw(3);
  }
}

function updateCameraFromXrView(view: XRView) {
  const baseView = view.transform.inverse.matrix;
  const baseViewInv = view.transform.matrix;

  const scaledView = m.mat4.mul(
    baseView,
    xrSceneTransform,
    m.mat4.identity(),
  );
  const scaledViewInv = m.mat4.mul(
    xrSceneTransformInv,
    baseViewInv,
    m.mat4.identity(),
  );

  const projMatrix = mat4FromArrayLike(view.projectionMatrix);
  const invertedProj = m.mat4.invert(
    view.projectionMatrix,
    m.mat4.identity(),
  );
  const projInvMatrix = invertedProj
    ? mat4FromArrayLike(invertedProj)
    : identityMat4();

  cameraUniform.write({
    view: mat4FromArrayLike(scaledView),
    viewInv: mat4FromArrayLike(scaledViewInv),
    proj: projMatrix,
    projInv: projInvMatrix,
  });
}

function updateXrHandInteraction(
  frame: XRFrame,
  time: DOMHighResTimeStamp,
): number | null {
  if (!xrSession || !xrRefSpace) {
    commitHandJointUniform(0);
    if (time - xrHandInteraction.lastSeenTime > XR_HAND_RELEASE_MS) {
      releaseHandInteraction();
    }
    return xrHandInteraction.dragX;
  }

  let jointWriteIndex = 0;
  let bestCandidate: XrHandCandidate | null = null;
  let activeCandidate: XrHandCandidate | null = null;
  for (const inputSource of xrSession.inputSources) {
    const candidates: XrHandCandidate[] = [];
    if (inputSource.hand) {
      const { nextIndex, candidate: handCandidate } = collectHandSourceData(
        frame,
        inputSource,
        jointWriteIndex,
      );
      jointWriteIndex = nextIndex;
      if (handCandidate) {
        candidates.push(handCandidate);
      }
    }
    if (inputSource.gamepad) {
      const controllerCandidate = collectControllerSourceData(
        frame,
        inputSource,
      );
      if (controllerCandidate) {
        candidates.push(controllerCandidate);
      }
    }

    for (const candidate of candidates) {
      if (!bestCandidate || candidate.score < bestCandidate.score) {
        bestCandidate = candidate;
      }
      if (candidate.handedness === xrHandInteraction.activeHand) {
        if (
          !activeCandidate ||
          candidate.score < activeCandidate.score
        ) {
          activeCandidate = candidate;
        }
      }
    }
  }

  commitHandJointUniform(jointWriteIndex);

  const isCurrentlyActive = xrHandInteraction.activeHand !== null &&
    xrHandInteraction.dragX !== null;
  const candidate = isCurrentlyActive
    ? activeCandidate
    : bestCandidate;

  if (isCurrentlyActive && !candidate) {
    if (time - xrHandInteraction.lastSeenTime > XR_HAND_RELEASE_MS) {
      releaseHandInteraction();
    }
    logXrDebug('Active XR drag lost candidate; waiting for release timeout');
    return xrHandInteraction.dragX;
  }

  if (!candidate) {
    if (time - xrHandInteraction.lastSeenTime > XR_HAND_RELEASE_MS) {
      releaseHandInteraction();
    }
    logXrDebug('No XR candidates available this frame');
    return xrHandInteraction.dragX;
  }

  if (!candidate.isPinching) {
    if (isCurrentlyActive) {
      releaseHandInteraction();
    }
    logXrDebug('XR candidate not interacting because pinch/button not active', {
      handedness: candidate.handedness,
      withinBounds: candidate.withinBounds,
      tipDistance: candidate.tipDistance,
    });
    return xrHandInteraction.dragX;
  }

  xrHandInteraction.lastSeenTime = time;

  if (!isCurrentlyActive) {
    if (
      !candidate.withinBounds ||
      candidate.tipDistance > XR_HAND_CAPTURE_RADIUS
    ) {
      logXrDebug('XR candidate skipped (outside capture range)', {
        handedness: candidate.handedness,
        withinBounds: candidate.withinBounds,
        tipDistance: candidate.tipDistance,
      });
      return xrHandInteraction.dragX;
    }
    xrHandInteraction.dragX = candidate.dragX;
    xrHandInteraction.activeHand = candidate.handedness;
    xrHandInteraction.isPinching = true;
    logXrDebug('XR candidate captured slider', {
      handedness: candidate.handedness,
      dragX: candidate.dragX,
      tipDistance: candidate.tipDistance,
    });
    return xrHandInteraction.dragX;
  }

  const sliderTipX = slider.tipX;
  const closeness = Math.max(
    0,
    1 - candidate.tipDistance / XR_HAND_CAPTURE_RADIUS,
  );
  const pullStrength = XR_HAND_APPROACH_MIN +
    closeness * (XR_HAND_APPROACH_MAX - XR_HAND_APPROACH_MIN);
  const desired = sliderTipX +
    (candidate.dragX - sliderTipX) * pullStrength;
  const previous = xrHandInteraction.dragX ?? sliderTipX;
  const smoothed = previous +
    (desired - previous) * XR_HAND_SMOOTHING;

  xrHandInteraction.dragX = smoothed;
  xrHandInteraction.activeHand = candidate.handedness;
  xrHandInteraction.isPinching = true;
  if (XR_DEBUG_LOGS && Math.abs(smoothed - previous) > 0.01) {
    logXrDebug('XR drag updated', {
      handedness: candidate.handedness,
      dragX: smoothed,
      desired,
      tipDistance: candidate.tipDistance,
    });
  }
  return xrHandInteraction.dragX;
}

function releaseHandInteraction() {
  xrHandInteraction.dragX = null;
  xrHandInteraction.activeHand = null;
  xrHandInteraction.isPinching = false;
  logXrDebug('XR interaction released');
}

type Vec3Tuple = [number, number, number];

function collectHandSourceData(
  frame: XRFrame,
  inputSource: XRInputSource,
  writeIndex: number,
): { nextIndex: number; candidate: XrHandCandidate | null } {
  if (
    !xrRefSpace ||
    !inputSource.hand ||
    inputSource.handedness === 'none' ||
    !frame.getJointPose
  ) {
    return { nextIndex: writeIndex, candidate: null };
  }

  let candidatePoint: Vec3Tuple | null = null;
  let thumbTipPoint: Vec3Tuple | null = null;
  const handedness = inputSource.handedness;

  for (const jointName of XR_HAND_JOINTS) {
    if (writeIndex >= MAX_HAND_JOINTS) {
      break;
    }
    const jointSpace = inputSource.hand.get(jointName);
    if (!jointSpace) {
      continue;
    }
    const jointPose = frame.getJointPose(jointSpace, xrRefSpace);
    if (!jointPose) {
      continue;
    }
    const point = transformPoint(
      xrSceneTransformInv,
      jointPose.transform.position,
    );
    const radius = jointPose.radius ?? 0.008;
    writeHandJointData(writeIndex, point, radius, handedness);
    if (jointName === XR_INTERACTION_JOINT) {
      candidatePoint = point;
    }
    if (jointName === 'thumb-tip') {
      thumbTipPoint = point;
    }
    writeIndex++;
  }

  if (!candidatePoint) {
    return { nextIndex: writeIndex, candidate: null };
  }

  let pinchDistance = Infinity;
  if (thumbTipPoint) {
    pinchDistance = distanceBetweenPoints(candidatePoint, thumbTipPoint);
  }
  const isPinching = pinchDistance < XR_PINCH_MAX_DISTANCE;
  const actionPressed = inputSource.gamepad
    ? isControllerActionPressed(inputSource)
    : false;

  const candidate = createHandCandidateFromPoint(
    candidatePoint,
    handedness,
    isPinching || actionPressed,
  );
  logXrDebug('Hand candidate', {
    handedness,
    isPinching: candidate?.isPinching,
    pointer: candidatePoint,
    withinBounds: candidate?.withinBounds,
    tipDistance: candidate?.tipDistance,
    actionPressed,
  });

  return { nextIndex: writeIndex, candidate };
}

function collectControllerSourceData(
  frame: XRFrame,
  inputSource: XRInputSource,
): XrHandCandidate | null {
  if (
    !xrRefSpace ||
    inputSource.handedness === 'none'
  ) {
    return null;
  }

  const point = getControllerInteractionPoint(frame, inputSource);
  if (!point) {
    logXrDebug('Controller interaction point unavailable', {
      handedness: inputSource.handedness,
    });
    return null;
  }

  const isPressed = isControllerActionPressed(inputSource);
  logXrDebug('Controller candidate', {
    handedness: inputSource.handedness,
    isPressed,
    point,
  });

  return createHandCandidateFromPoint(
    point,
    inputSource.handedness,
    isPressed,
    XR_CONTROLLER_Z_THRESHOLD,
  );
}

function writeHandJointData(
  index: number,
  point: Vec3Tuple,
  radius: number,
  handedness: XRHandedness,
) {
  if (index >= MAX_HAND_JOINTS) {
    return;
  }
  const target = handJointData[index];
  target[0] = point[0];
  target[1] = point[1];
  target[2] = point[2];
  const visualRadius = Math.max(radius, 0.006) *
    XR_SCENE_SCALE *
    XR_HAND_VISUAL_SCALE;
  target[3] = visualRadius * (handedness === 'right' ? 1 : -1);
}

function commitHandJointUniform(count: number) {
  for (let i = count; i < MAX_HAND_JOINTS; i++) {
    handJointData[i][3] = 0;
  }
  handJointsUniform.write({
    joints: handJointData,
    count: d.u32(count),
    padding: handJointPadding,
  });
}

function createHandCandidateFromPoint(
  point: Vec3Tuple,
  handedness: XRHandedness,
  isPinching: boolean,
  zThreshold = XR_HAND_Z_THRESHOLD,
): XrHandCandidate {
  const withinBounds = pointWithinSliderBounds(point, zThreshold);
  const [top, , bottom] = bezierBbox;
  const sliderMidY = (top + bottom) * 0.5;
  const verticalRange = Math.max(0.001, top - bottom);
  const normalizedY = Math.abs(point[1] - sliderMidY) /
    (verticalRange * 0.5 + XR_HAND_Y_MARGIN);
  const normalizedZ = Math.abs(point[2]) / zThreshold;
  const sliderTipX = slider.tipX;
  const tipDistance = Math.abs(point[0] - sliderTipX);
  const score = tipDistance + normalizedY * 0.35 + normalizedZ * 0.5;

  return {
    dragX: point[0],
    score,
    handedness,
    tipDistance,
    withinBounds,
    isPinching,
  };
}

function pointWithinSliderBounds(
  [x, y, z]: Vec3Tuple,
  zThreshold = XR_HAND_Z_THRESHOLD,
) {
  const [top, right, bottom, left] = bezierBbox;
  return (
    x >= left - XR_HAND_X_MARGIN &&
    x <= right + XR_HAND_X_MARGIN &&
    y >= bottom - XR_HAND_Y_MARGIN &&
    y <= top + XR_HAND_Y_MARGIN &&
    Math.abs(z) <= zThreshold
  );
}

function distanceBetweenPoints(a: Vec3Tuple, b: Vec3Tuple) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function getControllerInteractionPoint(
  frame: XRFrame,
  inputSource: XRInputSource,
): Vec3Tuple | null {
  if (!xrRefSpace) {
    return null;
  }

  if (inputSource.gripSpace) {
    const gripPose = frame.getPose(inputSource.gripSpace, xrRefSpace);
    if (gripPose) {
      const gripPoint = transformPoint(
        xrSceneTransformInv,
        gripPose.transform.position,
      );
      if (Math.abs(gripPoint[2]) <= XR_CONTROLLER_Z_THRESHOLD * 2) {
        logXrDebug('Using controller grip pose for interaction', {
          handedness: inputSource.handedness,
          point: gripPoint,
        });
        return gripPoint;
      }
    }
  }

  if (!inputSource.targetRaySpace) {
    return null;
  }

  const rayPose = frame.getPose(inputSource.targetRaySpace, xrRefSpace);
  if (!rayPose) {
    return null;
  }

  const origin = transformPoint(
    xrSceneTransformInv,
    rayPose.transform.position,
  );
  const direction = transformDirection(rayPose.transform.orientation);
  if (!direction) {
    return null;
  }

  const zComponent = direction[2];
  if (Math.abs(zComponent) < 1e-4) {
    return null;
  }

  const t = -origin[2] / zComponent;
  if (!Number.isFinite(t)) {
    return null;
  }

  const intersection: Vec3Tuple = [
    origin[0] + direction[0] * t,
    origin[1] + direction[1] * t,
    0,
  ];
  logXrDebug('Using controller target-ray intersection', {
    handedness: inputSource.handedness,
    origin,
    direction,
    point: intersection,
  });
  return intersection;
}

function isControllerActionPressed(inputSource: XRInputSource) {
  const gamepad = inputSource.gamepad;
  if (!gamepad) {
    return false;
  }

  for (const index of XR_CONTROLLER_BUTTON_INDICES) {
    const button = gamepad.buttons[index];
    if (button && (button.pressed || button.value > XR_CONTROLLER_ACTION_THRESHOLD)) {
      return true;
    }
  }

  return gamepad.buttons.some((button) =>
    button.pressed && button.value > XR_CONTROLLER_ACTION_THRESHOLD
  );
}

function transformDirection(
  orientation: DOMPointReadOnly | undefined,
): Vec3Tuple | null {
  if (!orientation) {
    return null;
  }
  return rotateVectorByQuaternion([0, 0, -1], orientation);
}

function rotateVectorByQuaternion(
  vector: Vec3Tuple,
  orientation: DOMPointReadOnly,
): Vec3Tuple {
  const { x, y, z, w } = orientation;
  const uvx = y * vector[2] - z * vector[1];
  const uvy = z * vector[0] - x * vector[2];
  const uvz = x * vector[1] - y * vector[0];

  const uuvx = y * uvz - z * uvy;
  const uuvy = z * uvx - x * uvz;
  const uuvz = x * uvy - y * uvx;

  const uvScale = 2 * w;
  const uuvScale = 2;

  return [
    vector[0] + uvx * uvScale + uuvx * uuvScale,
    vector[1] + uvy * uvScale + uuvy * uuvScale,
    vector[2] + uvz * uvScale + uuvz * uuvScale,
  ];
}

function mat4FromArrayLike(source: ArrayLike<number>) {
  const mat = d.mat4x4f();
  for (let i = 0; i < 16; i++) {
    mat[i] = source[i] ?? 0;
  }
  return mat;
}

function transformPoint(mat: m.Mat4, point: DOMPointReadOnly): Vec3Tuple {
  const x = point.x;
  const y = point.y;
  const z = point.z;
  const w = mat[3] * x + mat[7] * y + mat[11] * z + mat[15];
  const invW = w && Number.isFinite(w) ? 1 / w : 1;

  return [
    (mat[0] * x + mat[4] * y + mat[8] * z + mat[12]) * invW,
    (mat[1] * x + mat[5] * y + mat[9] * z + mat[13]) * invW,
    (mat[2] * x + mat[6] * y + mat[10] * z + mat[14]) * invW,
  ];
}

function identityMat4() {
  const mat = d.mat4x4f();
  mat[0] = 1;
  mat[5] = 1;
  mat[10] = 1;
  mat[15] = 1;
  return mat;
}

const resizeObserver = new ResizeObserver(() => {
  handleResize();
});
resizeObserver.observe(canvas);

void setupXrButtons();
requestAnimationFrame(render);

// #region Example controls and cleanup

async function autoSetQuaility() {
  if (!hasTimestampQuery) {
    return 0.5;
  }

  const targetFrameTime = 5;
  const tolerance = 2.0;
  let resolutionScale = 0.3;
  let lastTimeMs = 0;

  const measurePipeline = getRayMarchPipeline(RAY_MARCH_FORMAT)
    .withPerformanceCallback((start, end) => {
      lastTimeMs = Number(end - start) / 1e6;
    });

  for (let i = 0; i < 8; i++) {
    const testTexture = root['~unstable'].createTexture({
      size: [canvas.width * resolutionScale, canvas.height * resolutionScale],
      format: 'rgba8unorm',
    }).$usage('render');

    measurePipeline
      .withColorAttachment({
        view: root.unwrap(testTexture).createView(),
        loadOp: 'clear',
        storeOp: 'store',
      })
      .with(
        root.createBindGroup(rayMarchLayout, {
          backgroundTexture: backgroundTexture.sampled,
        }),
      )
      .draw(3);

    await root.device.queue.onSubmittedWorkDone();
    testTexture.destroy();

    if (Math.abs(lastTimeMs - targetFrameTime) < tolerance) {
      break;
    }

    const adjustment = lastTimeMs > targetFrameTime ? -0.1 : 0.1;
    resolutionScale = Math.max(
      0.3,
      Math.min(1.0, resolutionScale + adjustment),
    );
  }

  console.log(`Auto-selected quality scale: ${resolutionScale.toFixed(2)}`);
  return resolutionScale;
}

export const controls = {
  'Quality': {
    initial: 'Auto',
    options: [
      'Auto',
      'Very Low',
      'Low',
      'Medium',
      'High',
      'Ultra',
    ],
    onSelectChange: (value: string) => {
      if (value === 'Auto') {
        autoSetQuaility().then((scale) => {
          qualityScale = scale;
          handleResize();
        });
        return;
      }

      const qualityMap: { [key: string]: number } = {
        'Very Low': 0.3,
        'Low': 0.5,
        'Medium': 0.7,
        'High': 0.85,
        'Ultra': 1.0,
      };

      qualityScale = qualityMap[value] || 0.5;
      handleResize();
    },
  },
  'Light dir': {
    initial: 0,
    min: 0,
    max: 1,
    step: 0.01,
    onSliderChange: (v: number) => {
      const dir1 = std.normalize(d.vec3f(0.18, -0.30, 0.64));
      const dir2 = std.normalize(d.vec3f(-0.5, -0.14, -0.8));
      const finalDir = std.normalize(std.mix(dir1, dir2, v));
      lightUniform.writePartial({
        direction: finalDir,
      });
    },
  },
  'Jelly Color': {
    initial: [1.0, 0.45, 0.075],
    onColorChange: (c: [number, number, number]) => {
      jellyColorUniform.write(d.vec4f(...c, 1.0));
    },
  },
  'Blur': {
    initial: false,
    onToggleChange: (v: boolean) => {
      blurEnabledUniform.write(d.u32(v));
    },
  },
};

export function onCleanup() {
  resizeObserver.disconnect();
  root.destroy();
}

// #endregion
