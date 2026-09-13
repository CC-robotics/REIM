import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Users/Kyrie Zhang/Desktop/REIM";
const SKILL_DIR = "C:/Users/Kyrie Zhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.903.11726/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-build", "figure1-overview");
const FINAL_PPTX = path.join(workspaceDir, "paper_assets", "Figure1_overview_editable_v6.pptx");
const RUNTIME_PYTHON = "C:/Users/Kyrie Zhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const RUNTIME_NODE_MODULES = "C:/Users/Kyrie Zhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const RUNTIME_BIN_DIR = "C:/Users/Kyrie Zhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override";
process.env.RUNTIME_NODE_MODULES ??= RUNTIME_NODE_MODULES;
process.env.RUNTIME_BIN_DIR ??= RUNTIME_BIN_DIR;

const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
const fontFamily = resolvePresentationFont({ fontFamily: "Times New Roman" });

const W = 1280;
const H = 500;
const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const slide = presentation.slides.add();
slide.background.fill = "#FFFFFF";

const C = {
  ink: "#242424",
  muted: "#5F6872",
  faint: "#8B929A",
  line: "#C9CDD2",
  lightLine: "#DFE2E5",
  problem: "#9B3E3E",
  problemFill: "#FCF7F6",
  blue: "#3E6FA3",
  blueFill: "#EEF4F9",
  amber: "#B97823",
  amberFill: "#FBF3E6",
  green: "#3E8764",
  greenFill: "#EDF6F1",
  neutralFill: "#F6F7F7",
};

function addShape({ x, y, w, h, geometry = "rect", fill = "none", line = "none", radius = 0, name }) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line === "none" ? { fill: "none", width: 0 } : line,
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function addText(text, x, y, w, h, {
  size = 20,
  color = C.ink,
  bold = false,
  italic = false,
  align = "left",
  valign = "middle",
  fill = "none",
  line = "none",
  radius = 0,
  insets = { top: 0, right: 0, bottom: 0, left: 0 },
  name,
} = {}) {
  const box = addShape({ x, y, w, h, geometry: "textbox", fill, line, radius, name });
  box.text = text;
  box.text.style = {
    typeface: fontFamily,
    fontSize: size,
    bold,
    italic,
    color,
    alignment: align,
    verticalAlignment: valign,
    autoFit: "shrinkText",
    wrap: "square",
    insets,
  };
  return box;
}

function addModule(text, x, y, w, h, color, fill, { size = 20, name } = {}) {
  return addText(text, x, y, w, h, {
    size,
    color,
    bold: true,
    align: "center",
    valign: "middle",
    fill,
    line: { style: "solid", fill: color, width: 1.4 },
    radius: 9,
    insets: { top: 5, right: 6, bottom: 5, left: 6 },
    name,
  });
}

function connect(from, to, color, {
  fromSide = "right",
  toSide = "left",
  kind = "straight",
  width = 2.1,
  dashed = false,
} = {}) {
  const connector = slide.shapes.connect(from, to, {
    kind,
    fromSide,
    toSide,
    line: { style: dashed ? "dashed" : "solid", fill: color, width },
    tail: { type: "triangle", width: "sm", length: "sm" },
  });
  connector.bringToFront();
  return connector;
}

async function addImageFrame(file, x, y, w, h, borderColor, alt, crop = undefined) {
  addShape({
    x: x - 2,
    y: y - 2,
    w: w + 4,
    h: h + 4,
    fill: "#FFFFFF",
    line: { style: "solid", fill: borderColor, width: 1.6 },
    radius: 5,
  });
  const bytes = new Uint8Array(await fs.readFile(file));
  return slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: "cover",
    position: { left: x, top: y, width: w, height: h },
    geometry: "rect",
    ...(crop ? { crop } : {}),
  });
}

function addPanel(x, y, w, h, fill, title, titleColor, panelLetter) {
  addShape({
    x, y, w, h,
    fill,
    line: { style: "solid", fill: C.line, width: 1.1 },
    radius: 14,
    name: `${title} panel`,
  });
  addText(`(${panelLetter})`, x + 14, y + 9, 34, 34, {
    size: 23, color: C.ink, bold: true, align: "center",
  });
  addText(title, x + 49, y + 9, w - 63, 34, {
    size: 29, color: titleColor, bold: true,
  });
}

const frames = path.join(workspaceDir, "results", "figures", "recovery_operation_sequence_frames");

// Three broad sections, kept deliberately flat and sparse.
addPanel(18, 10, 288, 480, "#FEFDFC", "Problem", C.problem, "a");
addPanel(318, 10, 620, 480, "#FCFDFE", "REIM framework", C.blue, "b");
addPanel(950, 10, 312, 480, "#FCFEFD", "Closed-loop result", C.green, "c");

// -----------------------------------------------------------------------------
// (a) Motivation: perturbations invalidate a fixed nominal chunk.
// -----------------------------------------------------------------------------
const disturbPath = path.join(frames, "02_act_disturbance_seed8300042_t003.png");
const failurePath = path.join(frames, "04_act_failure_seed8300042_t200.png");
await addImageFrame(disturbPath, 39, 72, 112, 112, C.amber,
  "Object displaced during the paired Meta-World PickPlace rollout");
await addImageFrame(failurePath, 174, 72, 112, 112, C.problem,
  "ACT failure at the 200-step timeout");

const topArrowStart = addShape({ x: 154, y: 113, w: 8, h: 8, geometry: "ellipse", fill: "none" });
const topArrowEnd = addShape({ x: 165, y: 113, w: 8, h: 8, geometry: "ellipse", fill: "none" });
connect(topArrowStart, topArrowEnd, C.problem, { width: 1.8 });
topArrowStart.line = { fill: "none", width: 0 };
topArrowEnd.line = { fill: "none", width: 0 };

addText("object shifted", 38, 188, 114, 25, { size: 19, color: C.amber, bold: true, align: "center" });
addText("ACT timeout", 173, 188, 114, 25, { size: 19, color: C.problem, bold: true, align: "center" });

addText("Predicted action chunk", 40, 224, 244, 24, { size: 20, color: C.ink, bold: true });
for (let i = 0; i < 8; i += 1) {
  addShape({
    x: 42 + i * 30,
    y: 257,
    w: 23,
    h: 28,
    fill: i < 3 ? C.blue : "#9CB4CA",
    line: { style: "solid", fill: "#FFFFFF", width: 0.7 },
    radius: 4,
    name: `Action chunk step ${i + 1}`,
  });
}
addText("!", 126, 239, 28, 28, {
  size: 22, color: "#FFFFFF", bold: true, align: "center",
  fill: C.problem, line: { style: "solid", fill: C.problem, width: 0.8 }, radius: 14,
});
addText("disturbance", 102, 289, 78, 20, { size: 17, color: C.problem, italic: true, align: "center" });

addText("The state leaves the demonstration distribution", 39, 326, 248, 49, {
  size: 21, color: C.ink, bold: true, align: "center",
  fill: C.problemFill,
  line: { style: "solid", fill: "#E5C9C5", width: 1 },
  radius: 8,
  insets: { top: 4, right: 7, bottom: 4, left: 7 },
});
addText("Stale action chunk\nNo recovery available", 43, 389, 240, 56, {
  size: 20, color: C.problem, align: "center",
});

// -----------------------------------------------------------------------------
// (b) Framework: online arbitration and trigger-aligned supervised recovery.
// -----------------------------------------------------------------------------
const statePath = path.join(frames, "01_act_initial_seed8300042_t000.png");
await addImageFrame(statePath, 342, 78, 116, 116, C.muted,
  "Sawyer robot state in the simulated PickPlace environment");
addText("robot state  sₜ", 340, 198, 120, 26, { size: 19, color: C.muted, bold: true, align: "center" });

const act = addModule("ACT\nnominal chunk", 496, 72, 140, 62, C.blue, C.blueFill, { size: 20, name: "ACT nominal task policy" });
const risk = addModule("LSTM\nrisk pₜ", 496, 164, 140, 62, C.amber, C.amberFill, { size: 20, name: "Causal failure-risk monitor" });
const stateAnchorTop = addShape({ x: 463, y: 103, w: 8, h: 8, geometry: "ellipse", fill: "none" });
const stateAnchorBottom = addShape({ x: 463, y: 195, w: 8, h: 8, geometry: "ellipse", fill: "none" });
stateAnchorTop.line = { fill: "none", width: 0 };
stateAnchorBottom.line = { fill: "none", width: 0 };
connect(stateAnchorTop, act, C.blue, { width: 2.0 });
connect(stateAnchorBottom, risk, C.amber, { width: 2.0 });

const gate = addModule("risk gate\npₜ ≥ 0.20?", 680, 102, 106, 96, C.problem, "#FFF5F4", { size: 21, name: "Frozen deployment risk gate" });
connect(act, gate, C.blue, { fromSide: "right", toSide: "left", kind: "elbow", width: 2.0 });
connect(risk, gate, C.amber, { fromSide: "right", toSide: "left", kind: "elbow", width: 2.0 });

const nominal = addModule("ACT action", 812, 72, 90, 62, C.blue, C.blueFill, { size: 20, name: "Nominal action" });
const recovery = addModule("recovery\npolicy", 812, 164, 90, 62, C.green, C.greenFill, { size: 18, name: "Persistent recovery policy" });
connect(gate, nominal, C.blue, { fromSide: "right", toSide: "left", kind: "elbow", width: 2.0 });
connect(gate, recovery, C.green, { fromSide: "right", toSide: "left", kind: "elbow", width: 2.0 });
const lowRiskLabel = addText("low risk", 795, 49, 100, 21, { size: 18, color: C.blue, italic: true, align: "center", fill: "#FCFDFE" });
const highRiskLabel = addText("high risk", 789, 140, 86, 21, { size: 18, color: C.green, italic: true, align: "center", fill: "#FCFDFE" });
lowRiskLabel.bringToFront();
highRiskLabel.bringToFront();

const actionNode = addText("aₜ", 858, 240, 40, 32, {
  size: 22, color: "#FFFFFF", bold: true, italic: true, align: "center",
  fill: C.ink, line: { style: "solid", fill: C.ink, width: 1 }, radius: 16,
  name: "Executed action",
});
addShape({ x: 900, y: 102, w: 18, h: 2.2, fill: C.blue, name: "Nominal action route top" });
addShape({ x: 916, y: 102, w: 2.2, h: 154, fill: C.blue, name: "Nominal action route side" });
addShape({ x: 895, y: 249, w: 16, h: 14, geometry: "leftArrow", fill: C.blue, name: "Nominal action route arrow" });
connect(recovery, actionNode, C.green, { fromSide: "bottom", toSide: "top", kind: "elbow", width: 2.0 });
connect(actionNode, stateAnchorTop, C.muted, { fromSide: "left", toSide: "top", kind: "elbow4", width: 1.6, dashed: true });
const closedLoopLabel = addText("closed-loop next state  sₜ₊₁", 617, 243, 200, 22, { size: 17, color: C.muted, italic: true, align: "center", fill: "#FCFDFE" });
closedLoopLabel.bringToFront();
[act, risk, gate, nominal, recovery, actionNode].forEach((shape) => shape.bringToFront());
lowRiskLabel.bringToFront();
highRiskLabel.bringToFront();
closedLoopLabel.bringToFront();

addShape({ x: 338, y: 283, w: 580, h: 1.2, fill: C.lightLine, line: { fill: "none", width: 0 } });
addText("Trigger-aligned recovery training", 341, 291, 270, 28, { size: 23, color: C.ink, bold: true });
addText("42,386 train  ·  8,212 disjoint val. pairs", 610, 294, 306, 24, { size: 17, color: C.muted, align: "right" });

const triggerPath = path.join(frames, "03_trigger_seed5100042_t009.png");
const expertPath = path.join(frames, "04_relift_seed5100042_t040.png");
await addImageFrame(triggerPath, 349, 337, 83, 83, C.amber,
  "Online trigger state used to collect recovery data");
await addImageFrame(expertPath, 502, 337, 83, 83, C.green,
  "Expert continuation from the exact trigger state");
addText("trigger state", 343, 425, 96, 24, { size: 18, color: C.amber, bold: true, align: "center" });
addText("expert continuation", 486, 425, 116, 24, { size: 18, color: C.green, bold: true, align: "center" });

const triggerAnchor = addShape({ x: 438, y: 374, w: 8, h: 8, geometry: "ellipse", fill: "none" });
const expertAnchor = addShape({ x: 488, y: 374, w: 8, h: 8, geometry: "ellipse", fill: "none" });
triggerAnchor.line = { fill: "none", width: 0 };
expertAnchor.line = { fill: "none", width: 0 };
connect(triggerAnchor, expertAnchor, C.muted, { width: 1.8 });

const loss = addModule("Smooth-L1\nsupervision", 650, 346, 112, 66, C.green, C.greenFill, { size: 19, name: "Smooth-L1 recovery imitation objective" });
const learned = addModule("recovery\nactor", 803, 346, 112, 66, C.green, "#E3F1E9", { size: 20, name: "Learned recovery actor" });
const expertOut = addShape({ x: 589, y: 374, w: 8, h: 8, geometry: "ellipse", fill: "none" });
expertOut.line = { fill: "none", width: 0 };
connect(expertOut, loss, C.green, { width: 1.9 });
connect(loss, learned, C.green, { width: 1.9 });
addText("same states visited by the gate", 639, 421, 284, 27, { size: 17, color: C.muted, italic: true, align: "center" });

// -----------------------------------------------------------------------------
// (c) Outcome: one paired recovery sequence plus the primary task-success result.
// -----------------------------------------------------------------------------
const resultTrigger = path.join(frames, "05_reim_trigger_seed8300042_t009.png");
const resultRelift = path.join(frames, "06_reim_relift_seed8300042_t040.png");
const resultSuccess = path.join(frames, "08_reim_success_seed8300042_t062.png");
await addImageFrame(resultTrigger, 969, 72, 80, 80, C.amber, "REIM risk trigger at t equals 9");
await addImageFrame(resultRelift, 1070, 72, 80, 80, C.green, "Recovery actor re-grasps and lifts the object");
await addImageFrame(resultSuccess, 1171, 72, 72, 80, C.green, "REIM succeeds at t equals 62");

const resultA1 = addShape({ x: 1053, y: 108, w: 6, h: 7, geometry: "ellipse", fill: "none" });
const resultA2 = addShape({ x: 1153, y: 108, w: 6, h: 7, geometry: "ellipse", fill: "none" });
resultA1.line = { fill: "none", width: 0 };
resultA2.line = { fill: "none", width: 0 };
const resultB1 = addShape({ x: 1064, y: 108, w: 6, h: 7, geometry: "ellipse", fill: "none" });
const resultB2 = addShape({ x: 1165, y: 108, w: 6, h: 7, geometry: "ellipse", fill: "none" });
resultB1.line = { fill: "none", width: 0 };
resultB2.line = { fill: "none", width: 0 };
connect(resultA1, resultB1, C.muted, { width: 1.5 });
connect(resultA2, resultB2, C.muted, { width: 1.5 });

addText("risk", 968, 157, 82, 23, { size: 18, color: C.amber, bold: true, align: "center" });
addText("re-grasp", 1068, 157, 84, 23, { size: 18, color: C.green, bold: true, align: "center" });
addText("success", 1166, 157, 82, 23, { size: 18, color: C.green, bold: true, align: "center" });

addText("Task success (%)", 969, 191, 166, 30, { size: 24, color: C.ink, bold: true });
addText("+17.0 pp", 1137, 191, 107, 30, { size: 23, color: C.green, bold: true, align: "right" });

const chart = slide.charts.add("bar", {
  position: { left: 970, top: 225, width: 274, height: 202 },
  categories: ["ACT", "REIM"],
  series: [{
    name: "Task success (%)",
    values: [0.734, 0.904],
    fill: C.blue,
    line: { style: "solid", fill: "#FFFFFF", width: 0 },
    points: [
      { idx: 0, fill: C.blue, line: { style: "solid", fill: C.blue, width: 0.5 } },
      { idx: 1, fill: C.green, line: { style: "solid", fill: C.green, width: 0.5 } },
    ],
    dataLabelOverrides: [
      { idx: 0, text: "73.4", position: "outEnd", showValue: false, textStyle: { typeface: fontFamily, fill: C.ink, fontSize: 18, bold: true } },
      { idx: 1, text: "90.4", position: "outEnd", showValue: false, textStyle: { typeface: fontFamily, fill: C.ink, fontSize: 18, bold: true } },
    ],
    valuesFormatCode: "0.0%",
  }],
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 85, varyColors: true },
  hasLegend: false,
  xAxis: {
    visible: true,
    tickLabelPosition: "none",
    textStyle: { fill: C.ink, fontSize: 18, bold: true },
    line: { style: "solid", fill: C.muted, width: 1 },
    majorGridlines: null,
  },
  yAxis: {
    visible: false,
    min: 0,
    max: 1,
    majorUnit: 0.25,
    numberFormatCode: "0%",
    textStyle: { fill: C.muted, fontSize: 17 },
    line: { fill: "none", width: 0 },
    majorGridlines: null,
  },
  dataLabels: {
    showValue: true,
    position: "outEnd",
    textStyle: { fill: C.ink, fontSize: 18, bold: true },
  },
  chartFill: "none",
  chartLine: { style: "solid", fill: "#FFFFFF", width: 0 },
  plotAreaFill: "none",
  plotAreaLine: { style: "solid", fill: "#FFFFFF", width: 0 },
});
applyPresentationChartFont(chart, { fontFamily });

addText("1,000 paired PickPlace episodes", 968, 430, 276, 23, { size: 18, color: C.muted, align: "center" });
addText("gain 95% CI  +14.7 to +19.3 pp", 968, 455, 276, 22, { size: 18, color: C.muted, italic: true, align: "center" });

slide.speakerNotes.textFrame.setText(
  "Figure 1 overview source. All robot frames are local Meta-World/MuJoCo simulation outputs from results/figures/recovery_operation_sequence_frames. " +
  "Primary result: ACT 73.4% and REIM 90.4% task success over 1,000 paired PickPlace episodes; paired gain +17.0 percentage points with 95% CI [+14.7,+19.3]. " +
  "Metrics come from paper_assets/reim_macros.tex and paper_assets/Table1_final_baseline.tex. " +
  "Recovery training counts are 42,386 train pairs and 8,212 disjoint validation pairs, from the audited project documentation. " +
  "The figure shows simulation only and makes no real-robot claim."
);

const candidatePath = path.join(TMP_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const preview = await presentation.export({ slide, format: "png", scale: 2 });
await fs.writeFile(path.join(TMP_DIR, "Figure1_overview_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(TMP_DIR, "Figure1_overview.layout.json"), await layout.text());

if (process.env.FINALIZE === "1") {
  const stagingDir = path.join(workspaceDir, ".codex-finalizer", "figure1-overview");
  await fs.mkdir(stagingDir, { recursive: true });
  await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
  const stagingCandidate = path.join(stagingDir, "candidate.pptx");
  await fs.copyFile(candidatePath, stagingCandidate);
  const requirements = {
    explicitTotalSlideCount: 1,
    requiredNativeTableOwnerSlides: [],
    requiredNativeChartOwnerSlides: [1],
    materializeLiteralChartWorkbooks: true,
    nativeChartTargetApplication: "powerpoint",
  };
  await finalizePresentation({
    ...requirements,
    workspaceDir,
    candidatePath: stagingCandidate,
    finalPath: FINAL_PPTX,
    pythonExecutable: RUNTIME_PYTHON,
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: [
      "--expected-slide-size-emu", "12192000,4762500",
      "--validate-heading-fit",
    ],
    requiredNativeTableOwnerSlides: [],
    fontPolicy: { basis: "design", families: [fontFamily] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(stagingDir, "Figure1_overview_editable_v6.pptx.validation.json"),
  });
  console.log(FINAL_PPTX);
}

console.log(candidatePath);
console.log(path.join(TMP_DIR, "Figure1_overview_preview.png"));
