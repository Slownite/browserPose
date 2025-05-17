let frameStack = [];
const NUM_FRAMES = 16;
const FRAME_INTERVAL_MS = 150;
let session, labels;

async function setupWebcam() {
  const webcam = document.getElementById("webcam");
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcam.srcObject = stream;

  return new Promise((resolve) => {
    webcam.onloadeddata = () => resolve(webcam);
  });
}

function captureFrame(video) {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  if (video.readyState < 2) {
    throw new Error("Webcam not ready");
  }

  ctx.drawImage(video, 0, 0, 256, 256);
  const imageData = ctx.getImageData(0, 0, 256, 256);

  if (!imageData || !imageData.data) {
    throw new Error("Frame capture failed (no image data)");
  }

  const rgb = [];
  for (let i = 0; i < imageData.data.length; i += 4) {
    rgb.push(imageData.data[i] / 255);     // R
    rgb.push(imageData.data[i + 1] / 255); // G
    rgb.push(imageData.data[i + 2] / 255); // B
  }

  return rgb;
}

function createInputTensor(frames) {
  const tensor = new Float32Array(1 * 3 * NUM_FRAMES * 256 * 256);

  for (let t = 0; t < NUM_FRAMES; t++) {
    const frame = frames[t];
    for (let i = 0; i < 256 * 256; i++) {
      tensor[t * 256 * 256 + i] = frame[i * 3];                           // R
      tensor[NUM_FRAMES * 256 * 256 + t * 256 * 256 + i] = frame[i * 3 + 1]; // G
      tensor[2 * NUM_FRAMES * 256 * 256 + t * 256 * 256 + i] = frame[i * 3 + 2]; // B
    }
  }

  return new ort.Tensor("float32", tensor, [1, 3, NUM_FRAMES, 256, 256]);
}

async function startRecognition() {
  const predictionEl = document.getElementById("prediction");

  const webcam = await setupWebcam();

  if (!session) {
    predictionEl.innerText = "Loading model...";
    session = await ort.InferenceSession.create("model/x3d_m_softmax.onnx");
  }

  if (!labels) {
    rawLabels = await fetch("./labels.json").then((res) => res.json());
    labels = {};
    for (const [label, id] of Object.entries(rawLabels)) {
      labels[id.toString()] = label;
    }
  }

  const runLoop = async () => {
    frameStack = [];

    for (let i = 0; i < NUM_FRAMES; i++) {
      try {
        const frame = captureFrame(webcam);
        frameStack.push(frame);
      } catch (err) {
        console.warn("Skipping frame:", err.message);
        i--; // retry this frame
        await new Promise((r) => setTimeout(r, 100));
        continue;
      }
      await new Promise((r) => setTimeout(r, FRAME_INTERVAL_MS));
    }
    const inputTensor = createInputTensor(frameStack);

    try {
      const results = await session.run({ input: inputTensor });
      console.log("results", results)
      const probs = results.logits.data;
      const maxIdx = probs.indexOf(Math.max(...probs));
      const prediction = labels[maxIdx] || `Class ${maxIdx}`;
      predictionEl.innerText = `Prediction: ${prediction}`;
    } catch (err) {
      predictionEl.innerText = `Error: ${err.message}`;
      console.error(err);
    }

    requestAnimationFrame(runLoop);
  };

  runLoop();
}
