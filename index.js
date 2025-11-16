import express from "express";
import { fileTypeFromBuffer } from "file-type";
import { Canvas, Image, ImageData } from "canvas";
import faceapi from "face-api.js";
import path from "path";

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.raw({ limit: "10mb", type: "image/*" }));

app.post("/detect-and-recognize", async (req, res) => {
  try {
    const buffer = req.body;

    // ----- VALIDATIONS -----
    const isImage = await validateInput(buffer);
    if (!isImage) {
      return res.status(400).json({
        error: "Validations error",
        details: "Request body contains no valid Image",
      });
    }
    console.log("Validated Input");

    // ----- LOADING MODELS -----
    await loadModels();
    console.log("Loaded Models");

    // ----- PROCESSING IMAGE -----
    const embeddings = await getEmbeddings(buffer);
    const embeddingString = JSON.stringify(embeddings);
    const result = {
      success: true,
      message: "Image processed successfully",
      embedding: embeddingString,
    };
    return res.status(200).json(result);
  } catch (err) {
    console.error("Error:", err);
    return res
      .status(500)
      .json({ error: "Internal server error", details: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

async function validateInput(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length === 0) {
    return false;
  }
  const type = await fileTypeFromBuffer(buffer);
  return type ? type.mime.startsWith("image/") : false;
}

async function loadModels() {
  const MODEL_PATH = path.join(process.cwd(), "weights");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
  //   await faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_PATH);
  //   await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
}

async function getEmbeddings(buffer) {
  const image = new Image();
  image.src = buffer;
  const descriptor = await faceapi
    .detectSingleFace(image)
    .withFaceLandmarks()
    .withFaceDescriptor();
  return Array.from(descriptor.descriptor);
}
