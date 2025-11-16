import express from "express";
import { fileTypeFromBuffer } from "file-type";
import { Canvas, Image, ImageData } from "canvas";
import faceapi from "face-api.js";
import path from "path";
import heicConvert from "heic-convert";

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const PORT = process.env.PORT || 3000;

// --- In-Memory Vector Store ---
let vectorStore = [];

app.use(express.raw({ limit: "10mb", type: "image/*" }));

app.post("/add-face", async (req, res) => {
  try {
    const buffer = req.body;
    let processingBuffer = buffer;
    const { name } = req.query;

    // ----- VALIDATIONS -----
    if (!name) {
      return res.status(400).json({
        error: "Validation error",
        details: "Missing 'name' query parameter",
      });
    }

    const imageType = await validateInput(buffer);
    if (!imageType) {
      return res.status(400).json({
        error: "Validation error",
        details: "Request body contains no valid image (JPEG, PNG, or HEIC)",
      });
    }

    // ----- HEIC CONVERSION -----
    if (imageType === "image/heic" || imageType === "image/heif") {
      console.log("HEIC image detected. Converting to JPEG...");
      try {
        processingBuffer = await convertHeicToJpg(buffer);
      } catch (conversionError) {
        return res.status(500).json({
          error: "Image conversion error",
          details: conversionError.message,
        });
      }
    }

    console.log(`Validation passed for: ${name}`);

    // ----- PROCESSING IMAGE -----
    const embeddings = await getEmbeddings(processingBuffer);
    if (!embeddings) {
      return res.status(400).json({
        error: "Processing error",
        details: "No face detected in the provided image.",
      });
    }
    console.log(`Generated embeddings for: ${name}`);

    // ----- SAVE TO VECTOR STORE -----
    await saveToVectorStore(name, embeddings);

    const result = {
      success: true,
      message: `Face for ${name} saved successfully`,
    };
    return res.status(201).json(result);
  } catch (err) {
    console.error("Error in /add-face:", err);
    return res
      .status(500)
      .json({ error: "Internal server error", details: err.message });
  }
});

app.post("/detect-and-recognize", async (req, res) => {
  try {
    const buffer = req.body;
    let processingBuffer = buffer;

    // ----- VALIDATIONS -----
    const imageType = await validateInput(buffer);
    if (!imageType) {
      return res.status(400).json({
        error: "Validation error",
        details: "Request body contains no valid image (JPEG, PNG, or HEIC)",
      });
    }

    // ----- NEW: HEIC CONVERSION -----
    if (imageType === "image/heic" || imageType === "image/heif") {
      console.log("HEIC image detected. Converting to JPEG...");
      try {
        processingBuffer = await convertHeicToJpg(buffer);
      } catch (conversionError) {
        return res.status(500).json({
          error: "Image conversion error",
          details: conversionError.message,
        });
      }
    }
    console.log("Validated Input");

    // ----- PROCESSING IMAGE -----
    const embeddings = await getEmbeddings(processingBuffer);
    if (!embeddings) {
      return res.status(400).json({
        error: "Processing error",
        details: "No face detected in the provided image.",
      });
    }
    console.log("Generated embeddings for recognition");

    // ----- QUERY VECTOR STORE -----
    const personName = await queryVectorStore(embeddings);

    const result = {
      success: true,
      name: personName,
    };
    return res.status(200).json(result);
  } catch (err) {
    console.error("Error in /recognize:", err);
    return res
      .status(500)
      .json({ error: "Internal server error", details: err.message });
  }
});

// --- Helper Functions ---

async function validateInput(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length === 0) {
    return false;
  }

  const type = await fileTypeFromBuffer(buffer);
  if (!type) return false;

  if (
    type.mime.startsWith("image/jpeg") ||
    type.mime.startsWith("image/png") ||
    type.mime.startsWith("image/heic") ||
    type.mime.startsWith("image/heif")
  ) {
    return type.mime;
  }

  return false;
}

async function loadModels() {
  const MODEL_PATH = path.join(process.cwd(), "weights");
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH),
    faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH),
    faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH),
  ]);
  console.log("All face-api models loaded successfully"); // Updated log
}

async function convertHeicToJpg(heicBuffer) {
  try {
    const outputBuffer = await heicConvert({
      buffer: heicBuffer,
      format: "JPEG",
      quality: 0.9,
    });
    return outputBuffer;
  } catch (err) {
    console.error("Error during HEIC conversion:", err);
    throw new Error(`HEIC Conversion Failed: ${err.message}`);
  }
}

async function getEmbeddings(buffer) {
  const image = await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) =>
      reject(new Error(`Failed to load image: ${err.message}`));
    img.src = buffer;
  });

  const detection = await faceapi
    .detectSingleFace(image)
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) {
    return null;
  }
  return Array.from(detection.descriptor);
}

async function saveToVectorStore(name, embeddings) {
  vectorStore.push({
    name: name,
    embeddings: embeddings,
  });
  console.log(
    `Saved new entry for: ${name}. Total entries: ${vectorStore.length}`
  );
}

async function queryVectorStore(embeddings) {
  if (vectorStore.length === 0) {
    console.log("Vector store is empty. Returning 'unknown'.");
    return "unknown";
  }

  const labeledDescriptors = vectorStore.map(
    (item) =>
      new faceapi.LabeledFaceDescriptors(item.name, [
        new Float32Array(item.embeddings),
      ])
  );

  const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
  const bestMatch = faceMatcher.findBestMatch(new Float32Array(embeddings));

  console.log(`Best match found: ${bestMatch.label}`);
  return bestMatch.label;
}

async function startServer() {
  try {
    console.log("Loading face-api models...");
    await loadModels();

    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
      console.log("Ready to accept requests");
    });
  } catch (err) {
    console.error("Failed to load models or start server:", err);
    process.exit(1);
  }
}

startServer();
