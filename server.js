import express from "express";

import {
  validateInput,
  loadModels,
  convertHeicToJpg,
  getEmbeddings,
} from "./func-utils.js";
import {
  saveToVectorStore,
  queryVectorStore,
} from "./vec-store-utils/in-memory-store.js";

const app = express();
const PORT = process.env.PORT || 3000;

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
