import path from "path";
import heicConvert from "heic-convert";
import { fileTypeFromBuffer } from "file-type";
import { Canvas, Image, ImageData } from "canvas";
import faceapi from "face-api.js";

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

export async function validateInput(buffer) {
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

export async function loadModels() {
  const MODEL_PATH = path.join(process.cwd(), "weights");
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH),
    faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH),
    faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH),
  ]);
  console.log("All face-api models loaded successfully");
}

export async function convertHeicToJpg(heicBuffer) {
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

export async function getEmbeddings(buffer) {
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
