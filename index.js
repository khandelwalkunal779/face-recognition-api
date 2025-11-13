import express from "express";
import { fileTypeFromBuffer } from "file-type";
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.raw({ limit: "10mb", type: "image/*" }));

app.post("/detect-and-recognize", async (req, res) => {
  try {
    const buffer = req.body;

    if (!Buffer.isBuffer(buffer) || buffer.length === 0) {
      return res.status(400).json({
        success: false,
        message: "Request body is empty or invalid.",
      });
    }

    const type = await fileTypeFromBuffer(buffer);
    const isImage = type ? type.mime.startsWith("image/") : false;

    if (!isImage) {
      return res.status(400).json({
        success: false,
        message: "Request body contains no valid Image",
      });
    }

    return res.status(200).json({
      success: true,
      message: `Successfully validated image with mime type: ${type.mime}`,
    });
  } catch (error) {
    console.error("Error:", error);
    return res.status(500).json({
      success: false,
      message: `Internal Server Error: ${error.message}`,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
