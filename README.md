# Face Recognition REST API

A lightweight, Node.js-based REST API that provides face detection, embedding generation, and recognition capabilities. This project implements a custom **In-Memory Vector Store** to match faces using Euclidean distance, built on top of face-api.js and TensorFlow.js.

## Features

- Face Detection & Embedding: Extracts 128-float feature vectors (descriptors) from facial landmarks.
- Vector Store: Implements a custom in-memory vector search to store and retrieve identities.
- Format Agnostic: Automatic handling of JPEG, PNG, and HEIC/HEIF (Apple) image formats.
- Euclidean Distance Matching: Uses specific threshold logic to identify known faces or label them as "unknown".
- Lightweight: It is so lightweight that you can deploy it directly to AWS Lambda or Azure Functions as a microservice, without heavy cold start issues.

## Tech Stack

- Runtime: Node.js
- Framework: Express.js
- AI Engine: [face-api.js](https://github.com/justadudewhohacks/face-api.js) (TensorFlow.js wrapper)
- Image Processing: canvas (Node implementation), heic-convert

## Installation & Setup

1. Clone the repository

   ```bash
   git clone https://github.com/khandelwalkunal779/face-recognition-api.git
   cd face-recognition-api
   ```

2. Install Dependencies

   This project relies on canvas which requires system-level dependencies on some OS versions (especially Linux/Mac).

   ```bash
   npm install
   ```

3. Download Model Weights

   The system requires pre-trained models. Create a `weights` folder in the root directory and download the following models from the [face-api.js weights repository](https://github.com/justadudewhohacks/face-api.js/tree/master/weights):

   1. `ssd_mobilenetv1_model-weights_manifest.json` (plus the shard files)
   2. `face_landmark_68_model-weights_manifest.json` (plus the shard files)
   3. `face_recognition_model-weights_manifest.json` (plus the shard files)

   Ensure these files are placed inside the `/weights` directory.

4. Start the Server

   ```bash
   npm start
   # OR
   node server.js
   ```

   The server runs on `http://localhost:3000` by default.

## API Documentation

**Note:** This API expects **Raw Binary Data** in the request body, not `multipart/form-data`.

1. Add a Face (Register User)

   Generates embeddings for the face and saves them to the in-memory vector store.

   - Endpoint: POST /add-face
   - Query Params: ?name=Kunal
   - Body: Raw binary image file.

   Example (cURL):

   ```bash
   curl -X POST "http://localhost:3000/add-face?name=Kunal" \
       --header "Content-Type: image/jpeg" \
       --data-binary @/path/to/photo.jpg
   ```

2. Detect and Recognize

   Compares the uploaded face against the vector store to find the best match.

   - Endpoint: POST /detect-and-recognize
   - Body: Raw binary image file.

   Example (cURL):

   ```bash
   curl -X POST "http://localhost:3000/detect-and-recognize" \
    --header "Content-Type: image/jpeg" \
    --data-binary @/path/to/unknown_photo.jpg
   ```

   Response:

   ```JSONC
   {
    "success": true,
    "name": "Kunal" // Returns "unknown" if distance > 0.6
   }
   ```

## How it works

### The Vector Store

Unlike traditional SQL databases, this project uses a mathematical approach to identification:

1. Embedding: When an image is uploaded, face-api.js maps the face to a vector of 128 numbers.
2. Storage: This vector is stored in a JavaScript Array alongside the user's name.
3. Matching: When recognizing a face, the system calculates the **Euclidean Distance** between the new face's vector and every vector in the store.
4. Thresholding:
   - The system uses a strict threshold of `0.6.`
   - If the closest match has a distance `< 0.6`, the name is returned.
   - Otherwise, the face is classified as `"unknown"`.

## Credits & Author

Developed and maintained by Kunal Khandelwal.

- GitHub: @khandelwalkunal779
- [LinkedIn](https://www.linkedin.com/in/khandelwalkunal779/)
- [Portfolio](https://khandelwalkunal779.github.io/)
