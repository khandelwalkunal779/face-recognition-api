/*
Using an In-Memory-Array to implement a Vector Store which
will get cleaned as the server restarts.
*/

export let vectorStore = [];

export async function saveToVectorStore(name, embeddings) {
  vectorStore.push({
    name: name,
    embeddings: embeddings,
  });
  console.log(
    `Saved new entry for: ${name}. Total entries: ${vectorStore.length}`
  );
}

function getEuclideanDistance(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) {
    throw new Error("Vector length mismatch");
  }

  let sum = 0;
  for (let i = 0; i < vectorA.length; i++) {
    const diff = vectorA[i] - vectorB[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

export async function queryVectorStore(embeddings) {
  if (!vectorStore || vectorStore.length === 0) {
    console.log("Vector store is empty. Returning 'unknown'.");
    return "unknown";
  }

  const THRESHOLD = 0.6;
  let bestMatchLabel = "unknown";
  let minDistance = Number.MAX_VALUE;

  for (const item of vectorStore) {
    const distance = getEuclideanDistance(embeddings, item.embeddings);

    if (distance < minDistance) {
      minDistance = distance;
      bestMatchLabel = item.name;
    }
  }

  if (minDistance > THRESHOLD) {
    bestMatchLabel = "unknown";
  }

  console.log(
    `Best match found: ${bestMatchLabel} (Distance: ${minDistance.toFixed(4)})`
  );
  return bestMatchLabel;
}
