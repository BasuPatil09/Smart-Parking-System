import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api",
  timeout: 120_000,
});

export async function predictImage(
  imageFile,
  onUploadProgress,
  conf = 0.15,
  iou = 0.45,
  imgsz = 1600,
  augment = true,
  maxDet = 1000,
) {
  const formData = new FormData();
  formData.append("image", imageFile);

  const { data } = await api.post(
    `/predict-image?conf=${conf}&iou=${iou}&imgsz=${imgsz}&max_det=${maxDet}&augment=${augment}`,
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress,
    },
  );

  return data;
}

export async function getStatus() {
  const { data } = await api.get("/status");
  return data;
}

export async function getMetrics() {
  const { data } = await api.get("/metrics");
  return data;
}

export async function getModelEvaluation() {
  const { data } = await api.get("/model-evaluation");
  return data;
}

export async function healthCheck() {
  const { data } = await api.get("/health");
  return data;
}
