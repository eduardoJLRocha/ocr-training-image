const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export async function extractText(file, model, signal) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${API_BASE_URL}/api/ocr?model=${encodeURIComponent(model)}`, {
    method: "POST",
    body: formData,
    signal,
  });
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || "Não foi possível processar a imagem.");
  }

  return payload;
}
