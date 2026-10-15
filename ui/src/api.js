// Tiny fetch wrapper around the FastAPI backend. All paths are relative so the
// Vite dev-server proxy (see vite.config.js) forwards them to :8000.

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

async function upload(file, role) {
  const form = new FormData();
  form.append("file", file);
  form.append("role", role);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `/api/upload -> ${res.status}`);
  return data;
}

export const api = {
  status: () => fetch("/api/status").then((r) => r.json()),
  chat: (message, role) => post("/api/chat", { message, role }),
  ingest: (path, role) => post("/api/ingest", { path: path || null, role }),
  upload,
  reset: () => post("/api/reset", {}),
};
