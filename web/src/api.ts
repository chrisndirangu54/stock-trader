import { auth } from "./firebase";

const API_BASE = import.meta.env.VITE_API_BASE || "";

async function token() {
  const user = auth.currentUser;
  if (!user) throw new Error("Not signed in");
  return user.getIdToken();
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const idToken = await token();
  const res = await fetch(API_BASE + path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${idToken}`,
      ...(init.headers || {})
    }
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `HTTP ${res.status}`);
  }
  return res.json();
}
