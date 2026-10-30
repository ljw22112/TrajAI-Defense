import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, tokenStore } from "../api/client";

afterEach(() => { vi.restoreAllMocks(); tokenStore.set(null); });

function mockFetch(status: number, body: unknown) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(status === 204 ? null : JSON.stringify(body), { status }));
}

describe("api client", () => {
  it("sends the bearer token", async () => {
    tokenStore.set("abc");
    const f = mockFetch(200, { scenarios: 1 });
    await api.stats();
    expect((f.mock.calls[0][1] as RequestInit).headers).toMatchObject({ Authorization: "Bearer abc" });
  });

  it("maps FastAPI validation errors to per-field messages", async () => {
    mockFetch(422, { detail: [{ loc: ["body", "radius_km"], msg: "Input should be greater than 0" }] });
    const err = await api.createAsset({ name: "x", x_km: 0, y_km: 0, radius_km: -1, priority: 3 }).catch(e => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.fields.radius_km).toMatch(/greater than 0/);
  });

  it("clears the token on 401", async () => {
    tokenStore.set("expired");
    mockFetch(401, { detail: "invalid or expired token" });
    await expect(api.me()).rejects.toMatchObject({ status: 401 });
    expect(tokenStore.get()).toBeNull();
  });

  it("posts the login form as x-www-form-urlencoded", async () => {
    const f = mockFetch(200, { access_token: "t", token_type: "bearer", expires_in: 60 });
    await api.login("a@b.c", "pw");
    const init = f.mock.calls[0][1] as RequestInit;
    expect(String(init.body)).toBe("username=a%40b.c&password=pw");
  });

  it("reports network failures clearly", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(api.stats()).rejects.toMatchObject({ status: 0 });
  });
});
