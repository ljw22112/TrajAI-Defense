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

describe("csv export", () => {
  it("downloads with the bearer token and a fixed file name", async () => {
    tokenStore.set("tok");
    const f = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("scenario_id,method\n1,ekf_std\n", { status: 200 }));
    (URL as any).createObjectURL = vi.fn(() => "blob:x");
    (URL as any).revokeObjectURL = vi.fn();
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      expect(this.download).toBe("trajai_estimates.csv");
    });
    const { downloadEstimatesCsv } = await import("../api/client");
    await downloadEstimatesCsv();
    expect(f.mock.calls[0][0]).toMatch(/\/api\/scenarios\/export\.csv$/);
    expect((f.mock.calls[0][1] as RequestInit).headers).toMatchObject({ Authorization: "Bearer tok" });
    expect(click).toHaveBeenCalledOnce();
  });

  it("surfaces server errors", async () => {
    mockFetch(429, { detail: "too many failed logins; retry in 60 s" });
    const { downloadEstimatesCsv } = await import("../api/client");
    await expect(downloadEstimatesCsv()).rejects.toMatchObject({ status: 429 });
  });
});
