import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Plot } from "../components/Plot";
import { ErrorBox, ThreatBadge, fmt } from "../components/ui";
import { ApiError } from "../api/client";

describe("ui helpers", () => {
  it("formats numbers and missing values", () => {
    expect(fmt(1234.567, 1)).toBe("1,234.6");
    expect(fmt(null)).toBe("—");
    expect(fmt(Number.NaN)).toBe("—");
  });

  it("renders API error details", () => {
    render(<ErrorBox error={new ApiError(403, "requires role 'analyst' or higher")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("analyst");
  });

  it("renders threat level badge", () => {
    render(<ThreatBadge level={4} />);
    expect(screen.getByText("LEVEL 4")).toHaveClass("lv4");
  });

  it("exposes charts to assistive tech", () => {
    render(<Plot series={[{ kind: "line", x: [0, 1], y: [0, 1], color: "--ink" }]} xLabel="x" yLabel="y" ariaLabel="test chart" />);
    expect(screen.getByRole("img", { name: "test chart" })).toBeInTheDocument();
  });
});
