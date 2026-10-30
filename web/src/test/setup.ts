import "@testing-library/jest-dom/vitest";

// jsdom has no canvas / ResizeObserver
class RO { observe() {} unobserve() {} disconnect() {} }
(globalThis as any).ResizeObserver = RO;
HTMLCanvasElement.prototype.getContext = (() => null) as any;
window.matchMedia = window.matchMedia || ((() => ({ matches: false, addEventListener() {}, removeEventListener() {} })) as any);
