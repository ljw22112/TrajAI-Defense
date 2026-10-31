/** Opens the browser print dialog; the print stylesheet hides navigation and controls, so "Save as PDF" gives a clean report. */
export function PrintButton({ label = "PDF로 저장" }: { label?: string }) {
  return <button type="button" className="btn ghost no-print" onClick={() => window.print()}>{label}</button>;
}
