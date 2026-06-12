import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EIRA · ECE Information & Resource Assistant",
  description: "EIRA (ECE Information & Resource Assistant) — a friendly AI guide to the TAMU Electrical & Computer Engineering department",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
