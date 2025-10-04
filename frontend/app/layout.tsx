import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Exoplanet Classifier - AI-Powered Discovery",
  description: "Identify and classify exoplanets using machine learning trained on NASA data",
  keywords: ["exoplanet", "NASA", "machine learning", "classification", "astronomy"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-space-darker min-h-screen">
        {children}
      </body>
    </html>
  );
}
