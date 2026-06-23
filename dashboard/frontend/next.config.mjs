/** @type {import('next').NextConfig} */
const nextConfig = {
  // Plotly is a large browser-only library; exclude from server bundle
  webpack: (config) => {
    config.externals = config.externals || [];
    return config;
  },
};

export default nextConfig;
