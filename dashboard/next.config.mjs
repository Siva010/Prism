/** @type {import('next').NextConfig} */
const nextConfig = {
  // Everything is server-rendered against the admin API, so there is nothing to
  // export statically and no client bundle worth splitting.
  reactStrictMode: true,
};

export default nextConfig;
