/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Next 16 auto-generates AGENTS.md/CLAUDE.md on dev/build -- not something
  // this project asked for, disabled to keep the repo clean.
  agentRules: false,
};

module.exports = nextConfig;
