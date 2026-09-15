# Next.js operator/device web client.
FROM node:20-bookworm-slim

WORKDIR /app

# Baked into the client bundle at build time: reach the backend on the same
# origin the page was served from (the reverse proxy in front routes the API/WS
# paths). See web/lib/config.ts.
ARG NEXT_PUBLIC_BACKEND_SAME_ORIGIN=1
ENV NEXT_PUBLIC_BACKEND_SAME_ORIGIN=$NEXT_PUBLIC_BACKEND_SAME_ORIGIN

# Install deps first (cached unless the lockfile changes). npm ci needs the dev
# deps (typescript/types) to build, so NODE_ENV isn't forced to production here.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

ENV NODE_ENV=production
EXPOSE 3000
# package.json start = `next start --hostname 0.0.0.0`
CMD ["npm", "run", "start"]
