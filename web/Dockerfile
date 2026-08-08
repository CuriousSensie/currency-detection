FROM node:22-alpine AS deps
WORKDIR /app
RUN corepack enable
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM node:22-alpine AS builder
WORKDIR /app
RUN corepack enable
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
COPY --from=deps /app/node_modules ./node_modules
COPY web ./
COPY README.md THIRD_PARTY_NOTICES.md ./
COPY docs ./docs
COPY artifacts/results /artifacts/results
RUN node node_modules/next/dist/bin/next build

FROM node:22-alpine AS runner
WORKDIR /app/app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
USER node
COPY --from=builder --chown=node:node /app/.next/standalone /app
COPY --from=builder --chown=node:node /app/.next/static /app/app/.next/static
COPY --from=builder --chown=node:node /app/public /app/app/public
EXPOSE 3000
CMD ["node", "server.js"]
