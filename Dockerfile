FROM node:20-slim

WORKDIR /docs

# Copy package files first for better caching
COPY package*.json ./

# Create node_modules with correct permissions and install dependencies
RUN mkdir -p /docs/node_modules && \
    chown -R node:node /docs && \
    npm install

# Switch to non-root user
USER node

# Copy the rest of the application
COPY --chown=node:node . .

EXPOSE 5173

CMD ["npm", "run", "docs:dev"] 