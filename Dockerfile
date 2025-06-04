FROM node:20-slim

WORKDIR /docs

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "docs:dev"] 