.PHONY: install dev build preview clean

install:
	npm install

dev:
	npm run docs:dev

build:
	npm run docs:build

preview:
	npm run docs:preview

clean:
	rm -rf .vitepress/dist
	rm -rf node_modules 