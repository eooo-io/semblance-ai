import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: 'Semblance AI Documentation',
  description: 'Documentation for the Semblance AI Project',
  base: '/semblance-ai-project/',
  ignoreDeadLinks: true,  // Temporarily ignore dead links
  
  vite: {
    ssr: {
      noExternal: ['vitepress']
    },
    build: {
      target: 'esnext'
    }
  },
  
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#4051B5' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:site_name', content: 'Semblance AI Documentation' }],
  ],

  themeConfig: {
    logo: '/logo.png',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', items: [
        { text: 'Quick Start', link: '/getting-started/quick-start' },
        { text: 'Installation', link: '/getting-started/installation' },
        { text: 'Configuration', link: '/getting-started/configuration' }
      ]},
      { text: 'Core', items: [
        { text: 'Overview', link: '/core/' },
        { text: 'Architecture', link: '/core/architecture' },
        { text: 'Model Training', link: '/core/training' },
        { text: 'Configuration', link: '/core/configuration' }
      ]},
      { text: 'Components', items: [
        { text: 'Data Curation', items: [
          { text: 'Overview', link: '/curation/' },
          { text: 'Data Collection', link: '/curation/data-collection' },
          { text: 'Annotation', link: '/curation/annotation' },
          { text: 'Quality Control', link: '/curation/quality-control' },
          { text: 'API Reference', link: '/curation/api-reference' }
        ]},
        { text: 'RAG System', items: [
          { text: 'Overview', link: '/rag/' },
          { text: 'Architecture', link: '/rag/architecture' },
          { text: 'Deployment', link: '/rag/deployment' },
          { text: 'API Reference', link: '/rag/api-reference' },
          { text: 'Web Interface', link: '/rag/web-interface' }
        ]}
      ]},
      { text: 'Integration', items: [
        { text: 'Overview', link: '/integration/' },
        { text: 'Component Interaction', link: '/integration/component-interaction' },
        { text: 'Data Flow', link: '/integration/data-flow' },
        { text: 'Security', link: '/integration/security' }
      ]},
      { text: 'Development', items: [
        { text: 'Contributing', link: '/development/contributing' },
        { text: 'Local Setup', link: '/development/local-setup' },
        { text: 'Testing', link: '/development/testing' },
        { text: 'CI/CD', link: '/development/ci-cd' }
      ]},
      { text: 'Reference', items: [
        { text: 'API Reference', link: '/reference/api' },
        { text: 'Tools Reference', link: '/reference/tools' },
        { text: 'Configuration', link: '/reference/configuration' },
        { text: 'CLI Tools', link: '/reference/cli' },
        { text: 'Troubleshooting', link: '/reference/troubleshooting' }
      ]}
    ],

    sidebar: {
      '/getting-started/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Quick Start', link: '/getting-started/quick-start' },
            { text: 'Installation', link: '/getting-started/installation' },
            { text: 'Configuration', link: '/getting-started/configuration' }
          ]
        }
      ],
      '/core/': [
        {
          text: 'Core',
          items: [
            { text: 'Overview', link: '/core/' },
            { text: 'Architecture', link: '/core/architecture' },
            { text: 'Model Training', link: '/core/training' },
            { text: 'Configuration', link: '/core/configuration' }
          ]
        }
      ],
      '/curation/': [
        {
          text: 'Data Curation',
          items: [
            { text: 'Overview', link: '/curation/' },
            { text: 'Data Collection', link: '/curation/data-collection' },
            { text: 'Annotation', link: '/curation/annotation' },
            { text: 'Quality Control', link: '/curation/quality-control' },
            { text: 'API Reference', link: '/curation/api-reference' }
          ]
        }
      ],
      '/rag/': [
        {
          text: 'RAG System',
          items: [
            { text: 'Overview', link: '/rag/' },
            { text: 'Architecture', link: '/rag/architecture' },
            { text: 'Deployment', link: '/rag/deployment' },
            { text: 'API Reference', link: '/rag/api-reference' },
            { text: 'Web Interface', link: '/rag/web-interface' }
          ]
        }
      ],
      '/integration/': [
        {
          text: 'Integration',
          items: [
            { text: 'Overview', link: '/integration/' },
            { text: 'Component Interaction', link: '/integration/component-interaction' },
            { text: 'Data Flow', link: '/integration/data-flow' },
            { text: 'Security', link: '/integration/security' }
          ]
        }
      ],
      '/development/': [
        {
          text: 'Development',
          items: [
            { text: 'Contributing', link: '/development/contributing' },
            { text: 'Local Setup', link: '/development/local-setup' },
            { text: 'Testing', link: '/development/testing' },
            { text: 'CI/CD', link: '/development/ci-cd' }
          ]
        }
      ],
      '/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'API Reference', link: '/reference/api' },
            { text: 'Tools Reference', link: '/reference/tools' },
            { text: 'Configuration', link: '/reference/configuration' },
            { text: 'CLI Tools', link: '/reference/cli' },
            { text: 'Troubleshooting', link: '/reference/troubleshooting' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/eooo-io/semblance-ai' },
      { icon: 'twitter', link: 'https://twitter.com/eooo_io' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present eooo.io'
    },

    search: {
      provider: 'local'
    },

    editLink: {
      pattern: 'https://github.com/eooo-io/semblance-ai/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    }
  },

  markdown: {
    lineNumbers: true,
    config: (md) => {
      // Add markdown-it plugins here if needed
    }
  }
}) 