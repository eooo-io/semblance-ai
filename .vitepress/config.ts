import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Semblance AI Documentation',
  description: 'Documentation for the Semblance AI Project',
  base: '/semblance-ai-project/',
  
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
      { text: 'Overview', items: [
        { text: 'Project Vision', link: '/overview/project-vision' },
        { text: 'Ecosystem Map', link: '/overview/ecosystem-map' }
      ]},
      { text: 'Components', items: [
        { text: 'Data Curation', link: '/components/data-curation/' },
        { text: 'RAG System', link: '/components/rag-system/' }
      ]},
      { text: 'Standards', items: [
        { text: 'Coding Guidelines', link: '/standards/coding-guidelines' },
        { text: 'Documentation Guidelines', link: '/standards/documentation-guidelines' },
        { text: 'KAG Data Curation', link: '/standards/kag-data-curation' }
      ]},
      { text: 'Resources', items: [
        { text: 'Tools', link: '/resources/tools' },
        { text: 'Tools Checklist', link: '/resources/tools-checklist' },
        { text: 'FAQ', link: '/resources/faq' }
      ]}
    ],

    sidebar: {
      '/overview/': [
        {
          text: 'Overview',
          items: [
            { text: 'Project Vision', link: '/overview/project-vision' },
            { text: 'Ecosystem Map', link: '/overview/ecosystem-map' }
          ]
        }
      ],
      '/components/': [
        {
          text: 'Components',
          items: [
            { 
              text: 'Data Curation',
              link: '/components/data-curation/',
              collapsed: false
            },
            { 
              text: 'RAG System',
              link: '/components/rag-system/',
              collapsed: false
            }
          ]
        }
      ],
      '/standards/': [
        {
          text: 'Standards',
          items: [
            { text: 'Coding Guidelines', link: '/standards/coding-guidelines' },
            { text: 'Documentation Guidelines', link: '/standards/documentation-guidelines' },
            { text: 'KAG Data Curation', link: '/standards/kag-data-curation' }
          ]
        }
      ],
      '/resources/': [
        {
          text: 'Resources',
          items: [
            { text: 'Tools', link: '/resources/tools' },
            { text: 'Tools Checklist', link: '/resources/tools-checklist' },
            { text: 'FAQ', link: '/resources/faq' }
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