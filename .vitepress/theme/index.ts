import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // Add any custom layout slots here if needed
    })
  },
  enhanceApp({ app }) {
    // Register global components or add other app-level enhancements
  }
} 