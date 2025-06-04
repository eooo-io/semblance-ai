// Add copy button to code blocks
document.querySelectorAll('pre code').forEach((block) => {
  const button = document.createElement('button');
  button.className = 'md-clipboard md-icon';
  button.title = 'Copy to clipboard';
  button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"></path></svg>';
  
  button.addEventListener('click', () => {
    navigator.clipboard.writeText(block.textContent).then(() => {
      button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z"></path></svg>';
      setTimeout(() => {
        button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"></path></svg>';
      }, 1000);
    });
  });

  const pre = block.parentNode;
  pre.insertBefore(button, block);
});

// Add anchor links to headers
document.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]').forEach((header) => {
  const link = document.createElement('a');
  link.className = 'headerlink';
  link.href = `#${header.id}`;
  link.innerHTML = '#';
  header.appendChild(link);
});

// Initialize mermaid diagrams
if (typeof mermaid !== 'undefined') {
  mermaid.initialize({
    startOnLoad: true,
    theme: document.body.getAttribute('data-md-color-scheme') === 'slate' ? 'dark' : 'default',
    flowchart: {
      curve: 'basis'
    },
    sequence: {
      showSequenceNumbers: true
    }
  });
} 