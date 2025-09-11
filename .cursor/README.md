# Cursor MCP Configuration

This directory contains project-specific Model Context Protocol (MCP) server configurations for Cursor IDE.

## Configured MCP Servers

The `mcp.json` file configures **5 optimized MCP servers** specifically tailored for this questions-and-answers project:

### Analysis & Context Servers

- **`context7`** - Code documentation and analysis for AI editors
- **`sequential-thinking`** - Enhanced reasoning and step-by-step problem solving
- **`memory`** - Persistent memory across sessions

### Automation & Web Servers

- **`playwright`** - Browser automation and web scraping capabilities
- **`web-search`** - Free web search capabilities using DuckDuckGo (no API key required)

## Environment Variables and Secrets

The current MCP servers don't require environment variables or secrets configuration. All servers work out-of-the-box with no additional setup needed.

## Usage

1. **Automatic Loading**: Cursor will automatically load these MCP servers when opening this project
2. **Manual Restart**: If you modify `mcp.json`, restart Cursor to apply changes
3. **Server Status**: Check the Cursor status bar to see which MCP servers are active

## Project-Specific Benefits

These MCP servers are particularly useful for this questions-and-answers project because:

- **Web Search**: Research answers, documentation, and technical information
- **Context analysis**: Enhanced code understanding and documentation analysis
- **Browser automation**: Web scraping and automated testing capabilities
- **Memory**: Persistent knowledge storage across sessions
- **Sequential thinking**: Enhanced reasoning for complex problem solving and analysis

## Transport Configuration

The MCP servers are optimized with appropriate transport types:

### stdio Servers (3)

- **web-search** - Web search queries
- **playwright** - Browser automation
- **memory** - Persistent memory storage

### SSE Servers (2)

- **sequential-thinking** - Real-time reasoning steps
- **context7** - Streaming code analysis

## Troubleshooting

- **Server Not Starting**: Check that Node.js is installed for npx-based servers
- **Network Issues**: Verify internet connectivity for package downloads
- **Server Status**: Check Cursor status bar to see which MCP servers are active

## Setup Instructions

### 1. Restart Cursor

Simply restart Cursor to load the MCP servers. No additional configuration is required.

## Customization

To add or modify servers:

1. Edit `mcp.json`
2. Follow the format: `"server-name": { "command": "...", "args": [...] }`
3. Add transport type (`"stdio"` or `"sse"`) as needed
4. Restart Cursor

For more information, see the [Cursor MCP Documentation](https://cursordocs.com/en/docs/context/model-context-protocol).
