# Questions and Answers Project

This project contains technical questions and detailed answers, with Claude MCP (Model Context Protocol) integration for enhanced capabilities.

## Setup

First, install the required MCP servers:

```bash
claude mcp add sequential-thinking -s user -- npx -y @modelcontextprotocol/server-sequential-thinking@2025.7.1
claude mcp add web-search -s user -- npx -y @dannyboy2042/freebird-mcp@1.5.1
claude mcp add playwright -s user -- npx -y @playwright/mcp@0.0.37
claude mcp add memory -s user -- npx -y @modelcontextprotocol/server-memory@2025.8.4
claude mcp add --scope user --transport sse context7 https://context7.liam.sh/sse
```

**Note**: This project also includes a `.cursor/mcp.json` file that configures the same MCP servers for use with Cursor IDE, ensuring consistent functionality across both Claude and Cursor environments.

## Usage

1. **Start Claude**: Open a terminal and run:

   ```bash
   claude
   ```

2. **Integrate with your editor**: Run the following command to integrate Claude with Cursor or another supported editor:

   ```bash
   /ide
   ```

3. **Validate MCP servers**: Check that all MCP servers are available and working properly:

   ```bash
   /mcp
   ```

## Project Structure

- `question-001-terraform-eks-kubernetes-file-structure/` - Contains detailed analysis and answers for Terraform EKS and Kubernetes file structure questions
- `CLAUDE.md` - Claude-specific configuration and notes
- Log files for debugging and monitoring

## Features

With the configured MCP servers, this project provides:

- **Sequential Thinking**: Advanced reasoning and problem-solving capabilities
- **Web Search**: Real-time web search functionality via Freebird
- **Browser Automation**: Playwright integration for web testing and automation
- **Memory**: Persistent memory across sessions
- **Context7**: Enhanced context understanding and documentation access

## Getting Help

If you encounter issues:

1. Verify MCP server status with `/mcp`
2. Check log files for error details
3. Restart Claude if needed
