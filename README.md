# Agent with MCP Server and AI Agent

A sophisticated mathematical computation system with integrated PowerPoint visualization capabilities. The system leverages AI-driven natural language processing (via Gemini API) to perform complex mathematical operations. Built with two synergistic components, it offers a seamless workflow from computation to presentation.

## Components

### MCP Server (mcp-server.py)
- Provides various mathematical functions including:
  - Basic arithmetic operations (add, subtract, multiply, divide)
  - List operations (summing lists)
  - Advanced math functions (power, square root, cube root, factorial, logarithm)
  - Trigonometric functions (sin, cos, tan)
- PowerPoint automation capabilities:
  - Create and manage presentations
  - Professional slide formatting
  - Automated presentation handling using python-pptx

### AI Agent (agent.py)
- Provides intelligent natural language interface via Gemini API
- Performs step-by-step mathematical reasoning and problem-solving
- Executes complex mathematical computations with detailed explanations
- Maintains conversation state and iteration tracking
- Handles API timeouts and errors gracefully

## Setup and Usage

1. Ensure Python is installed on your system
2. Run the server:
   ```
   python mcp-server.py dev
   ```
3. Run the AI agent:
   ```
   python agent.py
   ```

## Example Operations

- Complex Mathematical Problem Solving
  - Step-by-step reasoning and computation
  - Advanced mathematical functions and operations
  - Detailed explanation of solution process
- Automated Visualization
  - PowerPoint generation of mathematical results
  - Visual representation of computational steps
  - Professional formatting and layout
- Interactive Mathematical Processing
  - Natural language query interpretation
  - Real-time computation and visualization
  - Dynamic result presentation

## Requirements

- Python 3.x
- PowerPoint installation (for presentation features)
- Required Python packages (mcp, pywinauto, python-pptx)
