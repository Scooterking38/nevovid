# Project Title: NevoRL

## One-line Tagline:
A Python-based procedural environment generator and parser for ASCII circuit layouts.

## Architecture Overview:
The `nevovid` repository is structured around two primary modules: `demo.py` and `nevorl.py`. The `demo.py` script serves as the entry point, utilizing libraries like Pygame and NumPy to render and simulate environments defined in ASCII format. It leverages the `NevoRLCompiler` class from `nevorl.py`, which is responsible for parsing a custom domain-specific language (DSL) that defines environment properties such as states, actions, observations, rewards, and terminal conditions.

The repository includes:
- **demo.py**: The main script that initializes Pygame, sets up the environment using ASCII circuit definitions, and simulates the environment.
- **nevorl.py**: Contains the `NevoRLCompiler` class which uses Lark to parse a DSL for defining environments. This class handles the grammar definition and transformation of parsed data into executable code.

## Tech Stack:
- **Programming Language:** Python
- **Libraries:**
  - Pygame: For rendering and handling game-related functionalities.
  - NumPy: For numerical operations and array manipulations.
  - ImageIO: For image processing, though not directly used in the provided samples.
  - Lark: A parsing library used for defining and parsing the custom DSL.

## Getting Started / Setup Guide:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/nevovid.git
   cd nevovid
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, then install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Demo Script:**
   Execute the `demo.py` script to start the environment simulation:
   ```bash
   python demo.py
   ```

4. **Customize Environment Definitions:**
   Modify the ASCII circuit definitions in `demo.py` or extend the DSL in `nevorl.py` to create new environments with different properties.

5. **Environment Variables (Optional):**
   If running on a headless server, ensure the SDL video driver is set correctly:
   ```bash
   export SDL_VIDEODRIVER=dummy
   ```

This setup will allow you to visualize and interact with procedurally generated environments defined using the custom DSL provided by `nevorl.py`.