# GitHub Link
Please view the README file on GitHub for better formatting: https://github.com/yohanajocelyn/diagnostic_network_ai.git

# Libraries and Algorithms Used

## 1. Bayesian Network (pgmpy):

We use the pgmpy library to create and work with the Bayesian Network for our probabilistic reasoning algorithm. The library provides tools to define the structure of the network, specify conditional probability distributions, and perform inference to compute the probabilities of various outcomes based on observed evidence.

## 2. Hill Climbing (pgmpy):

Hill Climbing is a local search algorithm used for optimization problems. In our implementation, we use the Hill Climbing algorithm from the pgmpy library to find the most probable configuration of the Bayesian Network. The algorithm iteratively explores neighboring states and moves to the state with the highest probability until it reaches a local maximum.

## 3. Streamlit:

Streamlit is an open-source Python library that allows us to create interactive web applications for data science and machine learning projects. We use Streamlit to build a user-friendly interface for our diagnostic network AI, enabling users to input evidence, visualize results, and interact with the Bayesian Network.

# IDE Used

## Visual Studio Code

# Diagnostic Network AI - Setup and Installation Guide

This guide will walk you through setting up and running the Python-based Diagnostic Network AI on a new computer.

## 1. Install Python

First, you need to install Python from the official source.

- Go to the official Python website: [python.org](https://python.org)
- Download the latest stable version for your operating system (e.g., Python 3.12).
- **NOTE:** Make sure to download Python 3.x, as Python 2.x is no longer supported.
- **NOTE:** Python 3.14+ won't work with this codebase; please use Python 3.13 or earlier.

## 2. Run the Python Installer

Once the download is complete, run the installer file (e.g., `python-3.12.x.exe` on Windows or the `.pkg` file on macOS).

## 3. IMPORTANT: Add Python to PATH (Windows)

This is the most critical step for the installer.
On the very first screen of the Windows installer, look at the bottom.
You **MUST** check the box that says "Add python.exe to PATH".
If you miss this step, your computer won't know where to find Python in the terminal.
After checking the box, click "Install Now" and follow the prompts to complete the installation.
(For macOS and Linux, the installer typically handles this for you.)

## 4. Verify Python Installation

After installation, open your terminal (Command Prompt on Windows, Terminal on macOS/Linux) and type:

```bash
python --version
```

or

```bash
python3 --version
```

You should see the installed Python version displayed.

## 5. Create Virtual Environment (Optional but Recommended)

**Note:** If you prefer not to create a virtual environment, you can proceed directly to Step 6.

It's a good practice to create a virtual environment for your Python projects to manage dependencies.
In your terminal, navigate to the directory where you want to set up the Diagnostic Network AI and run:

```bash
python -m venv .venv
```

Activate the virtual environment:

- **On Windows:**

```bash
.venv\Scripts\activate
```

- **On macOS/Linux:**

```bash
source .venv/bin/activate
```

## 6. Install Dependencies

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

## 7. Run the Diagnostic Network AI

Now you can run the Diagnostic Network AI. In your terminal, navigate to the directory where the Diagnostic Network AI code is located and run:

```bash
python app.py
```

or

```bash
python3 app.py
```

This will start the Diagnostic Network AI Streamlit app.

Enjoy using the Diagnostic Network AI!