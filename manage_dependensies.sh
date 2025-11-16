#!/bin/bash

print_usage() {
    echo "Usage: $0 [install|update|help|-h|--help]"
    echo
    echo "Commands:"
    echo "  install    Create a new virtual environment and install dependencies"
    echo "  update     Update dependencies in the existing virtual environment"
    echo "  help       Show this help and exit"
    echo "  -h, --help Show this help and exit"
}

if [ "$#" -ne 1 ]; then
    print_usage
    exit 1
fi

action=$1
venv_dir="venv"

if [ "$action" == "-h" ] || [ "$action" == "--help" ] || [ "$action" == "help" ]; then
    print_usage
    exit 0
fi

os=$(uname)
if [[ "$os" == "Darwin" || "$os" == "Linux" ]]; then
    activate_script="$venv_dir/bin/activate"
    python_cmd="python3.12"
else
    # otherwise we assume Windows (Git Bash)
    activate_script="$venv_dir/Scripts/activate"
    python_cmd="python"
fi

# Check that the chosen Python exists (mainly for macOS/Linux)
if ! command -v "$python_cmd" >/dev/null 2>&1; then
    echo "Error: '$python_cmd' not found."
    if [[ "$os" == "Darwin" || "$os" == "Linux" ]]; then
        echo "On macOS with Homebrew, you can install it with:"
        echo "  brew install python@3.12"
        echo "Then re-run this script."
    fi
    exit 1
fi

if [ "$action" == "install" ]; then
    # If a venv already exists, remove it
    if [ -d "$venv_dir" ]; then
        echo "Removing existing virtual environment..."
        rm -rf "$venv_dir"
    fi

    echo "Creating a new virtual environment using $python_cmd..."
    "$python_cmd" -m venv "$venv_dir"

    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$activate_script"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

elif [ "$action" == "update" ]; then
    if [ ! -d "$venv_dir" ]; then
        echo "Virtual environment not found. Please run '$0 install' first."
        exit 1
    fi

    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$activate_script"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Updating dependencies from requirements.txt..."
    pip install --upgrade -r requirements.txt

else
    echo "Invalid argument: $action"
    print_usage
    exit 1
fi