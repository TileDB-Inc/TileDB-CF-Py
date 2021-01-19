#!/bin/sh

# Function to automate running linting/formatting tests.
ask_run_tool() {
    name=$2
    read -r -p "Run ${name}? [Y/n] " response
    case "$response" in
	[nN][oO]|[nN])
	    echo "* Skipping ${name}"
	    ;;
	[yY][eE][sS]|[yY]|"")
	    echo "* Running ${name}.. "
	    echo "..................."
	    $1
	    echo "..................."
	    ;;
	*)
	    echo "Not a valid response. Skipping ${name}."
    esac
}

project_root=$(git rev-parse --show-toplevel)
source_dir="${project_root}/tiledb"
test_dir="${project_root}/tests/tiledb/cf"

ask_run_tool "poetry run isort ${project_root}" "isort"
ask_run_tool "poetry run black ${project_root}" "black"
ask_run_tool "poetry run flake8 ${project_root}" "flake8"
ask_run_tool "poetry run mypy ${source_dir} ${test_dir}" "mypy"
ask_run_tool "poetry run pytest --cov=${source_dir} ${test_dir}" "pytest"
