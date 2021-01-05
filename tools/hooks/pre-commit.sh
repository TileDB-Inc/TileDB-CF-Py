#!/bin/sh

# Function to automate running linting/formatting tests.
run_test() {
    name=$2
    fix_msg=$3
    echo "* Running ${name}.. "
    echo "..................."
    $1
    status=$?
    echo "..................."
    if [ $status -ne 0 ]; then
	read -r -p "..failed. Would you like continue with commit? [y/N] " response
	case "$response" in
	    [yY][eE][sS]|[yY])
		echo "Continuing with tests .."
		;;
	    *)
		echo $fix_msg
		exit $status
	esac
    else
	echo "..passed"
    fi
}

# get all python files that aren't deleted
python_files=$(git diff --cached --name-only --diff-filter=AM | grep '\.py$')

if [ ! -z "${python_files}" ]; then
    # run isort
    run_test "poetry run isort --check --diff ${python_files}" \
	     "isort" \
	     "Try running 'poetry run isort .' and add changes to git."
    # run black
    run_test "poetry run black --check ${python_files}" \
	     "black" \
	     "Try running 'poetry run black .' and add changes to git."
    # run flake8
    run_test "poetry run flake8 ${python_files}" "flake8" ""
    # run mypy
    run_test "poetry run mypy ${python_files}" "mypy" ""
fi

# Check for whitespace errors
if git rev-parse --verify HEAD >/dev/null 2>&1
then
    against=HEAD
else
    # Initial commit: diff against an empty tree object
    against=$(git hash-object -t tree /dev/null)
fi

exec git diff-index --check --cached $against --
