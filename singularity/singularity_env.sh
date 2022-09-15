#!/bin/bash

# get absolute path to this script
function get_path() {
  path=$(realpath "${BASH_SOURCE:-$0}")

  DIR_PATH=$(dirname "$path")

  echo "The absolute path is $path"
  echo "---------------------------------------------"
  echo "The Directory Path is $DIR_PATH"
}