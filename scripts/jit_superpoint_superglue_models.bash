#!/usr/bin/env bash

git submodule deinit -f .
git submodule update --init --recursive

readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly ROOT_DIR=$(realpath "$CURRENT_DIR"/..)

cd $ROOT_DIR/scripts/superglu