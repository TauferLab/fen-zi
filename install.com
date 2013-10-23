#!/bin/bash

mkdir -p build
cd build

../src/configure $* && echo &&
echo -n "Continue with this configuration? [Y/n] " &&
read ans && [ '!' "$ans" = "n" ] && [ '!' "$ans" = "N" ] &&
make && echo && echo "### Fenzi installation complete ###" && echo

