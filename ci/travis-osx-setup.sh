if [ "$TRAVIS_OS_NAME" = "osx" ]; then
  echo "OSX Python Information"
  python3 --version
  which python3
  which pip3
  echo unlink /usr/local/bin/python
  unlink /usr/local/bin/python
  echo ln -s /usr/local/bin/python3 /usr/local/bin/python
  ln -s /usr/local/bin/python3 /usr/local/bin/python
  echo unlink /usr/local/bin/pip
  unlink /usr/local/bin/pip
  echo ln -s /usr/local/bin/pip3 /usr/local/bin/pip
  ln -s /usr/local/bin/pip3 /usr/local/bin/pip
fi
