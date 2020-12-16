if [[ ${USE_CONDA} == "true" ]]; then
  echo "Updating the path for conda"
  echo "##vso[task.setvariable variable=PATH]${HOME}/miniconda3/bin:${PATH}"
  export PATH=${HOME}/miniconda3/bin:$PATH
  which python
fi
