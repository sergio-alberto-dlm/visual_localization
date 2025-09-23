submodules_path=$(dirname $(dirname $(realpath $BASH_SOURCE)))/submodules
export PYTHONPATH="${PYTHONPATH}:$submodules_path/vggt"