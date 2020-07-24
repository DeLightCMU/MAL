rm -r build
rm -r maskrcnn_benchmark.egg-info
rm ./maskrcnn_benchmark/*.so
python setup.py clean
python setup.py build develop

