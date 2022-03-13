python3 -m unittest
rm -r dist build recallme.egg-info
python3 setup.py clean
python3 setup.py sdist bdist_wheel
twine upload dist/*
rm -r dist