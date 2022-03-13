pyenv install -s 3.9.6
virtualenv -p ~/.pyenv/versions/3.9.6/bin/python --clear --always-copy --no-site-packages venv

source venv/bin/activate
pip3 install -r requirements_dev.txt

deactivate