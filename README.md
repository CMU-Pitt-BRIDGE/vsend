# VSend

Updated the scripts `murfi2/util/python/receive_vsend/receive_nii.py` and `murfi2/util/python/receive_vsend/external_image.py` at <https://github.com/gablab/murfi2>, written in python 2.7.18, to be compatible with python 3.12.

## Installation

### Pyenv

```shell
brew update
brew install pyenv
```

or

```shell
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
curl https://pyenv.run | bash
```

Pyenv setup

```shell
export PATH="$HOME/.pyenv/bin:$PATH" && \
eval "$(pyenv init --path)" && \
if which pyenv-virtualenv-init > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi
```

Install Python with pyenv

```shell
pyenv install 3.12 && \
pyenv global 3.12
```

### Poetry

```shell
export POETRY_HOME=/opt/poetry && \
sudo mkdir -p $POETRY_HOME && \
sudo chown -R $(whoami):$(id -gn) $POETRY_HOME && \
export XDG_CACHE_HOME=${POETRY_HOME}/cache && \
mkdir -p $XDG_CACHE_HOME && \
chown -R $(whoami):$(id -gn) $XDG_CACHE_HOME && \
export XDG_CONFIG_HOME=${POETRY_HOME}/config  && \
mkdir -p $XDG_CONFIG_HOME && \
chown -R $(whoami):$(id -gn) $XDG_CONFIG_HOME && \
export PATH="$POETRY_HOME/bin:$PATH" && \
export POETRY_VIRTUALENVS_IN_PROJECT=true && \
pip install --upgrade pip && \
curl -sSL https://install.python-poetry.org | python3 - && \
poetry config virtualenvs.create true && \
poetry config virtualenvs.in-project true
```

### Installing requirements for vsend package

```shell
poetry lock --no-update
poetry install
```

```shell
poetry shell
```

## Testing VSEND

Run the script `./vsend/receive_nii.py`, using the following flags:

- `-H`: Set to the IP address of the computer
- `-p`: Set to the port to listen
- `-d`: the directory to save the imaging data to

e.g:

```shell
python vsend/receive_nii.py \
  -H 192.168.2.5 \
  -p 50000 \
  -d ./received_data
```
