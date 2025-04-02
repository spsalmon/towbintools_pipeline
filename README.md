# TOWBINTOOLS PIPELINE

This is still in early development, hopefully in the future it will replace all of the lab's dozens of different bash scripts laying around and give the opportunity to everyone to easily customize their image analysis pipeline.

The documentation for the package used as a backbone for the pipeline can be found here : <https://towbintools.readthedocs.io/en/latest/towbintools.html>

## RTFM

There is a small wiki available here : <https://github.com/spsalmon/towbintools_pipeline/wiki/Building-Block> that should cover the basis of how to write your first configuration file.
If you don't understand something, feel free to ask, and I'll update the wiki to make it clearer !

## How to install ?

### How to set up Visual Studio Code ?

1. Download VS Code : <https://code.visualstudio.com/download>
2. Install it like you would install any software.
3. Inside of VS Code, open a terminal and run :

```bash
code --install-extension ms-vscode-remote.remote-ssh
```

Now, click on the remote explorer icon that should be on the left of the window and click on the + to add a new remote.
Enter the command you usually use to ssh into the cluster using PuTTY, for example:

```bash
ssh username@izblisbon.unibe.ch
```

Obviously, change username to your username (first letter of your first name + last name, eg : spsalmon)

Optionnal, but **HIGHLY** recommended. Open the Windows command line (cmd). Run :

```bash
ssh-keygen
```

- Select all the default options, except if you are extremely paranoid and want to set a passphrase.
  Go into the folder where the file was saved, it should be something like Users/username/.ssh/

- Open the file **id_rsa.pub** using the notepad or any text editing software.
  Copy the entire content of the file.

- In VS Code, go to your home folder : /home/username/

- Go into the .ssh folder

- If it doesn't exist, create a file named **authorized_keys**

- Paste the content of the **id_rsa.pub** file that you copied earlier into this file

- You will now be able to connect to the cluster without having to type your password

If you want to code using Python, you should run the following commands, while connected inside of VS Code, while being connected to your session on the cluster.

```bash
code --install-extension ms-python.python
```

```bash
code --install-extension ms-toolsai.jupyter
```

```bash
code --install-extension ms-python.vscode-pylance
```

### How do to install the pipeline itself

- In VS Code, open a terminal and cd to your home directory :

```bash
cd
```

- Clone the pipeline repo from github :

```bash
git clone https://github.com/spsalmon/towbintools_pipeline.git
```

- Install micromamba and restart your shell :

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

```bash
source ~/.bashrc
```

- Run the installation script :

```bash
chmod +x ~/towbintools_pipeline/pipeline_instalation_script.sh
```

```bash
~/towbintools_pipeline/pipeline_instalation_script.sh
```

For some reason, the script doesn't really work for some people. In case it doesn't work for you, just run every line of the installation script manually.

Follow the directions given, so basically, push enter a bunch of times and type yes (you want to answer yes everytime) when asked to.

## Running the pipeline

1. Read the WIKI !!!!!
2. Modify your config.yaml file according to what you want to do
3. run the following command

```bash
cd ~/towbintools_pipeline
```

```bash
bash run_pipeline.sh
```

### Using a custom config file

If you don't specify anything, the config file used will be "./config.yaml" but you can specify one using

```bash
bash run_pipeline.sh -c path_to_config_file
```

or

```bash
bash run_pipeline.sh --config path_to_config_file
```

## Updating the pipeline

I will update the pipeline frequently, so please try to keep up to date by running the following commands on a regular basis.

### Update script

Running the following script will update both the pipeline and the underlying towbintools python package

```bash
bash update_pipeline.sh
```

### If you want to reset the folder completely (manually)

First, fetch the repository

```bash
git fetch origin
```

Then, run :

```bash
git reset --hard origin/main
```

### If you want to save some changes you made

Run those commands before resetting the folder :

```bash
git commit -a -m "Saving my work, just in case"
git branch my-saved-work
```

This way, your changes will be changed into a new branch. Note that this branch will to be updated. Overall, I would advise to not modify anything directly. Copy your config file(s) to a backup folder and just reset the folder everytime you want to update it.

### Updating the package

It's also good to update the towbintools package from time to time even if the bugs and changes will be less frequent as it is more stable, to do so :

First activate the environment :

```bash
micromamba activate towbintools
```

Then upgrade the package :

```bash
pip install --upgrade towbintools
```
