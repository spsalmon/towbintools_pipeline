# TOWBINTOOLS PIPELINE

Towbintools Pipeline is a pipeline for processing and analyzing time lapse microscopy experiments. It implements many of the functions present in the towbintools package and bundles them with a pipelining tool to easily and reproducibly process large experiments.

A detailed documentation of the pipeline can be found here : <https://spsalmon.github.io/towbintools_pipeline/>
The documentation for the package used as a backbone for the pipeline can be found here : <https://towbintools.readthedocs.io/en/latest/towbintools.html>

## RTFM

## How to install ?

You will find detailed explanations on how to install, update and use the pipeline here : <https://spsalmon.github.io/towbintools_pipeline/getting-started/installation/>

## Running the pipeline

You will find a detailed explanation on how to run the pipeline here : <https://spsalmon.github.io/towbintools_pipeline/getting-started/runningfirstpipelinee/>

## Updating the pipeline

You will find a detailed explanation on how to update the pipeline here : <https://spsalmon.github.io/towbintools_pipeline/getting-started/update/>

## How to set up Visual Studio Code ?

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
