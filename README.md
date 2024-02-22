# Deep Learning for Paintings Authorship Prediction

Artificial intelligence techniques, particularly deep learning networks, have found widespread application in addressing various computational vision challenges. Motivated by the potential of neural networks to discern patterns in images, I endeavored to create a deep learning model aimed at identifying the authors of renowned paintings.

To provide context for the problem domain, I highlight the following points:

- **Assumption:** Each artist has their own style that a deep learning algorithm can learn and then generalize to other works by the same author.


- **Challenge:** Recognizing a painting can be a difficult task, as there are artists who tend to change their style or simply switch between several different styles.

Within this computational notebook, you have the opportunity to construct a model for predicting painting authors. Simply follow the logical sequence of cells, and feel free to adjust the parameters of the displayed variables as needed.

## Project structure

This project follows a structured organization to facilitate clarity and collaboration. Below is an overview of the folder structure:

**Folder Descriptions:**

- **`src`:** folder with the artifacts needed to run the notebook (images, trained models, code). Inside this folder we have the following structure.
  - **`images`:** folder that contains the training and testing images (paintings) of each artist.
  - **`models`:** folder with the files of the models that have already been trained and saved for later use.
  - **`init.py`:** python file required to run the program.
  - **`config.py`:** python file with the main program settings (image path, description messages, model architecture).
  - **`dataset.py`:** python file with the code that handles the images that are used to train and evaluate the model.
  - **`modelo.py`:** python file with the code that creates and evaluates a model, in addition to launching an application to predict authors of certain paintings.

- **`predict_authors.ipynb`:** jupyter notebook that contains functionality such as exploring images, training and evaluating models, and launching an application to use a given prediction model.

- **`header.jpg`:** jupyter notebook header image. If you want to modify the header image, simply replace this image and keep the name.
- **`requirements.yml`:** file with the dependencies that were installed to run the jupyter notebook. Please do not modify this file.

- **`README.md`:** Project overview, instructions, and explanations.

Feel free to modify the structure based on the specific needs of your machine learning project.


## Installation

To install this project run

### Step 1: Installing miniconda

Miniconda is a simplified tool used to create and manage isolated software development environments, install and update packages and libraries in the Python language. To install Miniconda on Windows, follow these steps:

#### Download the Installer:

- Go to the Miniconda download page: https://docs.conda.io/en/latest/miniconda.html
- Choose the installer for the version of Windows you are using (usually the 64-bit version).
- Click the link to download the executable installer.

#### Follow the Installation Wizard:

The installer will display a wizard to guide you through the installation process:
- Accept the License Agreement: Read and accept the license agreement.
- Choose an Installation Destination: Choose a directory to install Miniconda. The default directory is generally fine.
- Select Installation Options:
  - Choose "Just Me" to install just for your user.
  - Select the "Add Anaconda to my PATH environment variable" option to add Miniconda to your PATH. This is recommended to make it easier to use Conda from the Command Prompt.
Complete Installation: Click the "Install" button to begin the installation.

#### Open Command Prompt:

After installation, you can open Command Prompt and check if Miniconda installed correctly. Run the following command:

```bash
conda -- version
```

If Miniconda was installed successfully, you will see the version of Conda installed. Once this is done, we will create a virtual environment to manage the packages that our program needs to run.

### Step 2: Creating a virtual environment

A Conda virtual environment is an isolated, self-contained environment in which you can install and manage packages and libraries specific to a particular project or task. To create a virtual environment in Miniconda, follow these steps:

#### Get the folder path:

Once you are inside the folder where the program files are, you will need to get the full folder path. You can do this by right-clicking the folder while holding down the ‘Shift’ key and selecting ‘Copy as path’ from the context menu. This will copy the folder path to the clipboard.

#### Open Command Prompt:

Press the Windows key to open the Start menu, type “cmd” and press “Enter”. This will open a command prompt window.

#### Navigate to the folder:

In the command prompt window, type “cd” (with a space after cd) and right-click to paste the folder path from the clipboard. Press “Enter”. For example: cd C:\Path\To\Your\Folder

#### Create the Virtual Environment:

In Command Prompt, run the following command to create the virtual environment from the requirements.yml file.

```bash
conda create --name myenv --file requirements.yml
```

#### Activate the Virtual Environment:

After creating the virtual environment, you need to activate it. Use the following command:

```bash
conda activate myenv
```

Now you have a virtual environment created and activated in Miniconda. If you don't want to activate the virtual environment every time you start a Jupyter Notebook, run the following command:

#### Install ipykernel:

```bash
conda install ipykernel
```

#### Create a custom kernel:

While your virtual environment is active, run the following command to create a custom Jupyter kernel associated with your environment:

```bash
python -m ipykernel install --user --name myenv --display-name "My custom name"
```

### Step 3: Launch Jupyter Notebook:

After creating the custom kernel, you can launch Jupyter Notebook. Run the command:

```bash
jupyter notebook
```

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Author

- [@hrguarinv](https://github.com/hrguarinv)


## 🚀 About Me
I am particularly enthusiastic about developing data-driven solutions and integrating them with SE best practices to create efficient and scalable systems. Having worked in diverse environments, including finance and oil & gas sectors in Colombia and Brazil, I have improved my skills in adapting to different industry requirements and ensuring the delivery of high-quality software solutions.
