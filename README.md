# *Sketch2RaBit*

A very simple sketch-based 3D modeling system built with the help of *[3DBiCar](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html)* and [*RaBit*](https://github.com/zhongjinluo/RaBit).

## Demo

https://github.com/zhongjinluo/Sketch2RaBit/assets/22856460/f19c3082-7d01-4205-a1da-1adc7e6a67b2

https://github.com/zhongjinluo/Sketch2RaBit/assets/22856460/331fa89b-e145-435b-824c-737381f94662

https://github.com/zhongjinluo/Sketch2RaBit/assets/22856460/8e5ee370-444b-463b-bf8c-cef781c88ead

## Usage

This system has been tested with Python 3.8, PyTorch 1.7.1, CUDA 10.2 on Ubuntu 18.04. 

- Installation:

  ```
  conda create --name Sketch2RaBit -y python=3.8
  conda activate Sketch2RaBit
  pip install -r requirements.txt
  ```

- Start by cloning this repo:

  ```
  git clone git@github.com:zhongjinluo/Sketch2RaBit.git
  cd Sketch2RaBit
  ```

- Download pre-compiled user interface and checkpoints for backend algorithms from [sketch2rabit_files.zip](https://cuhko365-my.sharepoint.com/:u:/g/personal/220019015_link_cuhk_edu_cn/EXxKE9ZsOxZJh-FtTUVsEJQBb31JXpl_gCiiRdJgZ2suPw?e=aOJFKF) and then:

  ```
  unzip sketch2rabit_files.zip
  unzip App.zip
  mv sketch2rabit_files/data/transfer_uv /path-to-repo/
  mv sketch2rabit_files/data/pose /path-to-repo/networks/v0/
  mv sketch2rabit_files/data/embedding/* /path-to-repo/networks/v0/embedding/
  mv sketch2rabit_files/data/pSp/* /path-to-repo/networks/v0/pSp/
  ```

- Run the backend server:

  ```
  cd /path-to-repo/ && bash server.sh
  ```
  
- Launch the user interface and enjoy it:

  ```
  cd App/ && bash run.sh
  ```

- If you want to run the backend algorithms on a remote server, you may have to modify  `App/config.ini`.





