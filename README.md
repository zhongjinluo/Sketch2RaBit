# *Sketch2RaBit*

A very simple sketch-based 3D modeling system built with the help of *[3DBiCar](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html)* and [*RaBit*](https://github.com/zhongjinluo/RaBit).

## Demo

https://github.com/zhongjinluo/Sketch2RaBit/assets/22856460/4776d8eb-b5ea-4fdd-b4b8-5e040d6df7dc

https://github.com/zhongjinluo/Sketch2RaBit/assets/22856460/b3ae5573-de9a-4cea-9b35-f50a347eb3c2

https://github.com/zhongjinluo/Sketch2RaBit/assets/22856460/e8bde34c-6ed2-4811-ab90-cb645f31fa88

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

- Download pre-compiled user interface and checkpoints for backend algorithms from [sketch2rabit_files.zip](https://cuhko365-my.sharepoint.com/:u:/g/personal/220019015_link_cuhk_edu_cn/EfrbrGVpsUlDuE4zZDfsJlIB-QzgUDb9GZO9MInG0ecWkQ?e=IQH7rJ) and then:

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





