from setuptools import find_packages, setup
from typing import List

requriment_file_name = "requirements.txt"
REMOVE_PACKAGE = "-e ."

def get_requirements()->List[str]:
    with open(requriment_file_name) as requirement_file:
        requriment_list = requirement_file.readline()
    requriment_list = [requriment_name.replace("\n", "") for requriment_name in requriment_list]

    if REMOVE_PACKAGE in requriment_list:
        requriment_list.remove(REMOVE_PACKAGE)
    return requriment_list



setup(name='Car_parking_system',
      version='0.0.1',
      description='car parking project',
      author='Deep pratap',
      author_email='deeppratapsingh0996@gmail.com',
      packages=find_packages(),
      install_reqires = get_requirements()
     )