## Verification and validation of WAMI moving vehicle detector

### Dependencies: 

1. DeepConcolic (https://github.com/TrustAI/DeepConcolic)
Folder "DeepConcolic/src/" is necessary.

2. Other packages including 

       conda create -n wami-env python=3
       
       conda activate wami-env
       
       conda install -c menpo opencv keras nltk matplotlib scikit-image nomkl
      
       pip install saxpy sklearn hdf5storage adversarial-robustness-toolbox


### To start the tool

                  python main.py 
                  
with a few optional paramters: 

        [--attack True/False] with default False
        [--wasabi-image-folder path/to/wasabi/image/folder] with default "../../../wasabi-detection-python-new/WAPAFB_images_train/training/"
        [--input_image_idx n] with default "10"
        [--ROI_centre (x,y)] with default "4500, 5000"
        [--output-image-folder path/to/output/folder] with default "../savefig/"
        [--ROI_window n] with default "1000"
        [--num_of_template n] with default "3"





