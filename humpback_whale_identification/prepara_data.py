import sys, os
from glob import glob
import pandas as pd
# sys.path.append(os.getcwd() + "/models/keras_retinanet")
# sys.path.insert(0, "")

from models.retinanet.keras_retinanet.preprocessing import pascal_voc

data_dir = os.path.join(os.getcwd(), 'data')
train_image_dir = os.path.join(data_dir, 'train/')
print(data_dir)


# generate train images txt
sample_xml_files = glob(os.path.join(data_dir, 'Annotations/*'))
annotations = [train_image_dir + file.split('/')[-1].split('.')[0] for file in sample_xml_files]
print("Generating samples...")
pd.DataFrame(annotations).to_csv(os.path.join(data_dir, 'samples.csv'), header=False, index=False)
print("Finished samples!")

# cp the corresponding images
all_images = glob(os.path.join(data_dir, 'train/*'))
print("Copy sample images to JPEGImage dir...")
for sample_path in sample_xml_files:
    img_name = sample_path.split('/')[-1].split('.')[0]
    source_path = train_image_dir + img_name + '.jpg'
    dest_path = data_dir + '/JPEGImages'
    os.system('cp ' + source_path + ' ' + dest_path)

# generate annotation text
classes = {'whale': 0}
generator = pascal_voc.PascalVocGenerator(data_dir=data_dir, set_name='samples', classes=classes)
image_names = generator.image_names
train_annotations = []
valid_annotations = []
train_size = 0.8 * generator.size()
print("Generating annotations for training and validation...")
for i in range(generator.size()):
    # load the data
    annotation = generator.load_annotations(i)
    image_name = image_names[i]
    img_path = train_image_dir + image_name + '.jpg'
    bbox = annotation['bboxes'][0]
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if i < train_size:
        train_annotations.append([img_path, xmin, ymin, xmax, ymax, 'whale'])
    else:
        valid_annotations.append([img_path, xmin, ymin, xmax, ymax, 'whale'])
pd.DataFrame(train_annotations).to_csv(os.path.join(data_dir, 'sample_train_annotations.csv'), header=False, index=False)
pd.DataFrame(valid_annotations).to_csv(os.path.join(data_dir, 'sample_valid_annotations.csv'), header=False, index=False)

if __name__ == "__main__":
    print("Finished!")
    # print sample_xml_files[0]
    # print Annotations[0]
    # os.system('ls -l')