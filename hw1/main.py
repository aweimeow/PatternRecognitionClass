#! /usr/bin/python3

import cv2
import sys
import os


class FaceDetect:

    def __init__(self, directory):
        self.subname = ['jpg', 'jpeg', 'png', 'gif']
        self.directory = directory[:-1] if '/' == directory[-1] else directory
        self.Cascade = cv2.CascadeClassifier(
                                    'haarcascade_frontalface_default.xml')
        self.files = [x for x in os.listdir(self.directory)
                      if x.split('.')[-1].lower() in self.subname]
        self.process()

    def process(self):
        for image_name in self.files:
            self.image_name = image_name
            self.image = cv2.imread('{0}/{1}'.format(
                                     self.directory, image_name))
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            face = self.Cascade.detectMultiScale(
                    self.gray_image,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                   )
            self.save_file(face)
            # If you want to debug, uncomment this line
            # self.demo_or_debug(face)

    def demo_or_debug(self, face_list):
        for (x, y, w, h) in face_list:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (128, 255, 0), 2)
        cv2.namedWindow("DEMO OR DEBUG", cv2.WINDOW_NORMAL)
        cv2.imshow('DEMO OR DEBUG', self.image)
        cv2.waitKey(0)

    def save_file(self, face_list):
        index = 0
        for (x, y, w, h) in face_list:
            filename = '{0}_{2}.{1}'.format(
                    *self.image_name.split('.') + [index])
            path = './output/' + filename
            cv2.imwrite(path, self.gray_image[y:y + h, x:x + w])
            print('File: %s writed' % path)
            index += 1

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 %s <input_file_dir>' % sys.argv[0])
        sys.exit()
    filepath = sys.argv[1]
    FaceDetect(filepath)
