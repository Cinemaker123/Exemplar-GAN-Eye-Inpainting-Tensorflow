## IMPORTS
#import face_recognition
import numpy as np
from PIL import Image, ImageDraw
#from fer import FER
import cv2
from scipy.spatial import distance as dist
import os
import shutil
import json
import random
import pickle
import getopt
import shutil
from os import listdir
from os.path import isfile, join

#from mainscripts import Extractor
#from DFLIMG import *
#from pathlib import Path

#from core import pathex
#from core import osex





#try:
#    multiprocessing.set_start_method('spawn')
#except RuntimeError:
#    pass
def main():

    ## CONSTANTS 

    FILE = "./output.txt" # where to save PA-GAN test file description
    OFFSET = 0.3 # how many percentage points should be cropped around face
    EYE_AR_THRESH = 0.35 # frames the eye must be below the threshold
    MAX_DISTANCE_BETWEEN_FACES = 100 # pixel euclidean distance between faces in two images
    TESTSTEP = 10000

    class FileManager:
        def __init__(self, input_folder, face_folder):
            
            
            
            #self.select_first_image()
            self.input_folder = input_folder
            self.output_folder = face_folder
            self.files = self.get_filenames(self.output_folder)
            #self.images = self.load_images(input_folder + "/" + files)
            

        def extract_images(self, files):
            try:
                shutil.rmtree(self.input_folder)
            except:
                print("Cant delete directory")

            try:
                os.mkdir(self.input_folder)
            except:
                print("Folder already exists")


            try:
                shutil.rmtree(self.output_folder)
            except:
                print("Cant delete directory")

            try:
                os.mkdir(self.output_folder)
            except:
                print("Folder already exists")

            print("Copying Files to Workspace")
            for file in files:
                shutil.copyfile(file, self.input_folder + "/" + os.path.basename(file))
        
            # Extract Faces from all images
            print("Beginning Extraction")
            self.face_extraction()
            print("Extraction Done")
    
        def setup_face_obj(self, files):
            self.face_obj = self.analyze_images()
            
        def load_image(self, image_file):
            return cv2.imread(image_file)

        def load_images(self, array_of_files):
            return [self.load_image(image_file) for image_file in array_of_files]

        def analyze_images(self):
            # should copy over files to new folder
            face_obj = []
            print("Starting Detector")
            self.detector = FER()
                
            aligned_filenames = self.get_filenames(self.output_folder)
            original_filenames = self.get_filenames(self.input_folder)

            # Do Face Recognition
            print("Beginning Face Locating")
            positions = self.face_locations(original_filenames)
            all_centers = {}
            for file, image in positions.items():
                centers = np.array([[(t+b)/2, (r+l)/2] for (t, r, b, l) in image])
                all_centers[file] = centers
            print("Locating Done")

            print("Beginning Object Creation")
            for face_file in aligned_filenames:

                dflimg = DFLJPG.load(face_file)
                src_filename = dflimg.get_source_filename()

                landmarks = dflimg.get_landmarks()

                (l, t, r, b) = dflimg.get_source_rect()
                center = np.array([(t+b)/2, (r+l)/2])
                index = self.find_nearest_face(center, all_centers[self.input_folder + "/" + src_filename])
                
                embeddings, emotions, box, top_emotion, top_score = self.analyze_image(face_file)

                obj = {
                    "filename": face_file,
                    "source_filename": src_filename,
                    "landmarks": landmarks,
                    "position": (t, r, b, l),
                    "position_face_recognition": positions[self.input_folder + "/" + src_filename][index],
                    "embedding": embeddings,
                    "emotions": emotions,
                    "box": box,
                    "top_emotion": top_emotion,
                    "top_score": top_score,
                    "eyes_open": self.eyes_open(landmarks),
                }
                face_obj.append(obj)
            print("Done!")
            
            return face_obj

        def analyze_image(self, face): 
            # Learn Face Embedding
            embeddings = self.learn_face_embeddings(face)
            # Do Emotion Recogntion
            emotions, box, top_emotion, top_score = self.emotion_recognition(face)
            
            return embeddings, emotions, box, top_emotion, top_score

        def learn_face_embeddings(self, face):
            embedding = None
            try:
                embedding = face_recognition.face_encodings(self.load_image(face))[0]
            except: 
                try: 
                    embedding = face_recognition.face_encodings(self.load_image(face), None, 5)[0]
                except:
                    print("Could not find face")
                    return np.zeros((128))
            return embedding

        def emotion_recognition(self, face):
            emotion_list = self.detector.detect_emotions(self.load_image(face))
            emotions = emotion_list[0]['emotions']
            box = emotion_list[0]['box']
            top_emotion = max(emotions, key=emotions.get)
            top_score = emotions[top_emotion]
            return emotions, box, top_emotion, top_score

        def face_locations(self, images):
            positions = {}
            for image in images:
                print(image)
                position_single_image = face_recognition.face_locations(self.load_image(image), 1, "hog") # "cnn" for GPU
                positions[image] = position_single_image

            return positions
            
        def face_extraction(self):
            osex.set_process_lowest_prio()
            Extractor.main( detector                = "s3fd",
                            input_path              = Path(self.input_folder),
                            output_path             = Path(self.output_folder),
                            output_debug            = None,
                            manual_fix              = False,
                            manual_output_debug_fix = False,
                            manual_window_size      = 1368,
                            face_type               = "whole_face",
                            max_faces_from_image    = 0,
                            image_size              = 512,
                            jpeg_quality            = 90,
                            cpu_only                = False,
                            force_gpu_idxs          = 0,
                        )
            
        def find_nearest_face(self, center, centers):
            return np.argmin(np.sqrt(np.sum((centers - center)**2, axis=1)))

        def get_filenames(self, folder):
            # load output folder and return all file names
            onlyfiles = [folder + "/" + f for f in listdir(folder) if f.split(".")[-1] == "jpg" and isfile(join(folder, f))]
            return onlyfiles
        
        def sort_faces_by_identity(self):
            dict_of_identities = {}
            total_embeddings = np.array(list(map(lambda x: x["embedding"], self.face_obj))) # np array so we can filter efficiently
            
            total_faces = np.array(self.face_obj)
            w = len(total_embeddings)
            checked_faces = []
            while len(total_embeddings) > 0 and w > 0:
                w = w - 1 # makes sure this loop finishes
                
                # check if the face has already been compared and selected
                if w in checked_faces:
                    continue
                checked_faces = np.append(checked_faces, w)

                source_face = total_faces[w]
                source_face_encoding = total_embeddings[w]
                
                similar_faces_value = np.array(list(face_recognition.face_distance(total_embeddings, source_face_encoding))) # compare this face with all other faces
                similar_faces = similar_faces_value <= 0.6 # this is the default value of the face_recognition library, can be changed at a later date  
                
                selected_faces = total_faces[similar_faces]
                indices_list = np.arange(len(total_faces))[similar_faces]
                # check that faces come from same area in all images
                # so we don't confuse e.g. two asian people standing on opposite sites of the image
                for face_obj, idx in zip(selected_faces, indices_list):
                    remove_face = False

                    # check if checked image is same as source image
                    if idx == w:
                        print("Skipping this image")
                        continue

                    # check if source images are different
                    source_file = source_face["source_filename"]
                    current_file = face_obj["source_filename"]
                    if source_file == current_file:
                        remove_face = True
                        print("Warning: Detected same face twice in one image")

                    # check if position in images are too far apart
                    source_pos = np.array([source_face["position"][0], source_face["position"][1]])
                    current_pos = np.array([face_obj["position"][0], face_obj["position"][1]])
                    if np.sqrt(((source_pos - current_pos)**2).sum()) > MAX_DISTANCE_BETWEEN_FACES: 
                        remove_face = True
                        print("Warning: Detected faces are far apart in images")

                    # remove index from similar_faces array
                    if remove_face:
                        similar_faces[idx] = False
                
                # add selected faces to checked_faces, so we don't check them twice
                indices = np.arange(len(total_embeddings))
                checked_faces = np.concatenate((checked_faces, indices[similar_faces]))
                
                selected_faces = total_faces[similar_faces]

                # assembling all face into one identity object
                dict_of_identities[str(w)] = selected_faces 
            self.dict_of_identities = dict_of_identities
            return dict_of_identities

        def eyes_open(self, landmarks):
            ear = self.eyes_open_score(landmarks)
            if ear < EYE_AR_THRESH:
                return False
            return True # Eyes open

        def eyes_open_score(self, landmarks):
            left_eye, right_eye = Helper().get_eye_points(landmarks)
            ear = self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)
            return ear

        def eye_aspect_ratio(self, eye):
            # compute the euclidean distances between the two sets of
            # vertical eye landmarks (x, y)-coordinates
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])

            # compute the euclidean distance between the horizontal
            # eye landmark (x, y)-coordinates
            C = dist.euclidean(eye[0], eye[3])

            # compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)

            # return the eye aspect ratio
            return ear
    

        # p = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
        # p.add_argument('--detector', dest="detector", choices=['s3fd','manual'], default=None, help="Type of detector.")
        # p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
        # p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted files will be stored.")
        # p.add_argument('--output-debug', action="store_true", dest="output_debug", default=None, help="Writes debug images to <output-dir>_debug\ directory.")
        # p.add_argument('--no-output-debug', action="store_false", dest="output_debug", default=None, help="Don't writes debug images to <output-dir>_debug\ directory.")
        # p.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'whole_face', 'head', 'mark_only'], default=None)
        # p.add_argument('--max-faces-from-image', type=int, dest="max_faces_from_image", default=None, help="Max faces from image.")    
        # p.add_argument('--image-size', type=int, dest="image_size", default=None, help="Output image size.")
        # p.add_argument('--jpeg-quality', type=int, dest="jpeg_quality", default=None, help="Jpeg quality.")    
        # p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="Enables manual extract only frames where faces were not recognized.")
        # p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False, help="Performs manual reextract input-dir frames which were deleted from [output_dir]_debug\ dir.")
        # p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1368, help="Manual fix window size. Default: 1368.")
        # p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU..")
        # p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")

        
    class FaceImprover():
        def __init__(self, subset_faces, original_image_index):
            self.faces = subset_faces
            self.original_image_index = original_image_index

        def improve_face(self):
            pass        

            return improved_faces

        def select_best_face(self):
            pass
            return index

        def select_happiest_face(self, array):
            
            emotion_score = self.get_emotion_score(array)
            happiest_face_idx = np.argmax(emotion_score)

            return happiest_face_idx

        def select_eyes_opened_face(self):
            pass
            return index

        def get_emotions(self, array):
            value_array = np.array(list(map(lambda x: x["emotions"], array)))
            return value_array

        # calculates a weighted score of how good a person looks,
        # higher is more happy
        def get_emotion_score(self, array):
            ANGRY = -1
            DISGUST = -1
            FEAR = -1
            HAPPY = 2
            NEUTRAL = 1
            SAD = -1
            SURPRISE = -1

            value_array = []

            emotions_array = self.get_emotions(array)
            for emotions in emotions_array:
                score = (emotions["angry"]*ANGRY + emotions["disgust"]*DISGUST + emotions["fear"]*FEAR +
                        emotions["happy"]*HAPPY + emotions["neutral"]*NEUTRAL + emotions["sad"]*SAD + 
                        emotions["surprise"]*SURPRISE) 
                value_array.append(score)

            return value_array

        def get_eye_open_score(self, array):
            value_array = np.array(list(map(lambda x: x["eyes_open_score"])))
            return value_array

    class Helper():
        def get_eye_points(self, landmarks):
            left_eye = landmarks[36:41+1]
            right_eye = landmarks[42:47+1]
            return left_eye, right_eye

    class NeuralNets():
        def __init__(self, output_folder):
            self.output_folder = output_folder

        # passes the primary face and several faces to Eye In-Painting
        # generates the txt file describing the images and masks
        def generate_new_eyes(self, face_images):
            print("Preparing In-Painting...")
            # prepare inpainting_json
            inpainting_json = {}
            
            for key, face_obj in face_images.items():
                inpainting_json.update(self.format_one_inpainting_celeb(key, face_obj))

            with open(self.output_folder + '/data.json', 'w') as outfile:
                json.dump(inpainting_json, outfile)

            # launch inpainting test
            CHECKPOINT_RELATIVE = "./../../Desktop"
            os.system("python ../../../Desktop/ExemplarV2/main.py --OPER_FLAG=1 --path {} --is_load=True --test_step={} --notTest=False --checkpoint_relative={} --batch_size=1".format(self.output_folder, TESTSTEP, CHECKPOINT_RELATIVE))
        
            # collect resulting images
            
            pass

        def format_one_inpainting_celeb(self, name, face_obj_list):
            print("Formatting In-Painting JSON for {}".format(name))
            lst = []
            for face_obj in face_obj_list:
                data_obj = self.format_one_inpainting_face(face_obj)
                lst.append(data_obj)
            return {name: lst}


        def format_one_inpainting_face(self, face_obj):
            landmarks = face_obj["landmarks"]
            eye_left, box_left, eye_right, box_right = self.get_eye_points(landmarks)
            data_obj = {
                "filename": face_obj["filename"].split("/")[-1],
                "eye_left": eye_left,
                "box_left": box_left,
                "eye_right": eye_right,
                "box_right": box_right
            }
            if face_obj["eyes_open"]:
                data_obj["opened"] = True
                data_obj["closed"] = None
            else:
                data_obj["opened"] = None
                data_obj["closed"] = True
            return data_obj

        # takes an array of landmark points and gets eye position and a box around the eyes        
        def get_eye_points(self, landmarks):
            #left_eye = landmarks["left_eye"]
            #right_eye = landmarks["right_eye"] #replace with array expression if not correct

            left_eye, right_eye = Helper().get_eye_points(landmarks)

            eye_left_tuple = np.round(np.array(left_eye).sum(axis=0)/6, 0).astype(int)
            eye_right_tuple = np.round(np.array(right_eye).sum(axis=0)/6, 0).astype(int)

            eye_left = {"x": int(eye_left_tuple[0]), "y": int(eye_left_tuple[1])}
            eye_right = {"x": int(eye_right_tuple[0]), "y": int(eye_right_tuple[1])}
            
            box_left = {
                "w": int(abs(left_eye[0][0] - left_eye[3][0])),
                "h": int(abs(left_eye[2][1] - left_eye[5][1]))
            }
            
            box_right = {
                "w": int(abs(right_eye[0][0] - right_eye[3][0])),
                "h": int(abs(right_eye[2][1] - right_eye[5][1]))
            }

            return eye_left, box_left, eye_right, box_right

        def generate_new_mouth(self, primary_face, face_images):
            pass

        def poisson_blending(self, face, new_face, mask):
            pass
        

    image_list = ["group1.jpg", "group2.jpg"]
    image_list = ["input/group3.jpg", "input/6833864256_354f1b71c5_k.jpg"]
    image_list = ["input/group1.jpg", "input/group3.jpg"]

    WORKSPACE = "pipeline_tests"
    ALIGNED_FACES = "pipeline_tests/aligned"
    
    obj = FileManager(WORKSPACE, ALIGNED_FACES)
    #obj.extract_images(image_list)
    obj.setup_face_obj(image_list)
    idens = obj.sort_faces_by_identity()
    for d in idens.keys():
        print(d)
    subset = idens#["3"]

    #NeuralNets(ALIGNED_FACES).generate_new_eyes(subset)
    print("Done")
    print(obj.face_obj[0]["landmarks"])
    print(obj.face_obj[0]["filename"])

if __name__ == "__main__":
    # Fix for linux
    import multiprocessing
    multiprocessing.set_start_method("spawn")#, force=True)
#    multiprocessing.freeze_support()

    from core.leras import nn
    nn.initialize_main_env()
    main()

