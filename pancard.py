from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import re
import pandas as pd
import math
import cv2
import fitz
import shutil
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import words
from Levenshtein import distance
import warnings

warnings.simplefilter('ignore')
from logs import logger
import os

os.environ['DOCTR_MULTIPROCESSING_DISABLE'] = 'TRUE'
os.environ['DOCTR_CACHE_DIR'] = 'weights/'


def load_model():
    try:
        logger.debug('Loading the model load_model()')
        model = ocr_predictor(det_arch='db_resnet50',
                              reco_arch='crnn_vgg16_bn',
                              pretrained=True,
                              detect_language=True
                              )
        return model
    except:
        logger.error('Failed to load the model from load_model()')


def match_address(keyword, text):
    # Compare the keyword "address" with the returned string
    if not text:
        return None
    words = nltk.word_tokenize(text)
    for word in words:
        # Compare the keyword "address" with each word in the text
        score = distance(keyword, word[-4:])
        # If the score is less than a certain threshold, consider it a match
        if score <= 2:
            return word
    return False


def text_extract(path):
    try:
        logger.debug('Extracting the text text_extract()')
        dir = os.listdir(path)
        if dir[0].endswith('pdf'):
            img = DocumentFile.from_pdf(path + '/' + dir[0])
        else:
            img = DocumentFile.from_images(path + '/' + dir[0])
        model = load_model()
        result = model(img)
        output = result.export()
        text = result.render()
        check = text.lower().__contains__('income')
        if not check:
            logger.error('The provided input is not correct document text_extract()')
            return None, None
        return text, output
    except:
        logger.error('Getting issue in extracting the text text_extract()')


def convert_coordinates(geometry, page_dim):
    len_x = page_dim[1]
    len_y = page_dim[0]
    (x_min, y_min) = geometry[0]
    (x_max, y_max) = geometry[1]
    x_min = math.floor(x_min * len_x)
    x_max = math.ceil(x_max * len_x)
    y_min = math.floor(y_min * len_y)
    y_max = math.ceil(y_max * len_y)
    return [x_min, x_max, y_min, y_max]


def get_coordinates(output):
    try:
        logger.debug('Getting the co-ordinates for text get_coordinates()')
        page_dim = output['pages'][0]["dimensions"]
        text_coordinates = []
        for obj1 in output['pages'][0]["blocks"]:
            for obj2 in obj1["lines"]:
                for obj3 in obj2["words"]:
                    converted_coordinates = convert_coordinates(
                        obj3["geometry"], page_dim
                    )
                    text_coordinates.append([converted_coordinates, obj3["value"]])
        return text_coordinates
    except:
        logger.error('Getting issue in getting the coordinates from text get_coordinates()')


def convert_image(path):
    try:
        logger.debug('Convert Scanned Pdf to images convert_image()')
        # To get better resolution
        zoom_x = 2.0  # horizontal zoom
        zoom_y = 2.0  # vertical zoom
        mat = fitz.Matrix(zoom_x, zoom_y)
        dir = os.listdir(path)
        doc = fitz.open(path + '/' + dir[0])
        folderpath = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        for page in doc:  # iterate through the pages
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            pix.save("temp/page-%i.png" % page.number)
        ndir = os.listdir(folderpath)
        fullpath = folderpath + '/' + ndir[0]
        return fullpath
    except:
        logger.error('Getting issue in converting to image converting_image()')


def text_layout(path):
    try:
        logger.debug('Sorting the text based on co-ordinates text_layout()')
        text, output = text_extract(path)
        if not text and not output:
            logger.error('The provided input is not correct document text_layout()')
            return None
        result = get_coordinates(output)
        x, w, y, h, text = [], [], [], [], []
        df_left = pd.DataFrame(columns=['x', 'w', 'y', 'h', 'text'])
        dir = os.listdir(path)
        if dir[0].endswith('pdf'):
            img_path = convert_image(path)
            img = cv2.imread(img_path)
            shutil.rmtree('temp/')
        else:
            img = cv2.imread(path + '/' + dir[0])
        split = img.shape[1] // 2
        for i in range(len(result)):
            if result[i][0][0] < split:
                x.append(result[i][0][0])
                w.append(result[i][0][1])
                y.append(result[i][0][2])
                h.append(result[i][0][3])
                text.append(result[i][1])
        df_left['x'] = x
        df_left['w'] = w
        df_left['y'] = y
        df_left['h'] = h
        df_left['text'] = text

        left_side = ''
        if not df_left.empty:
            df1_left = df_left.sort_values(by=['y', 'x'])
            df1_left.dropna(inplace=True)

            df2_left = df1_left[df1_left['text'].str.strip().astype(bool)]

            df2_left.reset_index(drop=True, inplace=True)

            for i in range(len(df2_left) - 1):
                if abs(df2_left['y'][i] - df2_left['y'][i + 1]) > 10:
                    continue
                else:
                    df2_left['y'].loc[i + 1] = df2_left['y'][i]

            df3_left = df2_left.sort_values(by=['y', 'x'])

            left_side += '\n'.join(df3_left.groupby('y')['text'].apply(' '.join).values)
            return left_side
    except:
        logger.error('Getting issue in sorting the co-ordinates')


def get_info(path):
    try:
        logger.debug('Getting important info from Pan get_info()')
        text = text_layout(path)
        # print(text)
        if not text:
            logger.error('The provided input is not correct document get_info()')
            return None
        # DOB
        dob_pat = r'[0-9]{2}/[0-9]{2}/[0-9]{4}|[0-9]{2}-[0-9]{2}-[0-9]{4}|[0-9]{2}.[0-9]{2}.[0-9]{4}'
        dob = ''
        dob_mat = re.search(dob_pat, text)
        if dob_mat:
            dob += dob_mat.group()
        # PAN No
        pan_pat = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
        pan = ''
        pan_match = re.search(pan_pat, text)
        if pan_match:
            pan += pan_match.group()
        valid_match = match_address('Name', text)
        if valid_match:
            # Person Name and  Father Name
            person_name_extract = text.find(valid_match)
            father_name_extract = text.find("Father's Name")
            if person_name_extract != -1:
                person_name = [x for x in text[person_name_extract + len(valid_match):].split('\n') if x][0]
            else:
                return 'No name'
            if father_name_extract != -1:
                father_name = [x for x in text[father_name_extract + len("Father's Name"):].split('\n') if x][0]
            else:
                return 'No name'
        else:
            data = text
            end = data.find(dob)
            data = data[:end]
            data = data.split('\n')
            filter_data = list(filter(None, data))
            person_name = filter_data[-2]
            father_name = filter_data[-1]
        PersonalDetails = {
            'Name': person_name,
            'Father Name': father_name,
            'Dob': dob,
            'PAN No': pan
        }
        return PersonalDetails
    except:
        logger.error('Getting issue in getting information get_info()')


def pan_main(path):
    try:
        logger.debug('Upload Image is going inside pan_main()')
        li = set(['jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG', 'pdf', 'PDF'])
        files = os.listdir(path)
        for i in files:
            dir = i.split('.')[-1]
            if dir in li:
                result = get_info(path)
                if not result:
                    df = pd.DataFrame()
                    return df
                df = pd.DataFrame(list(result.items()), columns=['Field', 'Information'])
                return df
    except:
        logger.error('Getting issue after uploading the image pan_main()')
