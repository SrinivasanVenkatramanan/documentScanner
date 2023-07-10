import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import words
import PyPDF2
import re
from Levenshtein import distance
import glob, sys, fitz, cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import math
import shutil
import pandas as pd
import json
import warnings

warnings.simplefilter('ignore')
import os

os.environ['DOCTR_MULTIPROCESSING_DISABLE'] = 'TRUE'
os.environ['DOCTR_CACHE_DIR'] = 'weights/'
from logs import logger


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
        score = distance(keyword, word)
        # If the score is less than a certain threshold, consider it a match
        if score <= 3:
            return word
    return False


def name_check(text):
    if not text:
        return None
    nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            return name


def find_aadhar_number(ocr_text):
    """Function to find adhar number inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: 12 digit aadhaar number
    """
    if not ocr_text:
        return None
    adhar_number_patn = '\s[0-9]{4}\s[0-9]{4}\s[0-9]{4}|[0-9]{12}'
    mask_number_patn = '\sXXXX\sXXXX\s[0-9]{4}|xxxx\sxxxx\s[0-9]{4}'
    ocr_text = ocr_text.replace('\n', ' ')
    # for pattern in ocr_text:
    match = re.search(adhar_number_patn, ocr_text)
    if match:
        return match.group()
    mask_match = re.search(mask_number_patn, ocr_text)
    if mask_match:
        return mask_match.group()


def find_vid_number(ocr_text):
    """Function to find adhar vid number inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: 16 digit aadhaar number
    """
    if not ocr_text:
        return None
    adhar_number_patn = '\s[0-9]{4}\s[0-9]{4}\s[0-9]{4}\s[0-9]{4}|[0-9]{16}'
    mask_number_patn = '\sXXXX\sXXXX\s[0-9]{4}|xxxx\sxxxx\s[0-9]{4}'
    ocr_text = ocr_text.replace('\n', ' ')
    # for pattern in ocr_text:
    match = re.search(adhar_number_patn, ocr_text)
    if match:
        return match.group()
    mask_match = re.search(mask_number_patn, ocr_text)
    if mask_match:
        return mask_match.group()


def find_enroll_number(ocr_text):
    """Function to find adhar enroll number inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: 12 digit aadhaar number
    """
    if not ocr_text:
        return None
    adhar_number_patn = '[0-9]{4}/[0-9]{5}/[0-9]{5}'
    mask_number_patn = 'XXXX\sXXXX\s[0-9]{4}|xxxx\sxxxx\s[0-9]{4}'
    ocr_text = ocr_text.replace('\n', ' ')
    # for pattern in ocr_text:
    match = re.search(adhar_number_patn, ocr_text)
    if match:
        return match.group()
    mask_match = re.search(mask_number_patn, ocr_text)
    if mask_match:
        return mask_match.group()


def find_name(ocr_text):
    """Function to find adhar name inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: name on the aadhar card
    """
    if not ocr_text:
        return None

    start = ocr_text.find('To')
    if start != -1:
        lis = ocr_text.split('\n')
        for i, elem in enumerate(lis):
            if 'DOB' in elem:
                index = i
                break
        name = lis[index - 1]
        return name
    else:
        begin = ocr_text.find('Government of India')
        if begin != -1:
            index = ocr_text[begin:]
            end = ocr_text.find('DOB')
            index = index[:end]
            index = index.strip()
            index1 = index.split('\n')
            if len(index1) > 1:
                name = index1[-2]
                return name
            else:
                return None
        else:
            return 'No name matched'


def find_name_pdf(ocr_text):
    """Function to find adhar name inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: name on the aadhar card
    """
    if not ocr_text:
        return None
    start = ocr_text.find('To')
    if start != -1:
        index = ocr_text[start + len('To') + 1:]
        patn = r"S/O|D/O|W/O"
        address_end = 0
        loc = re.search(patn, index)
        if loc:
            address_end = loc.end()
        else:
            return 'Name is not matched'
        index = re.sub('\n', ' ', index[:address_end])
        index = index.strip()
        index = index.replace("\t", "").replace("  ", "")
        name = name_check(index)
        li = index.split(' ')
        if name is None:
            adhar_name_patn = r'[A-Za-z]+\s[A-Za-z]+|[A-Za-z]+\s[A-Za-z]+\s[A-za-z]+'
            match = re.search(adhar_name_patn, index)
            if match:
                return match.group()
        if name.strip() == li[-2] or name.strip() == li[-3] + ' ' + li[-2]:
            return name
        return name
    else:
        return 'No name matched'


def find_dob(ocr_text):
    """Function to find date of birth inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: Date of birth
    """
    if not ocr_text:
        return None
    dob_patn = '[0-9]{2}/[0-9]{2}/[0-9]{4}'
    ocr_text = ocr_text.replace('\n', " ")
    start = ocr_text.find('DOB')
    text = ocr_text[start:]
    dob = re.search(dob_patn, text)
    return dob.group()


def find_gender(ocr_text):
    """Function to find Gender inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: Gender
    """
    if not ocr_text:
        return None
    ocr_text = ocr_text.replace('\n', " ")
    GENDER = ''

    if ocr_text.find('MALE') != -1:
        GENDER += 'MALE'
    elif ocr_text.find('Male') != -1:
        GENDER += 'Male'
    elif ocr_text.find('FEMALE') != -1:
        GENDER += 'FEMALE'
    elif ocr_text.find('Female') != -1:
        GENDER += 'Female'
    return GENDER


def find_address(ocr_text):
    if not ocr_text:
        return None
    patn = r"S/O|D/O|W/O"
    loc = re.search(patn, ocr_text)
    if loc is not None:
        find = loc.group()
        start = loc.start()
        add = ocr_text[start + len(find) + 1:]
        add = str(add)
        pin_patn = r"[0-9]{6}"
        end_loc = re.search(pin_patn, add)
        end = end_loc.end()
        addr = add[:end]
        return addr
    else:
        new_text = ocr_text.split('\n')
        for i, elem in enumerate(new_text):
            if 'To' in elem:
                start = i
                break
        begin = start + 3
        starting_text = ' '.join(new_text[begin:])
        pin_patn = r"[0-9]{6}"
        end_loc = re.search(pin_patn, starting_text)
        end = end_loc.end()
        addr = starting_text[:end]
        return addr


def find_address_pdf(ocr_text):
    if not ocr_text:
        return None
    valid_address = match_address('Address', ocr_text)
    if valid_address:
        start = ocr_text.find(valid_address)
        address = ocr_text[start:]
        pinpatn = r'[0-9]{6}'
        address_end = 0
        pinloc = re.search(pinpatn, address)
        if pinloc:
            address_end = pinloc.end()
        else:
            print('Pin code not found in address')
        address = re.sub('\n', ' ', address[:address_end])
        address = address.strip()
        if valid_address == 'Address':
            loc = address.rfind('Address:')
            address = address[loc:].split(':')
            return address[-1].strip()
        else:
            loc = len(valid_address) + 1
            address = address[loc:].strip()
            return address
    else:
        return 'No Valid Address Found'


def find_mobile_number(ocr_text):
    if not ocr_text:
        return None
    ocr_text = ocr_text.replace('\n', " ")
    pattern = r'(^|\D)\d{10}($|\D)'
    match = re.search(pattern, ocr_text)
    if match:
        return match.group()
    else:
        return 'No Mobile Number Found'


def image_name_extraction(text):
    if not text:
        return None
    text = text.split("\n")
    name = [x for x in text if x != '']
    index = 0
    for i, data in enumerate(name):
        if 'government' in data.lower():
            index = i
            break
    return name[index + 2]


def image_dob_extraction(text):
    if not text:
        return None
    dob_patn = '[0-9]{2}/[0-9]{2}/[0-9]{4}'
    yob_patn = '[0-9]{4}'
    DateOfBirth = ''
    text = text.split('\n')
    for dob in text:
        if 'DOB' in dob:
            match = re.search(dob_patn, dob)
            DateOfBirth = match.group()
        if 'Year of Birth' in dob:
            match = re.search(yob_patn, dob)
            DateOfBirth = match.group()
    return DateOfBirth


def image_gender_extraction(text):
    """Function to find Gender inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: Gender
    """
    if not text:
        return None
    text = text.split("\n")
    GENDER = ''
    for gender in text:
        if gender.find('MALE') != -1:
            GENDER += 'MALE'
        elif gender.find('Male') != -1:
            GENDER += 'Male'
        elif gender.find('FEMALE') != -1:
            GENDER += 'FEMALE'
        elif gender.find('Female') != -1:
            GENDER += 'Female'
    return GENDER


def image_aadnum_extraction(text):
    """Function to find adhar number inside the image
    Args:
    ocr_text (list): text from the ocr
    Returns:
    str: 12 digit aadhaar number
    """
    if not text:
        return None
    adhar_number_patn = '[0-9]{4}\s[0-9]{4}\s[0-9]{4}'
    mask_number_patn = 'XXXX\sXXXX\s[0-9]{4}|xxxx\sxxxx\s[0-9]{4}'
    match = re.search(adhar_number_patn, text)
    if match:
        return match.group()
    mask_match = re.search(mask_number_patn, text)
    if mask_match:
        return mask_match.group()


def image_address_extraction(text):
    if not text:
        return None
    valid_address = match_address('Address', text)
    if valid_address:
        start = text.find(valid_address)
        address = text[start:]
        pinpatn = r'[0-9]{6}'
        address_end = 0
        pinloc = re.search(pinpatn, address)
        if pinloc:
            address_end = pinloc.end()
        else:
            print('Pin code not found in address')
        address = re.sub('\n', ' ', address[:address_end])
        address = address.strip()
        if valid_address == 'Address':
            loc = address.rfind('Address:')
            address = address[loc:].split(':')
            address = address[-1].strip()
            address = address.replace("\t", " ")
            return address
        else:
            loc = len(valid_address) + 1
            address = address[loc:].strip()
            address = address.replace("\t", " ")
            return address
    else:
        return 'No Valid Address Found'


def extract_info(path):
    try:
        logger.debug('Extracts the document information extract_info()')
        dir = os.listdir(path)
        reader = PyPDF2.PdfReader(path + '/' + dir[0])
        data = ''
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            iter = page.extract_text()
            data += iter
        check = data.lower().__contains__('account')
        check1 = data.lower().__contains__('income')
        if check or check1:
            logger.error('The provided input is not correct document extract_info()')
            return None
        if 'To' not in data and 'Enrolment Number' not in data:
            front_text, back_text = text_extraction(load_model(), convert_image(path))
            name = image_name_extraction(front_text)
            dob = image_dob_extraction(front_text)
            gender = image_gender_extraction(front_text)
            aad_num = image_aadnum_extraction(front_text)
            address = image_address_extraction(back_text)
            PersonDetails = {"Name": name,
                             "DOB": dob,
                             "Gender": gender,
                             "Aadhar Number": aad_num,
                             "Address": address,
                             }
            return PersonDetails
        else:
            user_name = find_name_pdf(data)
            date_of_birth = find_dob(data)
            gender = find_gender(data)
            mobile_number = find_mobile_number(data)
            address = find_address_pdf(data)
            aadhar_number = find_aadhar_number(data)
            vid_number = find_vid_number(data)
            if not vid_number:
                vid_number = 'No VID Number Found'
            enroll_number = find_enroll_number(data)
            PersonDetails = {"Name": user_name,
                             "DOB": date_of_birth,
                             "Gender": gender,
                             "Mobile Number": mobile_number,
                             "Address": address,
                             "Aadhar Number": aadhar_number,
                             "VID": vid_number,
                             "Enroll Number": enroll_number}
            return PersonDetails
    except:
        logger.error('Getting issue in extract_info()')


def text_layout(output):
    try:
        logger.debug('Sorting the text based on co-ordinates text_layout()')
        if not output:
            logger.error('The provided input is not correct document text_layout()')
            return None
        result = get_coordinates(output)
        x, w, y, h, text = [], [], [], [], []
        df_left = pd.DataFrame(columns=['x', 'w', 'y', 'h', 'text'])
        xl = list(zip(*list(zip(*result))[0]))[0]
        wl = list(zip(*list(zip(*result))[0]))[1]
        yl = list(zip(*list(zip(*result))[0]))[2]
        hl = list(zip(*list(zip(*result))[0]))[3]
        tl = list(zip(*result))[1]
        if 'To' not in tl and 'Enrolment Number' not in tl:
            df_left['x'] = xl
            df_left['w'] = wl
            df_left['y'] = yl
            df_left['h'] = hl
            df_left['text'] = tl
        else:
            ind = [i for i, item in enumerate(list(zip(*result))[1]) if re.search("[0-9]{4}/[0-9]{5}/[0-9]{5}", item)][
                0]
            thresh = list(zip(*list(zip(*result))[0]))[1][ind]
            mi = [j for j, i in enumerate(list(zip(*list(zip(*result))[0]))[1]) if i <= thresh]
            df_left['x'] = list(map(lambda x: xl[x], mi))
            df_left['w'] = list(map(lambda x: wl[x], mi))
            df_left['y'] = list(map(lambda x: yl[x], mi))
            df_left['h'] = list(map(lambda x: hl[x], mi))
            df_left['text'] = list(map(lambda x: tl[x], mi))
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
        logger.error('Getting issue in text_layout()')


def extract_scan_info(path):
    try:
        logger.debug('Extracts scanned document information')
        dir = os.listdir(path)
        model = load_model()
        if dir[0].endswith('.pdf'):
            img = DocumentFile.from_pdf(path + '/' + dir[0])
        else:
            img = DocumentFile.from_images(path + '/' + dir[0])
        result = model(img)
        output = result.export()
        data = text_layout(output)
        check = data.lower().__contains__('account')
        check1 = data.lower().__contains__('income')
        if check or check1:
            logger.error('The provided input is not correct document scan_info()')
            return None
        if 'To' not in data and 'Enrolment Number' not in data:
            front_text, back_text = text_extraction(model, convert_image(path))
            name = image_name_extraction(front_text)
            dob = image_dob_extraction(front_text)
            gender = image_gender_extraction(front_text)
            aad_num = image_aadnum_extraction(front_text)
            address = image_address_extraction(back_text)
            PersonDetails = {"Name": name,
                             "DOB": dob,
                             "Gender": gender,
                             "Aadhar Number": aad_num,
                             "Address": address,
                             }
            return PersonDetails
        else:
            user_name = find_name(data)
            date_of_birth = find_dob(data)
            gender = find_gender(data)
            mobile_number = find_mobile_number(data)
            address = find_address(data)
            aadhar_number = find_aadhar_number(data)
            vid_number = find_vid_number(data)
            if not vid_number:
                vid_number = 'No VID Number Found'
            enroll_number = find_enroll_number(data)
            PersonDetails = {"Name": user_name,
                             "DOB": date_of_birth,
                             "Gender": gender,
                             "Mobile Number": mobile_number,
                             "Address": address,
                             "Aadhar Number": aadhar_number,
                             "VID": vid_number,
                             "Enroll Number": enroll_number}
            return PersonDetails
    except:
        logger.error('Getting issue in scan_info()')


def convert_image(path):
    try:
        logger.debug('Convert Scanned Pdf to images convert_image()')
        dir = os.listdir(path)
        full_path = path + '/' + dir[0]
        name = dir[0].split('.')[0]
        doc = fitz.open(full_path)
        # To get better resolution
        zoom_x = 2.0  # horizontal zoom
        zoom_y = 2.0  # vertical zoom
        mat = fitz.Matrix(zoom_x, zoom_y)
        img_dir = "document/images/"
        for page in doc:  # iterate through the pages
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            pix.save(f"{img_dir}{name}-%i.png" % page.number)
        return img_dir
    except:
        logger.error('Getting issue in converting to image converting_image()')


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
        for i in range(2):
            page_dim = output['pages'][i]["dimensions"]
            text_coordinates = []
            for obj1 in output['pages'][i]["blocks"]:
                for obj2 in obj1["lines"]:
                    for obj3 in obj2["words"]:
                        converted_coordinates = convert_coordinates(
                            obj3["geometry"], page_dim
                        )
                        text_coordinates.append([converted_coordinates, obj3["value"]])
            return text_coordinates
    except:
        logger.error('Getting issue in getting the coordinates from text get_coordinates()')


def text_extraction(model, path):
    try:
        logger.debug('Extracting horizontal and vertical based text for address text_extraction()')
        lis_dir = os.listdir(path)
        lis_dir = sorted(lis_dir)
        data = ''
        if len(lis_dir) == 2:
            for i in range(len(lis_dir)):
                img = DocumentFile.from_images(path + '/' + lis_dir[i])
                result = model(img)
                output = result.export()
                data += text_layout(output)

            check = data.lower().__contains__('account')
            check1 = data.lower().__contains__('income')
            if check or check1:
                logger.error('The provided input is not correct document text_extraction()')
                return None, None
            if 'To' not in data and 'Enrolment Number' not in data:
                graphical_coordinates = get_coordinates(output)
                back = ""
                cord = []
                for i in graphical_coordinates:
                    if i[1] == 'Address:':
                        if i[0][0] < 300:
                            cord.append(i[0][0] - 50)
                        else:
                            cord.append(i[0][0] - 100)
                for j in graphical_coordinates:
                    if cord[0] < int(j[0][0]):
                        back += " " + j[1]
                return data, back
            else:
                return data, []
    except:
        logger.error('Getting issue in text_extraction()')


def image_extraction(path):
    try:
        logger.debug('Extracting text from images image_extraction()')
        model = load_model()
        dir = os.listdir(path)
        if len(dir) == 2:
            front_text, back_text = text_extraction(model, path)
            if not front_text and not back_text:
                logger.error('The provided input is not correct document image_extraction()')
                return None
            name = image_name_extraction(front_text)
            dob = image_dob_extraction(front_text)
            gender = image_gender_extraction(front_text)
            aad_num = image_aadnum_extraction(front_text)
            address = image_address_extraction(back_text)
            PersonDetails = {"Name": name,
                             "DOB": dob,
                             "Gender": gender,
                             "Aadhar Number": aad_num,
                             "Address": address,
                             }
            return PersonDetails
        else:
            data = text_extraction(model, path)
            if not data:
                return None
            user_name = find_name(data)
            date_of_birth = find_dob(data)
            gender = find_gender(data)
            mobile_number = find_mobile_number(data)
            address = find_address(data)
            aadhar_number = find_aadhar_number(data)
            vid_number = find_vid_number(data)
            enroll_number = find_enroll_number(data)
            PersonDetails = {"Name": user_name,
                             "DOB": date_of_birth,
                             "Gender": gender,
                             "Mobile Number": mobile_number,
                             "Address": address,
                             "Aadhar Number": aadhar_number,
                             "VID": vid_number,
                             "Enroll Number": enroll_number}
            return PersonDetails
    except:
        logger.error('Getting issue in image_extraction()')


def clearity_check(path):
    try:
        logger.info('Check for clarity of documents clearity_check()')
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
        # Open the pdf file
        image = cv2.imread("temp/page-0.png", cv2.IMREAD_UNCHANGED)
        # Get the resolution of the image
        resolution = image.shape[0] * image.shape[1]
        # Check the image quality
        quality = cv2.Laplacian(image, cv2.CV_64F).var()
        shutil.rmtree('temp/')
        if resolution > 2000000 and quality > 2500:
            return extract_info(path)
        else:
            return extract_scan_info(path)
    except:
        logger.error('Getting issue in clearity_check()')


def aadhar_main(path):
    try:
        logger.debug('Upload Image is going inside aadhar_main()')
        li = set(['jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG'])
        files = os.listdir(path)
        clear_path = 'document/images/'
        clear_dir = os.listdir(clear_path)
        for j in clear_dir:
            os.remove(clear_path + j)
        for i in files:
            dir = i.split('.')[-1]
            if dir in li:
                if len(files) > 1:
                    result = image_extraction(path)
                    if not result:
                        df = pd.DataFrame()
                        return df
                    df = pd.DataFrame(list(result.items()), columns=['Field', 'Information'])
                    return df
                else:
                    result = extract_scan_info(path)
                    if not result:
                        df = pd.DataFrame()
                        return df
                    df = pd.DataFrame(list(result.items()), columns=['Field', 'Information'])
                    return df
            else:
                result = clearity_check(path)
                if not result:
                    df = pd.DataFrame()
                    return df
                df = pd.DataFrame(list(result.items()), columns=['Field', 'Information'])
                return df

    except:
        logger.error('Getting issue after uploading the image aadhar_main()')
